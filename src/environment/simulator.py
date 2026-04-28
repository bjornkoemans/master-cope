import math
import os
from gymnasium.spaces import Discrete, MultiBinary, Box, Dict as GymDict
from pettingzoo import ParallelEnv  # type: ignore
import numpy as np

from typing import TypedDict, Optional, Mapping, Dict, List
from collections import Counter
import pandas as pd

# Pre-computed log constants (used in observation normalization)
_LOG1P_60 = math.log1p(60.0)
_LOG1P_100 = math.log1p(100.0)
_LOG1P_660 = math.log1p(660.0)

from .config import debug_print_colored

from .reward import get_reward
from .entities import Case, Task, ResourceAgent, Status
from .display import display_indented_list
from .data_handling import (
    compute_activity_duration_distribution_per_agent,
    compute_global_activity_medians,
)
from .duration_distribution import DurationDistribution
from .work_schedule import adjust_completion_time, is_within_work_hours, next_work_start, configure as configure_work_schedule


class SimulationParameters(TypedDict):
    start_timestamp: pd.Timestamp


class AgentOptimizerEnvironment(ParallelEnv):
    """The environment representing the business process."""

    metadata = {
        "name": "agent_optimizer_environment_v0",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        simulation_parameters: SimulationParameters,
        experiment_dir: str | None = None,
        enable_logging: bool = True,
        verbose: bool = True,
        pre_fitted_distributions: Optional[
            tuple[
                Mapping[str, Mapping[str, Optional[DurationDistribution]]],
                Mapping[str, Mapping[str, Optional[Dict[str, float]]]],
                Dict[str, float],
            ]
        ] = None,
        ground_truth_replay: bool = False,
        max_steps: int = 100_000,
        max_episodes: int = 1000,
        work_schedule_enabled: bool = False,
        work_start_hour: int = 8,
        work_end_hour: int = 20,
        reward_config: dict | None = None,
        use_agent_identity: bool = True,
        parallel_task_groups: list[list[str]] | None = None,
        fixed_agent_list: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.ground_truth_replay: bool = ground_truth_replay
        self.work_schedule_enabled: bool = work_schedule_enabled
        self.work_start_hour: int = work_start_hour
        self.work_end_hour: int = work_end_hour
        if work_schedule_enabled:
            configure_work_schedule(work_start_hour, work_end_hour)
        self.use_agent_identity: bool = use_agent_identity
        self._parallel_task_groups_config = parallel_task_groups  # Store raw config

        # Reward configuration (configurable weights and values)
        from .reward import DEFAULT_REWARD_CONFIG
        self.reward_config = {**DEFAULT_REWARD_CONFIG, **(reward_config or {})}
        self._fallback_this_step: bool = False
        self._volunteer_ids_this_step: list = []
        self._assigned_agent_this_step: int | None = None
        self._last_assigned_agent_id: int | None = None
        self._step_had_task: bool = False
        self.verbose: bool = verbose
        if verbose:
            print("Initializing environment...")
        self.data: pd.DataFrame = data
        self.enable_logging: bool = enable_logging

        # Set up logging directory
        if enable_logging:
            if experiment_dir:
                self.log_dir = os.path.join(experiment_dir, "logs")
            else:
                self.log_dir = "data/logs"

            # check that log_dir exists
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            current_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            self.log_file: str = os.path.join(self.log_dir, f"log_{current_timestamp}.csv")
        else:
            self.log_dir = None
            self.log_file = None

        # Log buffer for batch CSV writing (avoids per-task pd.DataFrame + file I/O)
        self._log_buffer: list = []
        self._log_columns = [
            "case_id", "case_nr_tasks", "case_open_time", "case_completed_time",
            "task_id", "task_name", "task_assigned_time", "task_started_time",
            "task_completed_time", "task_agent_id", "task_agent_name", "task_agents_required",
            "task_assignment_type", "task_volunteers",
        ]

        # Total number of steps and epochs
        self.steps: int = 0
        self.epochs: int = 0
        self.max_steps: int = max_steps
        self.max_episodes: int = max_episodes

        # Initialize the simulation time and event queue
        self.num_activities: int = len(set(self.data["activity_name"]))
        self.task_dict: dict[str, int] = {
            task: i for i, task in enumerate(sorted(set(self.data["activity_name"])))
        }
        self.inv_task_dict: dict[int, str] = {
            i: task for i, task in enumerate(sorted(set(self.data["activity_name"])))
        }
        # Convert parallel_task_groups from activity names to task IDs
        self._parallel_group_ids: list[set[int]] | None = None
        if parallel_task_groups:
            groups = []
            for group_names in parallel_task_groups:
                ids = {self.task_dict[name] for name in group_names if name in self.task_dict}
                if len(ids) > 1:
                    groups.append(ids)
            self._parallel_group_ids = groups if groups else None
            if self.verbose and self._parallel_group_ids:
                for g in self._parallel_group_ids:
                    names = [self.inv_task_dict[tid] for tid in g]
                    print(f"  Parallel group: {names}")

        self.current_time: pd.Timestamp = simulation_parameters["start_timestamp"]
        self.future_cases: list[Case] = self._initialize_future_cases()
        self.pending_cases: list[Case] = []
        self.completed_cases: list[Case] = []
        self.upcoming_case: Case | None = None
        self.completed_task: Task | None = None

        # Initialize the agents
        # Use fixed_agent_list if provided (ensures eval env has same agents as train env)
        if fixed_agent_list is not None:
            self.resources: list[str] = fixed_agent_list
        else:
            self.resources: list[str] = sorted(set(self.data["resource"]))
        self.resource_dict: dict[str, int] = {
            resource: i for i, resource in enumerate(self.resources)
        }

        # Fit distributions for each agent and activity
        if pre_fitted_distributions is not None:
            # Use pre-fitted distributions (fitted on training data)
            activity_durations_dict, stats_dict, global_activity_medians = (
                pre_fitted_distributions
            )
            if self.verbose:
                print("Using pre-fitted duration distributions from training data")
        else:
            # Fit distributions on the current data (original behavior)
            activity_durations_dict, stats_dict = (
                compute_activity_duration_distribution_per_agent(self.data)
            )
            # Compute global historical medians for each activity (across all agents)
            global_activity_medians = compute_global_activity_medians(self.data)
            if self.verbose:
                print("Fitting duration distributions on current dataset")

        # Store global activity medians
        self.global_activity_medians = global_activity_medians

        # Cache distributions to avoid recomputing in reset()
        self._cached_activity_durations = activity_durations_dict
        self._cached_stats_dict = stats_dict

        # Transform the global medians to use task IDs instead of activity names
        self.global_task_medians = {
            self.task_dict[activity]: median
            for activity, median in self.global_activity_medians.items()
        }

        # Transform the distributions to use task IDs instead of activity names
        transformed_activity_durations_dict = {
            self.resource_dict[resource]: {
                self.task_dict[task]: activity_durations_dict[resource][task]
                for task in sorted(set(self.data["activity_name"]))
            }
            for resource in self.resources
        }

        self.agents: list[ResourceAgent] = [
            ResourceAgent(
                self.resource_dict[resource],
                name=resource,
                capabilities={
                    self.task_dict[task]: transformed_activity_durations_dict[
                        self.resource_dict[resource]
                    ][self.task_dict[task]]
                    for task in sorted(set(self.data["activity_name"]))
                },
                stats_dict=stats_dict[resource],  # type: ignore
            )
            for resource in self.resources
        ]

        # Build role lookup for observations (agent_id one-hot + role one-hot)
        self.unique_roles: list[str] = sorted(set(a.role for a in self.agents))
        self.role_to_idx: dict[str, int] = {r: i for i, r in enumerate(self.unique_roles)}
        self.n_agents = len(self.agents)
        self.n_roles = len(self.unique_roles)

        # Pre-compute per-agent one-hot vectors (static, never change)
        self._agent_id_onehots: dict[int, np.ndarray] = {}
        self._agent_role_onehots: dict[int, np.ndarray] = {}
        for agent in self.agents:
            id_oh = np.zeros(self.n_agents, dtype=np.float32)
            id_oh[agent.id] = 1.0
            self._agent_id_onehots[agent.id] = id_oh
            role_oh = np.zeros(self.n_roles, dtype=np.float32)
            role_oh[self.role_to_idx[agent.role]] = 1.0
            self._agent_role_onehots[agent.id] = role_oh

        # Set environment reference for all tasks and cases
        for case in self.future_cases:
            case.environment = self
            for task in case.all_tasks:
                task.environment = self

        # Cache case data for fast reset (avoid expensive pandas groupby)
        self._cached_case_data = [
            {
                'case_id': case.id,
                'assigned_timestamp': case.assigned_timestamp,
                'tasks': [
                    {
                        'id': task.id,
                        'required_roles': list(task.required_roles),
                        'ground_truth_resource': task.ground_truth_resource,
                    }
                    for task in case.all_tasks
                ],
                'parallel_groups': [set(g) for g in case._parallel_groups] if case._parallel_groups else None,
            }
            for case in self.future_cases
        ]

        if self.verbose:
            print(f"Environment initialized. Start time: {self.current_time}")
            print(f"Max steps: {self.max_steps:,} | Max episodes: {self.max_episodes:,}")
            display_indented_list(self.agents, "Agents")
            print(f"# future cases: {len(self.future_cases)}")
            print(f"# tasks to be performed: {len(self.data)}")
            print("---------" * 8)

    def resource_name_to_id(self, resource_name: str) -> int:
        """Convert a resource name to its corresponding ID."""
        if resource_name in self.resource_dict:
            return self.resource_dict[resource_name]
        else:
            raise ValueError(
                f"Resource '{resource_name}' not found in resource dictionary."
            )

    def _initialize_future_cases(self) -> list[Case]:
        """Function that initializes the future cases with the first event of each case in the data."""
        future_cases: list[Case] = []
        has_required_roles = "required_roles" in self.data.columns
        # Group the data by case_id, and iterate over each case
        grouped_and_sorted = self.data.sort_values("start_timestamp").groupby("case_id")
        has_assign_ts = "assign_timestamp" in self.data.columns
        for case_id, case_data in grouped_and_sorted:
            # Case arrival time: use assign_timestamp (queue entry) if available,
            # otherwise fall back to earliest start_timestamp
            if has_assign_ts:
                start_timestamp = case_data["assign_timestamp"].min()
            else:
                start_timestamp = case_data["start_timestamp"].min()
            rows = case_data.sort_values("start_timestamp")
            tasks = []
            for _, row in rows.iterrows():
                task_id = self.task_dict[row["activity_name"]]
                # Derive required_roles from data; agents_required is implicit from len(required_roles)
                req_roles = (
                    row["required_roles"].split(",") if has_required_roles and pd.notna(row.get("required_roles", None)) and row.get("required_roles", "") else []
                )
                t = Task(
                    task_id,
                    int(str(case_id)),
                    required_roles=req_roles,
                )
                t.ground_truth_resource = row["resource"]
                tasks.append(t)
            # Compute parallel groups for this case (indices into tasks list)
            case_parallel = self._compute_case_parallel_groups(tasks)

            case = Case(
                int(str(case_id)),
                start_timestamp,
                tasks,
                parallel_groups=case_parallel,
            )
            case.environment = self  # Set environment reference
            for task in case.all_tasks:
                task.environment = self  # Set environment reference for all tasks

            future_cases.append(case)

        future_cases.sort(key=lambda x: x.assigned_timestamp)

        return future_cases

    def _compute_case_parallel_groups(self, tasks: list[Task]) -> list[set] | None:
        """Compute parallel group indices for a specific case's tasks.

        Maps the global parallel_group_ids (task_id based) to indices
        within this case's task list. Only returns groups with 2+ tasks.
        """
        if not self._parallel_group_ids:
            return None
        case_groups = []
        for group_ids in self._parallel_group_ids:
            indices = set()
            for idx, task in enumerate(tasks):
                if task.id in group_ids:
                    indices.add(idx)
            if len(indices) > 1:
                case_groups.append(indices)
        return case_groups if case_groups else None

    def _select_collaborative_agents(
        self, task: Task, actions: dict[int, int]
    ) -> tuple[List[ResourceAgent], int, int]:
        """Select agents for a collaborative task based on required_roles.

        For each required role, randomly pick a volunteer with that role.
        If not enough volunteers for a role, fall back to a random agent
        with the correct role from all agents.

        Returns:
            Tuple of (selected agents, number of agents matched from volunteers,
            total number of role-eligible volunteers).
        """
        required_roles = list(task.required_roles)
        role_set = set(required_roles)
        has_wildcard = "*" in role_set

        # For collaborative tasks, volunteers are filtered by ROLE match
        # (not capability), so agents without historical data for this task
        # can still participate as helpers. The primary agent (who has the
        # capability) provides the duration distribution.
        # Wildcard role "*" matches any agent (useful when all agents share
        # generic roles, e.g. BPIC12 where each agent has a unique role).
        volunteers = [
            self.agents[agent_id]
            for agent_id, action in actions.items()
            if action == 1 and (has_wildcard or self.agents[agent_id].role in role_set)
        ]

        # Count total volunteers that match any required role
        total_role_volunteers = len(volunteers)

        selected: List[ResourceAgent] = []
        remaining_roles = list(required_roles)
        num_from_volunteers = 0

        # First pass: for each required role, randomly pick from matching volunteers
        for role in list(remaining_roles):
            if role == "*":
                role_volunteers = [
                    v for v in volunteers if v not in selected
                ]
            else:
                role_volunteers = [
                    v for v in volunteers if v.role == role and v not in selected
                ]
            if role_volunteers:
                chosen = role_volunteers[np.random.randint(len(role_volunteers))]
                selected.append(chosen)
                remaining_roles.remove(role)
                num_from_volunteers += 1

        # Second pass: fill remaining roles randomly from all agents with that role
        for role in remaining_roles:
            if role == "*":
                candidates = [
                    agent for agent in self.agents
                    if agent not in selected
                ]
            else:
                candidates = [
                    agent for agent in self.agents
                    if agent.role == role and agent not in selected
                ]
            if candidates:
                chosen = candidates[np.random.randint(len(candidates))]
                selected.append(chosen)

        return selected, num_from_volunteers, total_role_volunteers

    def _find_upcoming_case(self) -> Case | None:
        """Find the next case that needs a task assignment at the current time.

        Checks pending cases for eligible next-tasks first (e.g. a task just
        completed and the next task in that case needs assignment). Then checks
        if any future cases have arrived at or before the current time.

        Returns None if nothing needs attention at the current timestamp.
        """
        current_time = self.current_time

        # Move all future cases that have arrived at or before current_time into pending
        future_cases = self.future_cases
        pending_cases = self.pending_cases
        while future_cases and future_cases[0].assigned_timestamp <= current_time:
            arrived = future_cases.pop(0)
            pending_cases.append(arrived)

        # Build set of cases already being handled by agents
        # Cases with pending parallel tasks are NOT added to handled_cases
        # so they can be offered again for additional parallel assignments.
        handled_cases: set[Case] = set()
        for agent in self.agents:
            cc = agent.current_case
            if cc is not None:
                if not (cc.is_in_parallel_group and cc.is_eligible_for_next_task(current_time)):
                    handled_cases.add(cc)
            # Iterate deque directly instead of peek(i) calls
            for case in agent.case_queue.queue:
                if not (case.is_in_parallel_group and case.is_eligible_for_next_task(current_time)):
                    handled_cases.add(case)

        # Find first pending case that is ready and not already handled
        for case in pending_cases:
            if case not in handled_cases and case.is_eligible_for_next_task(current_time):
                return case

        return None

    def _filter_completed_cases(self) -> None:
        """Filter out completed cases from pending_cases and move them to completed_cases."""
        i = 0
        while i < len(self.pending_cases):
            case = self.pending_cases[i]
            if case.status == Status.COMPLETED:
                self.completed_cases.append(case)
                self.pending_cases.pop(i)
            else:
                i += 1

        debug_print_colored(
            f"Completed: {len(self.completed_cases)}, remaining: {len(self.pending_cases)}"
        )
        if self.pending_cases:
            debug_print_colored(f"Remaining case: {self.pending_cases[0]}")
        else:
            debug_print_colored("No remaining cases")

    def step(self, actions: dict[int, int]) -> tuple[dict, dict, dict, dict, dict]:
        """Execute one step of the environment's dynamics."""
        self.steps += 1
        self.completed_task = None  # Reset so stale tasks don't re-trigger rewards
        self._step_had_task = (self.upcoming_case is not None and self.upcoming_case.current_task is not None)

        ### -------------------------------- ###
        ### HANDLE ACTION FROM CURRENT STEP  ###
        ### -------------------------------- ###
        if (
            self.upcoming_case is not None
            and self.upcoming_case.current_task is not None
        ):
            task = self.upcoming_case.current_task

            # --- GROUND TRUTH REPLAY MODE ---
            # If the task has a ground_truth_resource and replay mode is on,
            # skip the volunteer mechanism and assign directly.
            if self.ground_truth_replay and task.ground_truth_resource:
                try:
                    gt_agent_id = self.resource_name_to_id(task.ground_truth_resource)
                    selected_agent = self.agents[gt_agent_id]
                except (ValueError, IndexError):
                    # Resource not found (e.g. not in this split) — fall back to
                    # random capable agent
                    capable = [a for a in self.agents if a.can_perform_task(task.id)]
                    selected_agent = capable[0] if capable else self.agents[0]
                task.volunteer_ids = [selected_agent.id]
                task.assignment_type = "ground_truth"
                self.upcoming_case.assign_to_agent(selected_agent, self.current_time)
                self.upcoming_case = None
            else:
                # --- NORMAL VOLUNTEER MECHANISM ---
                # Record which *capable* agents volunteered (action == 1)
                task.volunteer_ids = [
                    agent_id for agent_id, action in actions.items()
                    if action == 1 and self.agents[agent_id].can_perform_task(task.id)
                ]

                if task.agents_required > 1 and task.required_roles:
                    # --- COLLABORATIVE TASK ASSIGNMENT ---
                    selected_agents, num_from_volunteers, total_role_volunteers = self._select_collaborative_agents(
                        task, actions
                    )
                    if selected_agents:
                        # Determine primary agent (one with historical data for this task)
                        primary = next(
                            (a for a in selected_agents if a.can_perform_task(task.id)),
                            selected_agents[0]
                        )
                        # Track assigned primary agent for reward computation
                        self._last_assigned_agent_id = primary.id
                        # Determine assignment type
                        if num_from_volunteers == 0:
                            task.assignment_type = "collab_fallback_random"
                        elif num_from_volunteers == len(selected_agents):
                            if total_role_volunteers == len(selected_agents):
                                task.assignment_type = "collab_volunteer"
                            else:
                                task.assignment_type = "collab_volunteer_all_random"
                        else:
                            task.assignment_type = "collab_volunteer_partial_random"
                        debug_print_colored(
                            f"Collaborative task {task.id}: agents {[a.id for a in selected_agents]}, primary={primary.id}"
                        )
                        self.upcoming_case.assign_to_agents(
                            selected_agents, primary, self.current_time
                        )
                        self.upcoming_case = None
                    else:
                        # Collaborative selection failed — fall back to single agent
                        debug_print_colored(
                            f"Collaborative selection failed for task {task.id}, falling back to single agent",
                            "yellow",
                        )
                        fallback_agents = [a for a in self.agents if a.can_perform_task(task.id)]
                        if not fallback_agents:
                            fallback_agents = list(self.agents)
                        selected_agent = fallback_agents[0]
                        # Track assigned agent for reward computation
                        self._last_assigned_agent_id = selected_agent.id
                        task.assignment_type = "fallback_no_collab"
                        self.upcoming_case.assign_to_agent(selected_agent, self.current_time)
                        self.upcoming_case = None
                else:
                    # --- SINGLE AGENT ASSIGNMENT ---
                    capable_volunteers = [
                        agent_id for agent_id, action in actions.items() if action == 1
                    ]
                    capable_volunteers = [
                        agent_id
                        for agent_id in capable_volunteers
                        if self.agents[agent_id].can_perform_task(task.id)
                    ]
                    if capable_volunteers:
                        available_agents = capable_volunteers
                        if len(capable_volunteers) == 1:
                            task.assignment_type = "solo_volunteer"
                        else:
                            task.assignment_type = "solo_volunteer_random"
                    else:
                        available_agents = [
                            agent.id
                            for agent in self.agents
                            if agent.can_perform_task(task.id)
                        ]
                        if not available_agents:
                            # No agent is capable — fall back to any agent
                            available_agents = [agent.id for agent in self.agents]
                            task.assignment_type = "fallback_no_capable"
                        else:
                            task.assignment_type = "solo_fallback_random"

                    selected_agent_id = np.random.choice(available_agents)
                    selected_agent = self.agents[selected_agent_id]
                    # Track assigned agent for reward computation
                    self._last_assigned_agent_id = selected_agent.id

                    debug_print_colored(f"Upcoming case: {self.upcoming_case}")
                    self.upcoming_case.assign_to_agent(selected_agent, self.current_time)
                    self.upcoming_case = None

                # Flag fallback for reward function
                if task.assignment_type and not task.assignment_type.startswith(('solo_volunteer', 'collab_volunteer')):
                    self._fallback_this_step = True
                    # Track which agents could have volunteered (for individual fallback penalty)
                    self._capable_agent_ids_this_step = [
                        a.id for a in self.agents if a.can_perform_task(task.id)
                    ] or [a.id for a in self.agents]

                # Track for R_volunteer and R_assignment reward components
                self._volunteer_ids_this_step = getattr(task, 'volunteer_ids', []) or []
                # Track assigned agent (set by single-agent or collaborative paths above)
                self._assigned_agent_this_step = self._last_assigned_agent_id

        ### ------------------------------- ###
        ### CHECK COMPLETED TASKS/CASES     ###
        ### ------------------------------- ###
        debug_print_colored(f"Active cases: {len(self.pending_cases)}")
        for agent in self.agents:
            debug_print_colored(agent, "yellow")
        for agent in self.agents:
            is_finished, finished_case = agent.work_case(self.current_time)
            if is_finished and finished_case:
                if finished_case.current_task:
                    self.completed_task = finished_case.current_task
                # Buffer completed task rows for batch CSV writing
                if self.enable_logging and self.log_file:
                    for task in finished_case.all_tasks:
                        # For collaborative tasks, log all agent IDs/names
                        # Use pipe separator to avoid breaking CSV parsing
                        if task.assigned_agents:
                            agent_ids = "|".join(str(a.id) for a in task.assigned_agents)
                            agent_names = "|".join(a.name for a in task.assigned_agents)
                        elif task.assigned_agent:
                            agent_ids = str(task.assigned_agent.id)
                            agent_names = task.assigned_agent.name
                        else:
                            agent_ids = ""
                            agent_names = ""

                        # Resolve task name from inv_task_dict
                        task_name = self.inv_task_dict.get(task.id, str(task.id))

                        # Volunteer IDs as pipe-separated string
                        volunteer_str = "|".join(str(v) for v in task.volunteer_ids) if task.volunteer_ids else ""

                        self._log_buffer.append([
                            finished_case.id,
                            len(finished_case.all_tasks),
                            finished_case.assigned_timestamp,
                            finished_case.completion_timestamp,
                            task.id,
                            task_name,
                            task.assigned_timestamp,
                            task.start_timestamp,
                            task.completion_timestamp,
                            agent_ids,
                            agent_names,
                            task.agents_required,
                            task.assignment_type,
                            volunteer_str,
                        ])

                    # Flush buffer every 500 rows to avoid memory buildup
                    if len(self._log_buffer) >= 500:
                        self._flush_log_buffer()
            if agent.busy_until and agent.busy_until <= self.current_time:
                agent.busy_until = None

        # ── Second pass: resolve collaborative sync missed due to iteration order ──
        # When the first pass iterates agents alphabetically (Pharmacist before
        # Technician), a waiting-for-collab agent may check sync BEFORE its
        # partner has dequeued the shared case.  Re-check those agents now so the
        # collaborative task can start at the correct timestamp instead of being
        # delayed until the next time-advance.
        for agent in self.agents:
            if not agent.waiting_for_collab or agent.current_case is None:
                continue
            is_finished, finished_case = agent.work_case(self.current_time)
            if is_finished and finished_case:
                if finished_case.current_task:
                    self.completed_task = finished_case.current_task
                # Buffer completed task rows for batch CSV writing
                if self.enable_logging and self.log_file:
                    for task in finished_case.all_tasks:
                        if task.assigned_agents:
                            agent_ids = "|".join(str(a.id) for a in task.assigned_agents)
                            agent_names = "|".join(a.name for a in task.assigned_agents)
                        elif task.assigned_agent:
                            agent_ids = str(task.assigned_agent.id)
                            agent_names = task.assigned_agent.name
                        else:
                            agent_ids = ""
                            agent_names = ""
                        task_name = self.inv_task_dict.get(task.id, str(task.id))
                        volunteer_str = "|".join(str(v) for v in task.volunteer_ids) if task.volunteer_ids else ""
                        self._log_buffer.append([
                            finished_case.id,
                            len(finished_case.all_tasks),
                            finished_case.assigned_timestamp,
                            finished_case.completion_timestamp,
                            task.id,
                            task_name,
                            task.assigned_timestamp,
                            task.start_timestamp,
                            task.completion_timestamp,
                            agent_ids,
                            agent_names,
                            task.agents_required,
                            task.assignment_type,
                            volunteer_str,
                        ])
                    if len(self._log_buffer) >= 500:
                        self._flush_log_buffer()

        # Filter out completed cases from pending cases
        self._filter_completed_cases()

        ### ------------------------------- ###
        ### CHECK IF SIMULATION SHOULD STOP ###
        ### ------------------------------- ###
        # Truncations specify when to stop based on training constraints
        truncations = {agent.id: self.steps >= self.max_steps for agent in self.agents}

        # Terminations specify when to stop based on reaching terminal state
        terminations = {
            agent.id: len(self.future_cases) == 0 and len(self.pending_cases) == 0
            for agent in self.agents
        }

        # Compute reward (dict[int, float] if individual_rewards, else float)
        reward_result = get_reward(self)
        if isinstance(reward_result, dict):
            rewards = reward_result
        else:
            rewards = {agent.id: reward_result for agent in self.agents}

        # Return early if simulation should stop
        if any(terminations.values()) or any(truncations.values()):
            self._flush_log_buffer()  # Flush remaining log entries
            return {}, rewards, terminations, truncations, {"step_has_task": self._step_had_task}

        ### --------------------------------------- ###
        ### DETERMINE CASE FOR NEXT SIMULATION STEP ###
        ### --------------------------------------- ###
        # First: check if anything is eligible at the CURRENT time before advancing.
        # This ensures all events at the same timestamp are resolved before time moves.
        self.upcoming_case = self._find_upcoming_case()

        # Only advance time if nothing is eligible at the current timestamp
        if self.upcoming_case is None:
            self.current_time = self._get_next_time()
            # After advancing, check again (new arrivals or completions at the new time)
            self.upcoming_case = self._find_upcoming_case()

        ### ------------------------------- ###
        ###      PREPARE OBSERVATIONS       ###
        ### ------------------------------- ###
        observations = self._get_all_observations()

        return observations, rewards, terminations, truncations, {"step_has_task": self._step_had_task}

    def _get_all_observations(self) -> dict:
        """Build observation dicts for ALL agents efficiently.

        Pre-computes values shared across agents once, then computes
        per-agent features. Returns the same dict format as the original
        _get_observations() — identical values, just faster.
        """
        # ── Shared values (computed once per step) ──────────────────
        upcoming = self.upcoming_case
        task = (
            upcoming.current_task
            if upcoming is not None and upcoming.current_task is not None
            else None
        )
        task_id = task.id if task is not None else -1

        # One-hot encoding of task type
        task_onehot = np.zeros(self.num_activities, dtype=np.float32)
        if task_id >= 0:
            task_onehot[task_id] = 1.0

        # Collaborative info
        if task is not None:
            is_collab = np.float32(1.0 if task.agents_required > 1 else 0.0)
            agents_required_norm = np.float32(task.agents_required / 2.0)
        else:
            is_collab = np.float32(0.0)
            agents_required_norm = np.float32(0.5)

        # System context
        total_cases = len(self.pending_cases) + len(self.completed_cases) + len(self.future_cases)
        pending_ratio = np.float32(len(self.pending_cases) / max(total_cases, 1))

        # Upcoming task name (for stats lookup)
        task_name = self.inv_task_dict.get(task_id) if task_id >= 0 else None

        # Cache for queue work estimation
        global_medians = self.global_activity_medians
        inv_task_dict = self.inv_task_dict
        current_time = self.current_time

        # ── Per-agent observations ──────────────────────────────────
        observations = {}
        _STATUS_IN_PROGRESS = Status.IN_PROGRESS
        _math_log1p = math.log1p

        for agent in self.agents:
            # Agent capability
            caps = agent.capabilities
            agent_can_perform = (
                task_id in caps
                and caps[task_id] is not None
            )

            # Task duration left (log-normalized)
            current_case = agent.current_case
            if current_case is not None:
                ct = current_case.current_task
                if ct is not None and ct.duration is not None and ct.status == _STATUS_IN_PROGRESS:
                    raw_duration_left = (ct.completion_timestamp - current_time).total_seconds()
                    if raw_duration_left < 0:
                        raw_duration_left = 0.0
                else:
                    raw_duration_left = 0.0
            else:
                raw_duration_left = 0.0
            duration_left_norm = _math_log1p(raw_duration_left / 60.0) / _LOG1P_60

            # Queue length (log-normalized)
            q_len = len(agent.case_queue)
            if current_case is not None:
                q_len += 1
            queue_length_norm = _math_log1p(q_len) / _LOG1P_100

            # Queue work remaining & collab count
            queue_work_min = 0.0
            collab_count = 0

            if current_case is not None:
                ct = current_case.current_task
                if ct is not None:
                    if ct.agents_required > 1:
                        collab_count += 1
                    if ct.status == _STATUS_IN_PROGRESS and ct.completion_timestamp is not None:
                        remaining = (ct.completion_timestamp - current_time).total_seconds()
                        if remaining > 0:
                            queue_work_min += remaining / 60.0

            # Iterate directly over the deque (avoids peek() method calls)
            for queued_case in agent.case_queue.queue:
                qt = queued_case.current_task
                if qt is not None:
                    if qt.agents_required > 1:
                        collab_count += 1
                    qt_name = inv_task_dict.get(qt.id)
                    if qt_name is not None:
                        median = global_medians.get(qt_name)
                        if median is not None:
                            queue_work_min += median / 60.0

            queue_work_norm = _math_log1p(queue_work_min) / _LOG1P_660

            # Collab queue features
            total_in_queue = q_len if q_len > 0 else 1
            queue_collab_count_norm = collab_count / total_in_queue
            queue_collab_ratio = collab_count / total_in_queue if q_len > 0 else 0.0

            # Upcoming task stats for this agent
            stats = agent.stats_dict
            if (
                task_name is not None
                and agent_can_perform
                and task_name in stats
                and stats[task_name] is not None
            ):
                s = stats[task_name]
                raw_mean = s["mean"]
                raw_std = s["std"]
            else:
                raw_mean = 0.0
                raw_std = 0.0
            task_mean_norm = _math_log1p(raw_mean / 60.0) / _LOG1P_60
            task_std_norm = _math_log1p(raw_std / 60.0) / _LOG1P_60

            obs = {
                "task_type_onehot": task_onehot,
                "is_collaborative": is_collab,
                "agents_required": agents_required_norm,
                "agent_is_capable": np.float32(float(agent_can_perform)),
                "agent_is_busy": np.float32(float(
                    agent.busy_until is not None or current_case is not None or agent.waiting_for_collab
                )),
                "task_duration_left": np.float32(duration_left_norm),
                "queue_length": np.float32(queue_length_norm),
                "queue_work_remaining": np.float32(queue_work_norm),
                "queue_collab_count": np.float32(queue_collab_count_norm),
                "queue_collab_ratio": np.float32(queue_collab_ratio),
                "upcoming_task_mean": np.float32(task_mean_norm),
                "upcoming_task_std": np.float32(task_std_norm),
                "pending_cases_ratio": pending_ratio,
            }

            # Conditionally add agent identity observations
            if self.use_agent_identity:
                obs["agent_id_onehot"] = self._agent_id_onehots[agent.id]
                obs["agent_role_onehot"] = self._agent_role_onehots[agent.id]

            observations[agent.id] = obs

        return observations

    def _get_observations(self, agent: ResourceAgent):
        """Build observation vector for a single agent.

        Kept for backward compatibility (used in reset). For bulk observations
        during step(), use _get_all_observations() instead.
        """
        # Delegate to the bulk method and extract this agent's observation
        return self._get_all_observations()[agent.id]

    def render(self) -> None:
        """Renders the environment."""
        print("\n--- Environment State ---")
        print(f"Time: {self.current_time}, Step: {self.steps}")
        display_indented_list(self.agents, "Agents")
        display_indented_list(
            self.pending_cases[:5], f"Pending Cases ({len(self.pending_cases)})"
        )
        if len(self.pending_cases) > 5:
            print("  ...")
        if self.future_cases:
            print(f"  Next arrival: {self.future_cases[0].assigned_timestamp}")
        display_indented_list(
            self.future_cases[:5], f"Future Cases ({len(self.future_cases)})"
        )
        if len(self.future_cases) > 5:
            print("  ...")
        print(f"Completed Cases: {len(self.completed_cases)}")
        print("--- End State ---")

    def observation_space(self, agent: int) -> GymDict:
        """Returns the observation space for a single agent."""
        space = {}
        if self.use_agent_identity:
            space["agent_id_onehot"] = Box(0.0, 1.0, shape=(self.n_agents,), dtype=np.float32)
            space["agent_role_onehot"] = Box(0.0, 1.0, shape=(self.n_roles,), dtype=np.float32)
        space.update({
            "task_type_onehot": Box(0.0, 1.0, shape=(self.num_activities,), dtype=np.float32),
            "is_collaborative": Box(0.0, 1.0, shape=(), dtype=np.float32),
            "agents_required": Box(0.0, 5.0, shape=(), dtype=np.float32),
            "agent_is_capable": Box(0.0, 1.0, shape=(), dtype=np.float32),
            "agent_is_busy": Box(0.0, 1.0, shape=(), dtype=np.float32),
            "task_duration_left": Box(0.0, 1.0, shape=(), dtype=np.float32),
            "queue_length": Box(0.0, 1.0, shape=(), dtype=np.float32),
            "queue_work_remaining": Box(0.0, 1.0, shape=(), dtype=np.float32),
            "queue_collab_count": Box(0.0, 1.0, shape=(), dtype=np.float32),
            "queue_collab_ratio": Box(0.0, 1.0, shape=(), dtype=np.float32),
            "upcoming_task_mean": Box(0.0, 1.0, shape=(), dtype=np.float32),
            "upcoming_task_std": Box(0.0, 1.0, shape=(), dtype=np.float32),
            "pending_cases_ratio": Box(0.0, 1.0, shape=(), dtype=np.float32),
        })
        return GymDict(space)

    def _flush_log_buffer(self):
        """Write buffered log rows to CSV file in one batch."""
        if not self._log_buffer or not self.log_file:
            return
        write_header = not os.path.exists(self.log_file)
        df = pd.DataFrame(self._log_buffer, columns=self._log_columns)
        df.to_csv(self.log_file, mode="a", header=write_header, index=False)
        self._log_buffer.clear()

    def reset(self, seed: int | None = None, options=None) -> tuple[dict, dict]:
        """Resets the environment to its initial state."""
        # Flush any remaining log entries from the previous episode
        self._flush_log_buffer()
        self.steps = 0
        self.epochs += 1
        self.current_time = (
            self.data["assign_timestamp"].min()
            if "assign_timestamp" in self.data.columns
            else self.data["start_timestamp"].min()
        )
        # Recreate cases from cached data (much faster than pandas groupby or deepcopy)
        self.future_cases = []
        for data in self._cached_case_data:
            tasks = []
            for t in data['tasks']:
                task = Task(
                    t['id'],
                    data['case_id'],
                    required_roles=list(t['required_roles']),
                )
                task.ground_truth_resource = t.get('ground_truth_resource')
                tasks.append(task)
            self.future_cases.append(Case(
                data['case_id'],
                data['assigned_timestamp'],
                tasks,
                parallel_groups=data.get('parallel_groups'),
            ))
        # Set environment references
        for case in self.future_cases:
            case.environment = self
            for task in case.all_tasks:
                task.environment = self
        self.pending_cases = []
        self.completed_cases = []
        self.upcoming_case = None
        self.completed_task = None
        self._fallback_this_step = False
        self._volunteer_ids_this_step = []
        self._assigned_agent_this_step = None
        self._last_assigned_agent_id = None
        self._step_had_task = False
        if self.enable_logging and self.log_dir:
            current_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            phase = (options or {}).get("phase", "train")
            # Extract subdirectory from phase: "train_ep0" → "train",
            # "eval_ep100_e2" → "eval", "final_eval_ep0" → "final_eval"
            if phase.startswith("final_eval"):
                subdir = "final_eval"
            elif phase.startswith("eval"):
                subdir = "eval"
            else:
                subdir = "train"
            log_subdir = os.path.join(self.log_dir, subdir)
            os.makedirs(log_subdir, exist_ok=True)
            self.log_file = os.path.join(log_subdir, f"log_{phase}_{current_timestamp}.csv")

        # Use cached distributions (computed once in __init__)
        activity_durations_dict = self._cached_activity_durations
        stats_dict = self._cached_stats_dict

        # Update agent capabilities with distributions and reset state
        for agent in self.agents:
            resource = self.resources[agent.id]
            agent.capabilities = {
                self.task_dict[task]: activity_durations_dict[resource][task]
                for task in sorted(set(self.data["activity_name"]))
            }
            agent.stats_dict = stats_dict[resource]  # type: ignore
            agent.waiting_for_collab = False
            agent.busy_until = None
            agent.current_case = None
            agent.case_queue.queue.clear()

        # Set environment reference for all tasks and cases
        for case in self.future_cases:
            case.environment = self
            for task in case.all_tasks:
                task.environment = self

        # Find the first case to offer (moves arrived future cases into pending)
        self.upcoming_case = self._find_upcoming_case()

        observations = self._get_all_observations()

        return observations, {}

    def action_space(self, agent: int) -> Discrete:
        """Returns the action space for a single agent."""
        return Discrete(2)

    def _get_next_time(self) -> pd.Timestamp:
        """Get the next time where action is needed."""
        current_time = self.current_time
        next_time = None

        # Check agent busy_until times
        for agent in self.agents:
            bu = agent.busy_until
            if bu is not None:
                if next_time is None or bu < next_time:
                    next_time = bu

        # Check task completion timestamps from pending cases
        for case in self.pending_cases:
            if case.is_in_parallel_group:
                # Parallel group: check ALL in-progress tasks (not just current_task)
                g_idx = case._task_to_group[case.current_task_index]
                group = case._parallel_groups[g_idx]
                for tidx in group:
                    task = case.all_tasks[tidx]
                    ts = task.completion_timestamp
                    if ts is not None and ts > current_time:
                        if next_time is None or ts < next_time:
                            next_time = ts
            else:
                ct = case.current_task
                if ct is not None:
                    ts = ct.completion_timestamp
                    if ts is not None and ts > current_time:
                        if next_time is None or ts < next_time:
                            next_time = ts

        # Check next case arrival time from future cases
        if self.future_cases:
            arrival_time = self.future_cases[0].assigned_timestamp
            if arrival_time > current_time:
                if next_time is None or arrival_time < next_time:
                    next_time = arrival_time

        # If there are events, advance time to the closest one
        if next_time is not None:
            # Safety check: ensure we always move forward in time
            if next_time <= current_time:
                debug_print_colored(
                    "Time not progressing. Forcing small time increment.",
                    "red",
                )
                next_time = current_time + pd.Timedelta(seconds=1)

            # Work schedule: skip to next work start if outside work hours
            if self.work_schedule_enabled and not is_within_work_hours(next_time, self.work_start_hour, self.work_end_hour):
                next_time = next_work_start(next_time, self.work_start_hour)

            return next_time

        # If no events, advance time by a fixed interval
        next_time = current_time + pd.Timedelta(seconds=1)
        if self.work_schedule_enabled and not is_within_work_hours(next_time, self.work_start_hour, self.work_end_hour):
            next_time = next_work_start(next_time, self.work_start_hour)
        return next_time
