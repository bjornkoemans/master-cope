from enum import Enum
import pandas as pd
from typing import Callable, Optional, List, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulator import AgentOptimizerEnvironment
    from .duration_distribution import DurationDistribution

from .typed_queue import Queue
from .config import debug_print_colored
from .work_schedule import adjust_completion_time


class Status(Enum):
    """Enum to represent the status of a task or case."""

    PENDING = "pending"  # Not yet started but eligible to be started
    OPEN = "open"  # Assigned but not started
    IN_PROGRESS = "in_progress"  # Currently being worked on
    COMPLETED = "completed"  # Successfully completed


class Task:
    def __init__(
        self,
        id: int,
        case_id: int,
        duration: Optional[float] = None,
        agents_required: int = 1,
        required_roles: Optional[List[str]] = None,
    ) -> None:
        """Initialize a task within a case."""
        self.id: int = id
        self.case_id: int = case_id
        self.duration: Optional[float] = duration
        self.assigned_agent: Optional["ResourceAgent"] = None
        self.assigned_agents: List["ResourceAgent"] = []
        self.required_roles: List[str] = required_roles or []
        # Derive agents_required from required_roles if not explicitly set
        self.agents_required: int = len(self.required_roles) if self.required_roles else agents_required
        self.status: Status = Status.PENDING
        self.assigned_timestamp: Optional[pd.Timestamp] = None
        self.start_timestamp: Optional[pd.Timestamp] = None
        self.completion_timestamp: Optional[pd.Timestamp] = None
        self.volunteer_ids: List[int] = []  # Agent IDs that volunteered for this task
        self.assignment_type: str = ""  # e.g. volunteer_single, volunteer_all, volunteer_partial, fallback_random
        self.ground_truth_resource: Optional[str] = None  # Original resource name from the dataset
        self.environment: Optional["AgentOptimizerEnvironment"] = (
            None  # Will be set when task is assigned to an agent
        )

    def __repr__(self) -> str:
        return f"Task(ID: {self.id}, case: {self.case_id}, status: {self.status.value})"

    def assign_to_agent(self, agent: "ResourceAgent", timestamp: pd.Timestamp) -> None:
        """Assign this task to an agent."""
        self.assigned_agent = agent
        if agent not in self.assigned_agents:
            self.assigned_agents.append(agent)
        self.assigned_timestamp = timestamp
        self.status = Status.OPEN
        # Get reference to environment through the agent
        if (
            agent.current_case is not None
            and hasattr(agent.current_case, "environment")
            and agent.current_case.environment is not None
        ):
            self.environment = agent.current_case.environment
        if (
            len(agent.case_queue) > 0
            and agent.case_queue.peek(0) is not None
            and hasattr(agent.case_queue.peek(0), "environment")
            and agent.case_queue.peek(0).environment is not None
        ):
            self.environment = agent.case_queue.peek(0).environment

    def _start(self, timestamp: pd.Timestamp, duration: float) -> None:
        """Start working on this task."""
        if self.assigned_agent is None:
            raise ValueError("Task must be assigned to an agent before starting.")
        if self.status == Status.COMPLETED:
            raise ValueError("Task is already completed.")

        self.status = Status.IN_PROGRESS
        self.start_timestamp = timestamp
        self.duration = duration
        # Respect work schedule during evaluation: task completion spans only work hours
        work_schedule_active = (
            self.environment is not None
            and getattr(self.environment, "work_schedule_enabled", False)
        )
        if work_schedule_active:
            self.completion_timestamp = adjust_completion_time(timestamp, duration)
        else:
            self.completion_timestamp = timestamp + pd.Timedelta(seconds=duration)
        # Set busy_until on all assigned agents (collaborative tasks)
        for agent in self.assigned_agents:
            agent.busy_until = self.completion_timestamp
        # Fallback for single-agent tasks where assigned_agents may be empty
        if not self.assigned_agents:
            self.assigned_agent.busy_until = self.completion_timestamp
        debug_print_colored(
            f"Task {self.format()} started at {timestamp}, will finish at {self.completion_timestamp}",
            "purple",
        )

    def duration_left(self, current_time: pd.Timestamp) -> float:
        """Get the remaining duration of the task."""
        if self.status != Status.IN_PROGRESS:
            raise ValueError("Task must be in progress to check duration left.")
        if self.completion_timestamp is None:
            raise ValueError(
                "Task must have a completion timestamp to check duration left."
            )

        return (self.completion_timestamp - current_time).total_seconds()

    def _handle_completion(self, timestamp: pd.Timestamp) -> None:
        """Handle task completion."""
        if self.status == Status.COMPLETED:
            raise ValueError("Task is already completed.")
        if self.status != Status.IN_PROGRESS:
            raise ValueError("Task must be in progress to be completed.")

        self.status = Status.COMPLETED
        self.completion_timestamp = (
            timestamp  # Set completion timestamp to current time
        )
        # Free all assigned agents (collaborative tasks)
        for agent in self.assigned_agents:
            agent.busy_until = None
        # Fallback for single-agent tasks
        if not self.assigned_agents and self.assigned_agent:
            self.assigned_agent.busy_until = None
        # Set the completed task in the environment
        if self.environment:
            self.environment.completed_task = self

        debug_print_colored(f"Task {self.format()} completed at {timestamp}", "green")

    def work(
        self,
        timestamp: pd.Timestamp,
        duration_distribution: Optional["DurationDistribution"],
    ) -> bool:
        """Work on this task."""
        if self.assigned_agent is None:
            raise ValueError("Task must be assigned to an agent before starting.")
        if self.status == Status.COMPLETED:
            raise ValueError("Task is already completed.")

        # If task is not started yet, start it
        if self.status != Status.IN_PROGRESS:
            if duration_distribution is not None:
                # Homogeneous collaborative: average durations from all capable agents
                if self.agents_required > 1 and self._is_homogeneous():
                    durations = []
                    for agent in self.assigned_agents:
                        if agent.can_perform_task(self.id) and agent.capabilities[self.id] is not None:
                            durations.append(float(agent.capabilities[self.id].generate_sample(1)[0]))
                    if durations:
                        duration = sum(durations) / len(durations)
                    else:
                        duration = float(duration_distribution.generate_sample(1)[0])
                else:
                    # Single agent or heterogeneous: use primary agent's distribution
                    duration = float(duration_distribution.generate_sample(1)[0])
            else:
                duration = 0.0

            self._start(timestamp, duration)

            # If duration is 0, immediately complete the task
            if duration == 0.0:
                self._handle_completion(timestamp)
        # If task is in progress, check if it should be completed
        elif (
            self.completion_timestamp is not None
            and timestamp >= self.completion_timestamp
        ):
            self._handle_completion(timestamp)

        return self.status == Status.COMPLETED

    def _is_homogeneous(self) -> bool:
        """Check if all required roles are the same (homogeneous collaboration)."""
        if not self.required_roles:
            return False
        return len(set(self.required_roles)) == 1

    def format(self) -> str:
        """Format the task for display."""
        return f"{self.case_id}.{self.id}"


class Case:
    """Represents a case (workflow instance) containing multiple tasks."""

    def __init__(
        self,
        case_id: int,
        assign_timestamp: pd.Timestamp,
        tasks: List[Task],
        parallel_groups: Optional[List[set]] = None,
    ) -> None:
        self.id: int = case_id
        self.assigned_timestamp: pd.Timestamp = assign_timestamp
        self.start_timestamp: Optional[pd.Timestamp] = None
        self.completion_timestamp: Optional[pd.Timestamp] = None
        self.status: Status = Status.PENDING
        self.environment: Optional["AgentOptimizerEnvironment"] = None

        # Task management
        self.all_tasks: List[Task] = tasks
        self.current_task_index: int = 0

        # Parallel task groups: each set contains task indices that run in parallel.
        # When None or empty, all tasks run sequentially (original behavior).
        self._parallel_groups: Optional[List[set]] = parallel_groups
        # Quick lookup: task_index → group_index
        self._task_to_group: dict[int, int] = {}
        if parallel_groups:
            for g_idx, group in enumerate(parallel_groups):
                for t_idx in group:
                    self._task_to_group[t_idx] = g_idx
        # For parallel: maps agent_id → task_index (which task each agent works on)
        self._agent_task_map: dict[int, int] = {}

        # Agent assignment
        self.assigned_agent: Optional["ResourceAgent"] = None

    def __repr__(self) -> str:
        agent_id = self.assigned_agent.id if self.assigned_agent else "None"
        return f"Case(ID: {self.id}, agent_id: {agent_id}, status: {self.status.value}, completes: {self.completes_at}, progress: {self.completed_tasks_count}/{len(self.all_tasks)} tasks)"

    @property
    def is_in_parallel_group(self) -> bool:
        """Check if the current task index is within a parallel group."""
        return bool(self._task_to_group) and self.current_task_index in self._task_to_group

    @property
    def current_task(self) -> Optional[Task]:
        """Get the current active task or None if all tasks are completed.

        In a parallel group, returns the first PENDING task (for assignment).
        Falls back to first non-completed task (for status checks).
        """
        if self.current_task_index >= len(self.all_tasks):
            return None
        if self.is_in_parallel_group:
            g_idx = self._task_to_group[self.current_task_index]
            group = self._parallel_groups[g_idx]
            # Prefer first PENDING (needs assignment)
            for i in sorted(group):
                if self.all_tasks[i].status == Status.PENDING:
                    return self.all_tasks[i]
            # Fall back to first non-completed (in-progress or open)
            for i in sorted(group):
                if self.all_tasks[i].status != Status.COMPLETED:
                    return self.all_tasks[i]
            # All completed — group is done
            return None
        return self.all_tasks[self.current_task_index]

    @property
    def completed_tasks(self) -> List[Task]:
        """Get all completed tasks."""
        return [task for task in self.all_tasks if task.status == Status.COMPLETED]

    @property
    def completed_tasks_count(self) -> int:
        """Get the number of completed tasks."""
        # Cannot use current_task_index here because _complete_task() checks
        # is_completed BEFORE incrementing the index, so the count would be
        # off by one at that critical moment.
        count = 0
        _COMPLETED = Status.COMPLETED
        for task in self.all_tasks:
            if task.status == _COMPLETED:
                count += 1
        return count

    @property
    def is_completed(self) -> bool:
        """Check if all tasks in this case are completed."""
        return self.completed_tasks_count == len(self.all_tasks)

    @property
    def completes_at(self) -> pd.Timestamp | None:
        """Get the timestamp when this case is completed."""
        if self.current_task is None:
            return self.completion_timestamp
        if self.is_in_parallel_group:
            # Parallel: case advances when the LAST parallel task finishes
            g_idx = self._task_to_group[self.current_task_index]
            group = self._parallel_groups[g_idx]
            timestamps = [
                self.all_tasks[i].completion_timestamp
                for i in group
                if self.all_tasks[i].completion_timestamp is not None
            ]
            return max(timestamps) if timestamps else None
        return self.current_task.completion_timestamp

    def assign_to_agent(self, agent: "ResourceAgent", timestamp: pd.Timestamp) -> None:
        """Assign this case to an agent."""
        self.assigned_agent = agent
        # Only set assigned_timestamp on FIRST assignment (= case open time).
        # Subsequent task re-assignments must not overwrite it.
        if self.status == Status.PENDING:
            self.assigned_timestamp = timestamp
        self.status = Status.OPEN
        # Get reference to environment through the agent
        if (
            agent.current_case is not None
            and hasattr(agent.current_case, "environment")
            and agent.current_case.environment is not None
        ):
            self.environment = agent.current_case.environment
        if (
            len(agent.case_queue) > 0
            and agent.case_queue.peek(0) is not None
            and hasattr(agent.case_queue.peek(0), "environment")
            and agent.case_queue.peek(0).environment is not None
        ):
            self.environment = agent.case_queue.peek(0).environment

        if agent.current_case is None:
            debug_print_colored(
                f"Case {self.id} assigned to agent {agent.id} (current case)", "green"
            )
            agent.current_case = self
        else:
            debug_print_colored(
                f"Case {self.id} assigned to agent {agent.id} (queued)", "green"
            )
            self.assigned_agent.case_queue.enqueue(self)

        task = self.current_task
        if task:
            # Capture task reference BEFORE assign changes its status to OPEN,
            # which would cause current_task to return a different PENDING task.
            task.assign_to_agent(agent, timestamp)
            # Track which task this agent works on (for parallel groups)
            if self.is_in_parallel_group:
                task_idx = self.all_tasks.index(task)
                self._agent_task_map[agent.id] = task_idx

    def assign_to_agents(self, agents: List["ResourceAgent"], primary_agent: "ResourceAgent", timestamp: pd.Timestamp) -> None:
        """Assign this case to multiple agents for a collaborative task.

        The case is enqueued in each agent's queue. The primary agent is used
        for duration sampling and is set as assigned_agent for backward compat.
        """
        self.assigned_agent = primary_agent
        # Only set assigned_timestamp on FIRST assignment (= case open time).
        if self.status == Status.PENDING:
            self.assigned_timestamp = timestamp
        self.status = Status.OPEN

        # Get reference to environment through any agent that already has one
        for agent in agents:
            if (
                agent.current_case is not None
                and hasattr(agent.current_case, "environment")
                and agent.current_case.environment is not None
            ):
                self.environment = agent.current_case.environment
                break
            if (
                len(agent.case_queue) > 0
                and agent.case_queue.peek(0) is not None
                and hasattr(agent.case_queue.peek(0), "environment")
                and agent.case_queue.peek(0).environment is not None
            ):
                self.environment = agent.case_queue.peek(0).environment
                break

        if self.current_task:
            self.current_task.assigned_agents = list(agents)
            self.current_task.assigned_agent = primary_agent
            self.current_task.assigned_timestamp = timestamp
            self.current_task.status = Status.OPEN

        for agent in agents:
            if agent.current_case is None:
                debug_print_colored(
                    f"Case {self.id} (collab) assigned to agent {agent.id} (current case)", "green"
                )
                agent.current_case = self
            else:
                debug_print_colored(
                    f"Case {self.id} (collab) assigned to agent {agent.id} (queued)", "green"
                )
                agent.case_queue.enqueue(self)

    def work(self, timestamp: pd.Timestamp, agent: Optional["ResourceAgent"] = None) -> bool:
        """Work on the case.

        Args:
            timestamp: Current simulation time.
            agent: The agent calling work (used to find their parallel task).
        """
        # Determine which task to work on
        if agent is not None and agent.id in self._agent_task_map:
            task = self.all_tasks[self._agent_task_map[agent.id]]
        else:
            task = self.current_task

        if task is None:
            self.assigned_agent = None
            return True

        # Check if the task is already completed
        if task.status == Status.COMPLETED:
            debug_print_colored(
                f"Task {task.format()} is already completed", "yellow"
            )
            return self.is_completed

        # Update case state
        self.status = Status.IN_PROGRESS
        if self.start_timestamp is None:
            self.start_timestamp = timestamp

        # Determine the agent working on this task
        working_agent = task.assigned_agent or self.assigned_agent or agent
        if working_agent is None:
            raise ValueError("Task must be assigned to an agent before working on it.")
        # If the task had no assigned agent (e.g. after a collaborative task
        # completed and the case advanced), adopt the calling agent.
        if task.assigned_agent is None:
            task.assigned_agent = working_agent
            task.assigned_agents = [working_agent]
            task.status = Status.OPEN
        if self.assigned_agent is None:
            self.assigned_agent = working_agent

        if task.agents_required > 1 and len(task.assigned_agents) > 1:
            # Collaborative task — compute duration based on homogeen/heterogeen
            duration_distribution = self._get_collaborative_duration(task)
        else:
            duration_distribution = working_agent.capabilities[task.id]

        if duration_distribution is None:
            # Agent cannot perform this task — release the case so it can
            # be re-offered to a capable agent.
            debug_print_colored(
                f"Agent {working_agent.id} ({working_agent.name}) cannot perform "
                f"task {task.id} (case {self.id}), releasing case",
                "yellow",
            )
            # Clean up agent references
            if working_agent.id in self._agent_task_map:
                del self._agent_task_map[working_agent.id]
            if working_agent.current_case == self:
                working_agent.current_case = None
                if len(working_agent.case_queue) > 0:
                    working_agent.current_case = working_agent.case_queue.dequeue()
            self.status = Status.OPEN
            self.assigned_agent = None
            task.assigned_agent = None
            task.assigned_agents = [a for a in task.assigned_agents if a != working_agent]
            task.status = Status.PENDING
            return False
        task_is_done = task.work(timestamp, duration_distribution)

        if task_is_done:
            self._complete_task(timestamp, completed_task=task)
        else:
            debug_print_colored(
                f"Task {task.format()} is still in progress", "yellow"
            )

        return self.is_completed

    def _get_collaborative_duration(self, task: Task) -> Optional["DurationDistribution"]:
        """Get duration distribution for a collaborative task.

        Homogeen (all same role): returns primary agent's distribution
            (averaging happens at sample time in Task.work via collaborative_agents)
        Heterogeen (mixed roles): returns the primary agent's distribution
            (the agent who historically performed this task)
        """
        # Primary agent is the one with historical data for this task
        primary = task.assigned_agent
        if primary and primary.can_perform_task(task.id):
            return primary.capabilities[task.id]
        # Fallback: find any agent that can perform the task
        for agent in task.assigned_agents:
            if agent.can_perform_task(task.id):
                return agent.capabilities[task.id]
        return None

    def _complete(self, timestamp: pd.Timestamp) -> None:
        """Complete the case"""
        debug_print_colored(f"Case {self.id} is completed", "green")
        self.status = Status.COMPLETED
        self.completion_timestamp = timestamp
        if self.current_task:
            self.current_task.status = Status.COMPLETED
            self.current_task.completion_timestamp = timestamp
            # Free all collaborative agents
            for agent in self.current_task.assigned_agents:
                if agent.current_case == self:
                    agent.current_case = None
                agent.waiting_for_collab = False
        if self.assigned_agent:
            if self.assigned_agent.current_case == self:
                self.assigned_agent.current_case = None
            self.assigned_agent = None

    def _complete_task(self, timestamp: pd.Timestamp, completed_task: Optional[Task] = None) -> None:
        """Complete the current task and update the case status."""
        # Check if completing the task also completes the case
        if self.is_completed or (self.current_task is None and completed_task is None):
            self._complete(timestamp)
            return  # Return early after completing the case

        # Use the explicitly passed task (avoids current_task returning wrong
        # task after the completed one's status changed to COMPLETED).
        if completed_task is None:
            completed_task = self.current_task
        debug_print_colored(
            f"Task {completed_task.format()} completed, case returns to open state",
            "green",
        )

        # Remove agent from parallel task map
        agents_to_remove = [
            aid for aid, tidx in self._agent_task_map.items()
            if self.all_tasks[tidx] == completed_task
        ]
        for aid in agents_to_remove:
            del self._agent_task_map[aid]

        # Clear all collaborative agent references
        for agent in completed_task.assigned_agents:
            if agent.current_case == self:
                agent.current_case = None
            agent.waiting_for_collab = False
            if len(agent.case_queue) > 0:
                agent.current_case = agent.case_queue.dequeue()

        # Clear primary agent reference (for non-collaborative or fallback).
        # Skip this for parallel groups: assigned_agent may still be working
        # on another task in the same group.
        if (
            self.assigned_agent
            and self.assigned_agent not in completed_task.assigned_agents
            and not self.is_in_parallel_group
        ):
            if self.assigned_agent.current_case == self:
                self.assigned_agent.current_case = None
            if len(self.assigned_agent.case_queue) > 0:
                self.assigned_agent.current_case = (
                    self.assigned_agent.case_queue.dequeue()
                )

        # Parallel group: wait for ALL group tasks before advancing (fan-in)
        if self.is_in_parallel_group:
            g_idx = self._task_to_group[self.current_task_index]
            group = self._parallel_groups[g_idx]
            all_done = all(
                self.all_tasks[i].status == Status.COMPLETED for i in group
            )
            if not all_done:
                # Group not finished — keep case open for more work
                self.status = Status.OPEN
                self.assigned_agent = None
                return
            # All group tasks done: advance past the entire group
            self.current_task_index = max(group) + 1
        else:
            # Sequential: advance by 1
            self.current_task_index += 1

        # After advancing, reset the status to OPEN
        self.status = Status.OPEN

        # The agent is no longer handling this case
        self.assigned_agent = None

        # Clear the next task's agent here
        if self.current_task:
            self.current_task.assigned_agent = None
            self.current_task.assigned_agents = []
            self.current_task.status = Status.PENDING

    def is_eligible_for_next_task(self, current_time: pd.Timestamp) -> bool:
        """Check if the case is ready to advance to the next task."""
        # No current task means the case is completed or not started
        if self.current_task is None:
            return False

        # Parallel group: eligible if ANY group task is PENDING (needs assignment)
        if self.is_in_parallel_group:
            g_idx = self._task_to_group[self.current_task_index]
            group = self._parallel_groups[g_idx]
            return any(
                self.all_tasks[i].status == Status.PENDING for i in group
            )

        # If the current task is completed, the case needs a new agent assignment
        if self.current_task.status == Status.COMPLETED:
            return True

        # If the case is PENDING, it's waiting for an initial assignment
        if self.status == Status.PENDING:
            return True

        # If the case is OPEN, the current task needs to be started
        if (
            self.status == Status.OPEN
            and self.current_task.status != Status.IN_PROGRESS
        ):
            return True

        # Don't consider cases that are actively being worked on
        return False


class ResourceAgent:
    """Represents an agent that can work on cases and tasks."""

    def __init__(
        self,
        resource_id: int,
        name: str,
        capabilities: dict[int, "DurationDistribution | None"],
        stats_dict: dict[str, dict[str, float]],
    ) -> None:
        self.id: int = resource_id
        self.name: str = name
        # Derive role from name: "Technician-000001" -> "Technician"
        self.role: str = name.rsplit("-", 1)[0] if "-" in name else name
        self.case_queue: Queue["Case"] = Queue()
        self.current_case: Optional[Case] = None
        self.waiting_for_collab: bool = False  # True when waiting at a collaborative sync point
        self.busy_until: Optional[pd.Timestamp] = None
        self.capabilities: dict[int, "DurationDistribution | None"] = capabilities
        self.stats_dict: dict[str, dict[str, float]] = stats_dict

    def __repr__(self) -> str:
        status = "busy" if self.is_busy() else "available"
        return f"Agent(ID: {self.id}, status: {status}, current: {self.current_case}, queue: {len(self.case_queue)} cases, busy until: {self.busy_until})"

    def work_case(self, timestamp: pd.Timestamp) -> Tuple[bool, Case | None]:
        """Work on the next assigned case.

        When a case/task finishes and the agent dequeues the next case from
        its queue, the agent immediately starts working on it at the same
        timestamp so there is no artificial gap between tasks.
        """
        case = self.current_case
        if case is None:
            case = self.case_queue.dequeue()
            self.current_case = case

        if case is None:
            return False, None

        # Determine the task this agent should work on
        if case.is_in_parallel_group and self.id in case._agent_task_map:
            task = case.all_tasks[case._agent_task_map[self.id]]
            # If our mapped task is already completed, clean up and release
            if task.status == Status.COMPLETED:
                del case._agent_task_map[self.id]
                if self.current_case == case:
                    self.current_case = None
                if len(self.case_queue) > 0:
                    self.current_case = self.case_queue.dequeue()
                return False, case
        elif case.is_in_parallel_group and self.id not in case._agent_task_map:
            # Agent dequeued this case but is not mapped to any parallel task
            # (e.g. original task was completed by someone else). Release case.
            if self.current_case == case:
                self.current_case = None
            if len(self.case_queue) > 0:
                self.current_case = self.case_queue.dequeue()
            return False, case
        else:
            task = case.current_task

        # Check if current task is collaborative and needs sync
        if task is not None and task.agents_required > 1 and task.status != Status.IN_PROGRESS:
            # Check if all assigned agents have this case as their current_case
            assigned = task.assigned_agents
            all_ready = all(
                agent.current_case == case
                for agent in assigned
            )
            if not all_ready:
                # Wait for other agents — don't work, don't dequeue next
                self.waiting_for_collab = True
                return False, case
            else:
                self.waiting_for_collab = False
                # Clear waiting flag on all agents
                for agent in assigned:
                    agent.waiting_for_collab = False

        case_is_done = case.work(timestamp, agent=self)
        if case_is_done:
            # Clear the current case if it's done
            if self.current_case == case:
                self.current_case = None
            # Free all collaborative agents from this case
            if task is not None:
                for agent in task.assigned_agents:
                    if agent.id != self.id and agent.current_case == case:
                        agent.current_case = None
                        if len(agent.case_queue) > 0:
                            agent.current_case = agent.case_queue.dequeue()
            # Try to get next case from queue and immediately start working
            if self.current_case is None and len(self.case_queue) > 0:
                self.current_case = self.case_queue.dequeue()
                # Immediately start working on the dequeued case so there
                # is no gap between finishing one task and starting the next.
                next_case = self.current_case
                if next_case is not None:
                    next_task = next_case.current_task
                    if next_task is not None:
                        # Only auto-start non-collaborative tasks that are assigned
                        # and that this agent can actually perform
                        if (next_task.agents_required <= 1
                                and next_task.assigned_agent is not None
                                and self.can_perform_task(next_task.id)):
                            next_case.work(timestamp)
        else:
            # The case's current task may have completed (advancing to next task
            # in the case) even though the case itself is not done.  In that
            # scenario _complete_task() freed the agent and dequeued the next
            # case.  If we now have a NEW current_case (different from the one
            # we just worked on), start it immediately.
            if self.current_case is not None and self.current_case != case:
                next_case = self.current_case
                next_task = next_case.current_task
                if next_task is not None:
                    if (next_task.agents_required <= 1
                            and next_task.assigned_agent is not None
                            and self.can_perform_task(next_task.id)):
                        next_case.work(timestamp)

        return case_is_done, case

    def can_perform_task(self, task_id: int) -> bool:
        """Check if the agent can perform a specific task."""
        return task_id in self.capabilities and self.capabilities[task_id] is not None

    def is_busy(self) -> bool:
        """Check if the agent is currently busy (includes waiting for collaboration)."""
        return self.busy_until is not None or self.current_case is not None or self.waiting_for_collab

    def task_duration(self, time: pd.Timestamp) -> Optional[float]:
        """Get the duration of the task the agent is currently working on."""
        if self.current_case is None or self.current_case.current_task is None:
            return -1
        if self.current_case.current_task.duration is not None:
            return self.current_case.current_task.duration_left(time)
        return -1
