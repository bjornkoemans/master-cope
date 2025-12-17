from enum import Enum
import pandas as pd
from typing import Callable, Optional, List, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .custom_environment import AgentOptimizerEnvironment
    from .duration_distribution import DurationDistribution

from .typed_queue import Queue
from ..core.env_config import debug_print_colored


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
    ) -> None:
        """Initialize a task within a case."""
        self.id: int = id
        self.case_id: int = case_id
        self.duration: Optional[float] = duration
        self.assigned_agent: Optional["ResourceAgent"] = None
        self.status: Status = Status.PENDING
        self.assigned_timestamp: Optional[pd.Timestamp] = None
        self.start_timestamp: Optional[pd.Timestamp] = None
        self.completion_timestamp: Optional[pd.Timestamp] = None
        self.environment: Optional["AgentOptimizerEnvironment"] = (
            None  # Will be set when task is assigned to an agent
        )

    def __repr__(self) -> str:
        return f"Task(ID: {self.id}, case: {self.case_id}, status: {self.status.value})"

    def assign_to_agent(self, agent: "ResourceAgent", timestamp: pd.Timestamp) -> None:
        """Assign this task to an agent."""
        self.assigned_agent = agent
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
            agent.case_queue.size() > 0
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
        self.completion_timestamp = timestamp + pd.Timedelta(seconds=duration)
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
        if self.assigned_agent:
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

        # Agent assignment
        self.assigned_agent: Optional["ResourceAgent"] = None

    def __repr__(self) -> str:
        agent_id = self.assigned_agent.id if self.assigned_agent else "None"
        return f"Case(ID: {self.id}, agent_id: {agent_id}, status: {self.status.value}, completes: {self.completes_at}, progress: {self.completed_tasks_count}/{len(self.all_tasks)} tasks)"

    @property
    def current_task(self) -> Optional[Task]:
        """Get the current active task or None if all tasks are completed."""
        if self.current_task_index >= len(self.all_tasks):
            return None
        return self.all_tasks[self.current_task_index]

    @property
    def completed_tasks(self) -> List[Task]:
        """Get all completed tasks."""
        return [task for task in self.all_tasks if task.status == Status.COMPLETED]

    @property
    def completed_tasks_count(self) -> int:
        """Get the number of completed tasks."""
        return len(self.completed_tasks)

    @property
    def is_completed(self) -> bool:
        """Check if all tasks in this case are completed."""
        return self.completed_tasks_count == len(self.all_tasks)

    @property
    def completes_at(self) -> pd.Timestamp | None:
        """Get the timestamp when this case is completed."""
        if self.current_task is None:
            return self.completion_timestamp
        return self.current_task.completion_timestamp

    def assign_to_agent(self, agent: "ResourceAgent", timestamp: pd.Timestamp) -> None:
        """Assign this case to an agent."""
        self.assigned_agent = agent
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
            agent.case_queue.size() > 0
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

        if self.current_task:
            self.current_task.assign_to_agent(agent, timestamp)

    def work(self, timestamp: pd.Timestamp) -> bool:
        """Work on the case."""
        if self.current_task is None:
            self.assigned_agent = None
            return True

        # Check if the task is already completed
        if self.current_task.status == Status.COMPLETED:
            debug_print_colored(
                f"Task {self.current_task.format()} is already completed", "yellow"
            )
            return True

        # Update case state
        self.status = Status.IN_PROGRESS
        if self.current_task_index < 1:
            self.start_timestamp = timestamp

        # Update task state
        if self.assigned_agent is None:
            raise ValueError("Task must be assigned to an agent before working on it.")

        duration_distribution = self.assigned_agent.capabilities[self.current_task.id]
        if duration_distribution is None:
            raise ValueError(
                f"Agent {self.assigned_agent.id} cannot perform task {self.current_task.id}"
            )
        task_is_done = self.current_task.work(timestamp, duration_distribution)

        if task_is_done:
            self._complete_task(timestamp)
        else:
            debug_print_colored(
                f"Task {self.current_task.format()} is still in progress", "yellow"
            )

        return self.is_completed

    def _complete(self, timestamp: pd.Timestamp) -> None:
        """Complete the case"""
        debug_print_colored(f"Case {self.id} is completed", "green")
        self.status = Status.COMPLETED
        self.completion_timestamp = timestamp
        if self.current_task:
            self.current_task.status = Status.COMPLETED
            self.current_task.completion_timestamp = timestamp
        if self.assigned_agent:
            self.assigned_agent.current_case = None
            self.assigned_agent = None

    def _complete_task(self, timestamp: pd.Timestamp) -> None:
        """Complete the current task and update the case status."""
        # Check if completing the task also completes the case
        if self.is_completed or self.current_task is None:
            self._complete(timestamp)
            return  # Return early after completing the case

        debug_print_colored(
            f"Task {self.current_task.format()} completed, case returns to open state",
            "green",
        )

        # Clear agent references properly before advancing to next task
        if self.assigned_agent:
            # Only clear current_case if it matches this case
            if self.assigned_agent.current_case == self:
                self.assigned_agent.current_case = None
            # Let the agent pick up the next case from queue
            if self.assigned_agent.case_queue.size() > 0:
                self.assigned_agent.current_case = (
                    self.assigned_agent.case_queue.dequeue()
                )

        # Move to next task
        self.current_task_index += 1

        # After moving to next task, reset the status to PENDING
        self.status = Status.OPEN

        # The agent is no longer handling this case
        self.assigned_agent = None

        # Clear the next task's agent here
        if self.current_task:
            self.current_task.assigned_agent = None
            self.current_task.status = Status.PENDING

    def is_eligible_for_next_task(self, current_time: pd.Timestamp) -> bool:
        """Check if the case is ready to advance to the next task."""
        # No current task means the case is completed or not started
        if self.current_task is None:
            return False

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
        self.case_queue: Queue["Case"] = Queue()
        self.current_case: Optional[Case] = None
        self.busy_until: Optional[pd.Timestamp] = None
        self.capabilities: dict[int, "DurationDistribution | None"] = capabilities
        self.stats_dict: dict[str, dict[str, float]] = stats_dict

    def __repr__(self) -> str:
        status = "busy" if self.is_busy() else "available"
        return f"Agent(ID: {self.id}, status: {status}, current: {self.current_case}, queue: {len(self.case_queue)} cases, busy until: {self.busy_until})"

    def work_case(self, timestamp: pd.Timestamp) -> Tuple[bool, Case | None]:
        """Work on the next assigned case."""
        case = None
        if self.current_case is not None:
            case = self.current_case
        else:
            case = self.case_queue.dequeue()
            self.current_case = case

        if case is not None:
            case_is_done = case.work(timestamp)
            if case_is_done:
                # Clear the current case if it's done
                if self.current_case == case:
                    self.current_case = None
                # Try to get next case from queue
                if self.current_case is None and self.case_queue.size() > 0:
                    self.current_case = self.case_queue.dequeue()
            return case_is_done, case

        return False, case

    def can_perform_task(self, task_id: int) -> bool:
        """Check if the agent can perform a specific task."""
        return task_id in self.capabilities and self.capabilities[task_id] is not None

    def is_busy(self) -> bool:
        """Check if the agent is currently busy."""
        return self.busy_until is not None or self.current_case is not None

    def task_duration(self, time: pd.Timestamp) -> Optional[float]:
        """Get the duration of the task the agent is currently working on."""
        if self.current_case is None or self.current_case.current_task is None:
            return -1
        if self.current_case.current_task.duration is not None:
            return self.current_case.current_task.duration_left(time)
        return -1
