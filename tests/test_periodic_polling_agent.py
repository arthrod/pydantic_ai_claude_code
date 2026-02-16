"""Tests for periodic polling and multi-agent auto-messaging patterns.

Simulates a scenario where:
- A 'worker' agent processes a long-running task
- A 'monitor' agent polls every minute to check if work is done
- A 'notifier' agent automatically sends status messages

Uses FunctionModel to simulate deterministic agent behavior without
calling real LLM APIs.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent, models
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# Block real model requests
models.ALLOW_MODEL_REQUESTS = False


# ===== Shared Domain Models =====


class TaskStatus(str, Enum):
    """Status of a long-running task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class StatusReport(BaseModel):
    """Report from the monitor agent."""

    task_id: str
    status: TaskStatus
    progress_pct: int
    message: str


class Notification(BaseModel):
    """Notification from the notifier agent."""

    recipient: str
    subject: str
    body: str
    timestamp: str


@dataclass
class TaskStore:
    """Shared state representing a long-running task.

    Simulates real-world task state that progresses over time.
    """

    task_id: str = "task-001"
    status: TaskStatus = TaskStatus.PENDING
    progress_pct: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0
    result: str = ""
    poll_count: int = 0
    notifications: list[dict[str, str]] = field(default_factory=list)

    def advance(self, pct_increment: int = 25) -> None:
        """Advance the task by a percentage increment."""
        if self.status == TaskStatus.PENDING:
            self.status = TaskStatus.IN_PROGRESS
            self.started_at = time.monotonic()

        self.progress_pct = min(100, self.progress_pct + pct_increment)

        if self.progress_pct >= 100:
            self.status = TaskStatus.COMPLETED
            self.completed_at = time.monotonic()
            self.result = "Task completed successfully"

    def get_status_summary(self) -> str:
        """Get a human-readable status summary."""
        if self.status == TaskStatus.COMPLETED:
            return f"Task {self.task_id} completed: {self.result}"
        if self.status == TaskStatus.IN_PROGRESS:
            return f"Task {self.task_id} in progress: {self.progress_pct}% done"
        return f"Task {self.task_id}: {self.status.value}"


# ===== Agent Definitions =====


def create_worker_agent() -> Agent:
    """Create a worker agent that processes tasks."""
    return Agent(
        "claude-code:sonnet",
        system_prompt=(
            "You are a task processing worker. When asked to process a task, "
            "use the process_step tool to advance the task."
        ),
    )


def create_monitor_agent() -> Agent[None, StatusReport]:
    """Create a monitor agent that checks task status."""
    return Agent(
        "claude-code:sonnet",
        output_type=StatusReport,
        system_prompt=(
            "You are a task monitor. Check the current status of the task "
            "and report back with a StatusReport."
        ),
    )


def create_notifier_agent() -> Agent[None, Notification]:
    """Create a notifier agent that sends status messages."""
    return Agent(
        "claude-code:sonnet",
        output_type=Notification,
        system_prompt=(
            "You are a notification agent. Send status notifications "
            "to the user about task progress."
        ),
    )


# ===== FunctionModel factories =====


def make_worker_model(task_store: TaskStore):
    """Create a FunctionModel that simulates worker behavior."""

    def worker_fn(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        task_store.advance(25)
        return ModelResponse(
            parts=[
                TextPart(
                    f"Advanced task to {task_store.progress_pct}%. "
                    f"Status: {task_store.status.value}"
                )
            ]
        )

    return FunctionModel(worker_fn)


def make_monitor_model(task_store: TaskStore):
    """Create a FunctionModel that simulates monitor polling behavior."""

    def monitor_fn(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        task_store.poll_count += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "task_id": task_store.task_id,
                        "status": task_store.status.value,
                        "progress_pct": task_store.progress_pct,
                        "message": task_store.get_status_summary(),
                    },
                    tool_call_id=f"poll_{task_store.poll_count}",
                )
            ]
        )

    return FunctionModel(monitor_fn)


def make_notifier_model(task_store: TaskStore):
    """Create a FunctionModel that simulates notifier behavior."""

    def notifier_fn(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        ts = datetime.now(timezone.utc).isoformat()
        notification_data = {
            "recipient": "user@example.com",
            "subject": f"Task {task_store.task_id} - {task_store.status.value}",
            "body": (
                f"Progress: {task_store.progress_pct}%\n"
                f"{task_store.get_status_summary()}"
            ),
            "timestamp": ts,
        }
        task_store.notifications.append(notification_data)

        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args=notification_data,
                    tool_call_id=f"notify_{len(task_store.notifications)}",
                )
            ]
        )

    return FunctionModel(notifier_fn)


# ===== Test Classes =====


class TestPeriodicPolling:
    """Test periodic polling pattern where a monitor checks task progress."""

    async def test_single_poll_cycle(self):
        """Verify a single monitor poll returns correct status."""
        store = TaskStore()
        store.advance(50)  # Pre-advance to 50%

        monitor = create_monitor_agent()

        with monitor.override(model=make_monitor_model(store)):
            result = await monitor.run("Check task status")

        assert isinstance(result.output, StatusReport)
        assert result.output.task_id == "task-001"
        assert result.output.status == TaskStatus.IN_PROGRESS
        assert result.output.progress_pct == 50
        assert store.poll_count == 1

    async def test_polling_until_completion(self):
        """Simulate polling every 'minute' until the task is done.

        The monitor polls, the worker advances, and we repeat until
        the task reaches 100%.
        """
        store = TaskStore()
        worker = create_worker_agent()
        monitor = create_monitor_agent()

        poll_results: list[StatusReport] = []
        max_polls = 10

        with (
            worker.override(model=make_worker_model(store)),
            monitor.override(model=make_monitor_model(store)),
        ):
            for poll_number in range(1, max_polls + 1):
                # Worker advances the task
                await worker.run(f"Process step {poll_number}")

                # Monitor checks status
                status_result = await monitor.run(
                    f"Poll #{poll_number}: Check task status"
                )
                poll_results.append(status_result.output)

                if status_result.output.status == TaskStatus.COMPLETED:
                    break

        # Verify progression
        assert len(poll_results) >= 2  # Should take at least 2 polls
        assert poll_results[-1].status == TaskStatus.COMPLETED
        assert poll_results[-1].progress_pct == 100

        # Verify monotonic progress
        for i in range(1, len(poll_results)):
            assert poll_results[i].progress_pct >= poll_results[i - 1].progress_pct

    async def test_polling_with_simulated_intervals(self):
        """Test polling with simulated time intervals.

        Uses asyncio.sleep(0) to yield control (simulating intervals
        without actually waiting 60 seconds).
        """
        store = TaskStore()
        monitor = create_monitor_agent()

        poll_times: list[float] = []
        interval_seconds = 0.01  # Use small intervals for testing

        with monitor.override(model=make_monitor_model(store)):
            for _ in range(5):
                store.advance(20)
                poll_start = time.monotonic()

                result = await monitor.run("Periodic check")
                poll_times.append(time.monotonic() - poll_start)

                if result.output.status == TaskStatus.COMPLETED:
                    break

                # Simulate waiting for the next interval
                await asyncio.sleep(interval_seconds)

        assert store.poll_count == 5
        assert store.status == TaskStatus.COMPLETED

    async def test_polling_pending_task(self):
        """Verify polling a task that hasn't started yet."""
        store = TaskStore()  # Starts as PENDING
        monitor = create_monitor_agent()

        with monitor.override(model=make_monitor_model(store)):
            result = await monitor.run("Check status")

        assert result.output.status == TaskStatus.PENDING
        assert result.output.progress_pct == 0


class TestMultiAgentAutoMessaging:
    """Test multi-agent pattern where monitor triggers notifier automatically."""

    async def test_notifier_sends_on_completion(self):
        """Verify notifier sends a message when task completes."""
        store = TaskStore()
        store.advance(100)  # Mark as complete

        notifier = create_notifier_agent()

        with notifier.override(model=make_notifier_model(store)):
            result = await notifier.run("Send completion notification")

        assert isinstance(result.output, Notification)
        assert "completed" in result.output.subject.lower()
        assert result.output.recipient == "user@example.com"
        assert len(store.notifications) == 1

    async def test_monitor_triggers_notifier(self):
        """Simulate monitor polling then notifier sending updates.

        This is the full pattern: worker does work, monitor polls,
        notifier sends a message after each poll cycle.
        """
        store = TaskStore()
        worker = create_worker_agent()
        monitor = create_monitor_agent()
        notifier = create_notifier_agent()

        all_notifications: list[Notification] = []
        all_statuses: list[StatusReport] = []

        with (
            worker.override(model=make_worker_model(store)),
            monitor.override(model=make_monitor_model(store)),
            notifier.override(model=make_notifier_model(store)),
        ):
            for cycle in range(1, 8):
                # 1. Worker advances
                await worker.run(f"Work cycle {cycle}")

                # 2. Monitor polls
                status_result = await monitor.run(f"Poll cycle {cycle}")
                all_statuses.append(status_result.output)

                # 3. Notifier sends auto-message
                notify_result = await notifier.run(
                    f"Notify about cycle {cycle}"
                )
                all_notifications.append(notify_result.output)

                if status_result.output.status == TaskStatus.COMPLETED:
                    break

        # Verify the full cycle completed
        assert all_statuses[-1].status == TaskStatus.COMPLETED
        assert len(all_notifications) >= 2

        # Verify notifications track the progress
        for notification in all_notifications:
            assert notification.recipient == "user@example.com"
            assert "task-001" in notification.subject.lower()

        # Verify the notification store matches
        assert len(store.notifications) == len(all_notifications)

    async def test_multiple_notifications_per_cycle(self):
        """Verify notifier can send multiple notifications for one cycle."""
        store = TaskStore()
        store.advance(50)

        notifier = create_notifier_agent()

        with notifier.override(model=make_notifier_model(store)):
            n1 = await notifier.run("Send progress update")
            n2 = await notifier.run("Send ETA update")

        assert len(store.notifications) == 2
        assert n1.output.timestamp != n2.output.timestamp

    async def test_notification_content_reflects_progress(self):
        """Verify notification body contains actual progress info."""
        store = TaskStore()
        notifier = create_notifier_agent()

        notifications: list[Notification] = []

        with notifier.override(model=make_notifier_model(store)):
            # Send at 0%
            result = await notifier.run("Initial status")
            notifications.append(result.output)

            store.advance(50)

            # Send at 50%
            result = await notifier.run("Progress update")
            notifications.append(result.output)

            store.advance(50)

            # Send at 100%
            result = await notifier.run("Completion update")
            notifications.append(result.output)

        assert "0%" in notifications[0].body
        assert "50%" in notifications[1].body
        assert "100%" in notifications[2].body


class TestPollingWithTimeSimulation:
    """Test the every-minute polling pattern with time simulation.

    Simulates the scenario described by the user: a user checks every
    minute if work was done, and an agent auto-messages with status.
    """

    async def test_minute_by_minute_polling_simulation(self):
        """Full simulation of minute-by-minute polling.

        Simulates 5 'minutes' where:
        - Every minute the user (monitor agent) checks status
        - The notifier agent automatically sends a message
        - The worker makes progress each cycle
        """
        store = TaskStore()
        worker = create_worker_agent()
        monitor = create_monitor_agent()
        notifier = create_notifier_agent()

        # Track simulated minute-by-minute activity
        minute_log: list[dict[str, Any]] = []

        with (
            worker.override(model=make_worker_model(store)),
            monitor.override(model=make_monitor_model(store)),
            notifier.override(model=make_notifier_model(store)),
        ):
            for simulated_minute in range(1, 6):
                cycle_record: dict[str, Any] = {
                    "minute": simulated_minute,
                }

                # Worker does a unit of work
                work_result = await worker.run(
                    f"Continue processing at minute {simulated_minute}"
                )
                cycle_record["work_output"] = work_result.output
                cycle_record["progress"] = store.progress_pct

                # User polls: "Is the work done?"
                poll_result = await monitor.run(
                    f"Minute {simulated_minute}: Is the work done yet?"
                )
                cycle_record["poll_status"] = poll_result.output.status.value
                cycle_record["poll_progress"] = poll_result.output.progress_pct

                # Notifier automatically sends update
                notify_result = await notifier.run(
                    f"Auto-message for minute {simulated_minute}"
                )
                cycle_record["notification_subject"] = (
                    notify_result.output.subject
                )
                cycle_record["notification_body"] = notify_result.output.body

                minute_log.append(cycle_record)

                # Simulate waiting 1 minute (use tiny sleep for tests)
                await asyncio.sleep(0.001)

                if poll_result.output.status == TaskStatus.COMPLETED:
                    break

        # Assertions about the full simulation
        assert len(minute_log) >= 4, (
            "Should have polled at least 4 minutes (4 x 25% = 100%)"
        )

        # First minute should show progress starting
        assert minute_log[0]["progress"] == 25

        # Last minute should show completion
        last = minute_log[-1]
        assert last["poll_status"] == "completed"
        assert last["poll_progress"] == 100

        # Every minute should have a notification
        for record in minute_log:
            assert "notification_subject" in record
            assert "task-001" in record["notification_subject"].lower()

        # Progress should be monotonically increasing
        progress_values = [r["progress"] for r in minute_log]
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1]

    async def test_early_completion_stops_polling(self):
        """Verify polling stops as soon as task completes."""
        store = TaskStore()
        store.advance(75)  # Start at 75% - one more step to complete

        worker = create_worker_agent()
        monitor = create_monitor_agent()

        poll_count = 0

        with (
            worker.override(model=make_worker_model(store)),
            monitor.override(model=make_monitor_model(store)),
        ):
            for _ in range(10):  # Max 10 iterations
                await worker.run("Work")
                result = await monitor.run("Check")
                poll_count += 1

                if result.output.status == TaskStatus.COMPLETED:
                    break

        # Should have completed after just 1 poll (75% + 25% = 100%)
        assert poll_count == 1
        assert store.status == TaskStatus.COMPLETED

    async def test_failed_task_reported_correctly(self):
        """Verify a failed task is reported through the polling chain."""
        store = TaskStore()
        store.status = TaskStatus.FAILED

        monitor = create_monitor_agent()
        notifier = create_notifier_agent()

        with (
            monitor.override(model=make_monitor_model(store)),
            notifier.override(model=make_notifier_model(store)),
        ):
            status = await monitor.run("Check failed task")
            notification = await notifier.run("Report failure")

        assert status.output.status == TaskStatus.FAILED
        assert "failed" in notification.output.subject.lower()


class TestAgentDelegation:
    """Test patterns where one agent delegates to another."""

    async def test_worker_to_monitor_delegation(self):
        """Simulate delegation where worker completes then monitor reports."""
        store = TaskStore()
        worker = create_worker_agent()
        monitor = create_monitor_agent()

        with (
            worker.override(model=make_worker_model(store)),
            monitor.override(model=make_monitor_model(store)),
        ):
            # Worker completes multiple steps
            for _ in range(4):
                await worker.run("Process")

            # Monitor reports final state
            result = await monitor.run("Final report")

        assert result.output.status == TaskStatus.COMPLETED
        assert result.output.progress_pct == 100

    async def test_concurrent_agent_operations(self):
        """Test running multiple agents concurrently."""
        store = TaskStore()
        monitor = create_monitor_agent()
        notifier = create_notifier_agent()

        store.advance(50)

        with (
            monitor.override(model=make_monitor_model(store)),
            notifier.override(model=make_notifier_model(store)),
        ):
            # Run monitor and notifier concurrently
            status_task = asyncio.create_task(
                monitor.run("Concurrent poll")
            )
            notify_task = asyncio.create_task(
                notifier.run("Concurrent notify")
            )

            status_result, notify_result = await asyncio.gather(
                status_task, notify_task
            )

        assert isinstance(status_result.output, StatusReport)
        assert isinstance(notify_result.output, Notification)
        assert status_result.output.progress_pct == 50

    async def test_full_lifecycle_with_all_three_agents(self):
        """End-to-end test: worker processes, monitor polls, notifier sends.

        This is the comprehensive test that exercises the full multi-agent
        interaction pattern requested by the user.
        """
        store = TaskStore()
        worker = create_worker_agent()
        monitor = create_monitor_agent()
        notifier = create_notifier_agent()

        lifecycle_events: list[str] = []

        with (
            worker.override(model=make_worker_model(store)),
            monitor.override(model=make_monitor_model(store)),
            notifier.override(model=make_notifier_model(store)),
        ):
            while store.status != TaskStatus.COMPLETED:
                # Phase 1: Worker does work
                await worker.run("Process next step")
                lifecycle_events.append(
                    f"work:{store.progress_pct}%"
                )

                # Phase 2: Monitor checks
                status = await monitor.run("Check progress")
                lifecycle_events.append(
                    f"poll:{status.output.status.value}"
                )

                # Phase 3: Notifier auto-messages
                notification = await notifier.run("Auto-notify")
                lifecycle_events.append(
                    f"notify:{notification.output.subject}"
                )

                # Safety valve
                if len(lifecycle_events) > 30:
                    break

        # Verify the lifecycle
        assert store.status == TaskStatus.COMPLETED
        assert store.progress_pct == 100

        # Should have exactly 4 full cycles (4 x 25% = 100%)
        work_events = [e for e in lifecycle_events if e.startswith("work:")]
        poll_events = [e for e in lifecycle_events if e.startswith("poll:")]
        notify_events = [e for e in lifecycle_events if e.startswith("notify:")]

        assert len(work_events) == 4
        assert len(poll_events) == 4
        assert len(notify_events) == 4

        # Last poll should show completion
        assert "completed" in poll_events[-1]

        # All notifications should have been recorded
        assert len(store.notifications) == 4
