import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
import threading
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Enum for possible task statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressStats(BaseModel):
    """Statistics for a task's progress."""
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    error_message: Optional[str] = None

    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Calculate elapsed time since task started."""
        if not self.start_time:
            return None

        end = self.end_time if self.end_time else datetime.now()
        return end - self.start_time

    @property
    def completion_percentage(self) -> float:
        """Calculate percentage of task completion."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100

    @property
    def remaining_time(self) -> Optional[timedelta]:
        """Estimate remaining time until completion."""
        if not self.estimated_completion_time:
            return None

        remaining = self.estimated_completion_time - datetime.now()
        return remaining if remaining.total_seconds() > 0 else timedelta(seconds=0)

    def update_estimate(self) -> None:
        """Update the estimated completion time based on current progress."""
        if (
                0 < self.processed_items < self.total_items
                and self.start_time
        ):
            elapsed = (datetime.now() - self.start_time).total_seconds()
            items_per_second = self.processed_items / elapsed if elapsed > 0 else 0

            if items_per_second > 0:
                remaining_items = self.total_items - self.processed_items
                remaining_seconds = remaining_items / items_per_second
                self.estimated_completion_time = datetime.now() + timedelta(seconds=remaining_seconds)
            else:
                self.estimated_completion_time = None
        else:
            self.estimated_completion_time = None


class ProgressTracker:
    """Tracks progress of long-running operations with callbacks and persistence."""

    def __init__(self):
        self.tasks: Dict[str, ProgressStats] = {}
        self.callbacks: Dict[str, List[Callable[[ProgressStats], None]]] = {}
        self._lock = threading.RLock()

    def register_task(self, task_id: str, total_items: int) -> ProgressStats:
        """Register a new task to be tracked."""
        with self._lock:
            stats = ProgressStats(total_items=total_items)
            self.tasks[task_id] = stats
            self.callbacks[task_id] = []
            return stats

    def add_callback(self, task_id: str, callback: Callable[[ProgressStats], None]) -> None:
        """Add a callback for progress updates on a specific task."""
        with self._lock:
            if task_id not in self.callbacks:
                self.callbacks[task_id] = []
            self.callbacks[task_id].append(callback)

    def start_task(self, task_id: str) -> None:
        """Mark a task as started."""
        with self._lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not registered")

            stats = self.tasks[task_id]
            stats.start_time = datetime.now()
            stats.last_update_time = datetime.now()
            stats.status = TaskStatus.RUNNING
            self._notify_callbacks(task_id)

    def update_progress(
            self,
            task_id: str,
            processed: Optional[int] = None,
            successful: Optional[int] = None,
            failed: Optional[int] = None,
            message: Optional[str] = None
    ) -> None:
        """Update the progress of a task."""
        with self._lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not registered")

            stats = self.tasks[task_id]

            if processed is not None:
                stats.processed_items = processed

            if successful is not None:
                stats.successful_items = successful

            if failed is not None:
                stats.failed_items = failed

            stats.last_update_time = datetime.now()
            stats.update_estimate()

            if message:
                logger.info(f"Task {task_id}: {message}")

            self._notify_callbacks(task_id)

    def increment_progress(
            self,
            task_id: str,
            processed: int = 1,
            successful: int = 0,
            failed: int = 0,
            message: Optional[str] = None
    ) -> None:
        """Increment the progress counters of a task."""
        with self._lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not registered")

            stats = self.tasks[task_id]
            stats.processed_items += processed
            stats.successful_items += successful
            stats.failed_items += failed
            stats.last_update_time = datetime.now()
            stats.update_estimate()

            if message:
                logger.info(f"Task {task_id}: {message}")

            self._notify_callbacks(task_id)

    def complete_task(self, task_id: str, success: bool = True, error: Optional[str] = None) -> None:
        """Mark a task as completed or failed."""
        with self._lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not registered")

            stats = self.tasks[task_id]
            stats.end_time = datetime.now()

            if success:
                stats.status = TaskStatus.COMPLETED
                logger.info(f"Task {task_id} completed successfully")
            else:
                stats.status = TaskStatus.FAILED
                stats.error_message = error
                logger.error(f"Task {task_id} failed: {error}")

            self._notify_callbacks(task_id)

    def cancel_task(self, task_id: str, reason: Optional[str] = None) -> None:
        """Cancel a running task."""
        with self._lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not registered")

            stats = self.tasks[task_id]
            stats.end_time = datetime.now()
            stats.status = TaskStatus.CANCELLED
            stats.error_message = f"Cancelled: {reason}" if reason else "Cancelled"

            logger.info(f"Task {task_id} cancelled: {reason or 'No reason provided'}")
            self._notify_callbacks(task_id)

    def get_progress(self, task_id: str) -> Optional[ProgressStats]:
        """Get the current progress of a task."""
        with self._lock:
            return self.tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, ProgressStats]:
        """Get a dictionary of all tracked tasks."""
        with self._lock:
            return self.tasks.copy()

    def _notify_callbacks(self, task_id: str) -> None:
        """Notify all callbacks registered for a task."""
        stats = self.tasks[task_id]
        callbacks = self.callbacks.get(task_id, [])

        for callback in callbacks:
            try:
                callback(stats)
            except Exception as e:
                logger.error(f"Error in progress callback for task {task_id}: {e}")

    def create_progress_context(self, task_id: str, total_items: int):
        """Create a context manager for a task to automatically manage its lifecycle."""
        return ProgressContext(self, task_id, total_items)


class ProgressContext:
    """Context manager for task progress tracking."""

    def __init__(self, tracker: ProgressTracker, task_id: str, total_items: int):
        self.tracker = tracker
        self.task_id = task_id
        self.total_items = total_items

    def __enter__(self):
        self.tracker.register_task(self.task_id, self.total_items)
        self.tracker.start_task(self.task_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.tracker.complete_task(
                self.task_id,
                success=False,
                error=str(exc_val)
            )
        else:
            self.tracker.complete_task(self.task_id, success=True)
        return False  # Don't suppress exceptions

    def update(self, processed: int, message: Optional[str] = None):
        """Update the progress counter."""
        self.tracker.update_progress(
            self.task_id,
            processed=processed,
            message=message
        )

    def increment(self, success: bool = True, message: Optional[str] = None):
        """Increment the progress counter by 1."""
        self.tracker.increment_progress(
            self.task_id,
            processed=1,
            successful=1 if success else 0,
            failed=0 if success else 1,
            message=message
        )
