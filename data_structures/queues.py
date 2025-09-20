"""
FIFO Queues for Judgment Processing
==================================

This module implements queue data structures for judgment import and processing
workflows in Find Case Law (FCL). Provides first-in-first-out (FIFO) processing
for document ingestion, analysis, and publishing pipelines.

Key FCL Use Cases:
- Judgment import/processing queue management
- Background task scheduling for document analysis
- Bulk document operations with proper ordering
- Async processing coordination between services
- Rate-limited operations for external API calls
"""

from typing import Any, Optional, List, Dict, Callable, Generic, TypeVar
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
import json
from enum import Enum
from abc import ABC, abstractmethod

T = TypeVar('T')


class Priority(Enum):
    """Priority levels for judgment processing"""
    CRITICAL = 1    # System failures, urgent fixes
    HIGH = 2        # Supreme Court, House of Lords judgments
    NORMAL = 3      # Court of Appeal, High Court judgments
    LOW = 4         # Tribunal decisions, bulk imports


class TaskStatus(Enum):
    """Status of queued tasks"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class JudgmentTask:
    """Represents a judgment processing task in the queue"""
    task_id: str
    judgment_id: str
    operation: str  # 'import', 'reprocess', 'enrich', 'publish'
    priority: Priority
    created_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3
    status: TaskStatus = TaskStatus.PENDING
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Task {self.task_id}: {self.operation} {self.judgment_id} ({self.priority.name})"


class Queue(ABC, Generic[T]):
    """Abstract base class for queue implementations"""

    @abstractmethod
    def enqueue(self, item: T) -> None:
        """Add item to queue"""
        pass

    @abstractmethod
    def dequeue(self) -> Optional[T]:
        """Remove and return item from queue"""
        pass

    @abstractmethod
    def peek(self) -> Optional[T]:
        """View next item without removing"""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get queue size"""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        pass


class SimpleQueue(Queue[T]):
    """
    Simple FIFO queue implementation using deque for efficient operations.
    Suitable for basic judgment processing workflows.
    """

    def __init__(self):
        self._queue = deque()
        self._lock = threading.Lock()

    def enqueue(self, item: T) -> None:
        """Add item to the end of queue (FIFO)"""
        with self._lock:
            self._queue.append(item)

    def dequeue(self) -> Optional[T]:
        """Remove and return item from front of queue"""
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None

    def peek(self) -> Optional[T]:
        """View next item without removing it"""
        with self._lock:
            if self._queue:
                return self._queue[0]
            return None

    def size(self) -> int:
        """Get current queue size"""
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        """Check if queue is empty"""
        with self._lock:
            return len(self._queue) == 0

    def clear(self) -> None:
        """Clear all items from queue"""
        with self._lock:
            self._queue.clear()

    def to_list(self) -> List[T]:
        """Return queue contents as list (for inspection)"""
        with self._lock:
            return list(self._queue)


class PriorityQueue(Queue[JudgmentTask]):
    """
    Priority queue for judgment processing tasks.
    Higher priority items are processed first.
    """

    def __init__(self):
        self._queues: Dict[Priority, SimpleQueue[JudgmentTask]] = {
            priority: SimpleQueue() for priority in Priority
        }
        self._lock = threading.Lock()

    def enqueue(self, task: JudgmentTask) -> None:
        """Add task to appropriate priority queue"""
        with self._lock:
            self._queues[task.priority].enqueue(task)

    def dequeue(self) -> Optional[JudgmentTask]:
        """Remove and return highest priority task"""
        with self._lock:
            # Check queues in priority order
            for priority in Priority:
                queue = self._queues[priority]
                if not queue.is_empty():
                    return queue.dequeue()
            return None

    def peek(self) -> Optional[JudgmentTask]:
        """View next highest priority task without removing"""
        with self._lock:
            for priority in Priority:
                queue = self._queues[priority]
                if not queue.is_empty():
                    return queue.peek()
            return None

    def size(self) -> int:
        """Get total number of tasks across all priorities"""
        with self._lock:
            return sum(queue.size() for queue in self._queues.values())

    def is_empty(self) -> bool:
        """Check if all priority queues are empty"""
        with self._lock:
            return all(queue.is_empty() for queue in self._queues.values())

    def get_priority_counts(self) -> Dict[Priority, int]:
        """Get count of tasks by priority level"""
        with self._lock:
            return {priority: queue.size() for priority, queue in self._queues.items()}


class JudgmentProcessor:
    """
    High-level judgment processing system using queues.
    Coordinates multiple processing stages with proper error handling.
    """

    def __init__(self, max_workers: int = 3):
        self.processing_queue = PriorityQueue()
        self.completed_queue = SimpleQueue[JudgmentTask]()
        self.failed_queue = SimpleQueue[JudgmentTask]()
        self.max_workers = max_workers
        self.workers_running = False
        self.workers: List[threading.Thread] = []
        self._stats = {
            'processed': 0,
            'failed': 0,
            'retries': 0,
            'start_time': None
        }

        # Mock processing functions for different operations
        self._processors: Dict[str, Callable[[JudgmentTask], bool]] = {
            'import': self._process_import,
            'reprocess': self._process_reprocess,
            'enrich': self._process_enrichment,
            'publish': self._process_publish
        }

    def add_judgment_task(self, judgment_id: str, operation: str,
                         priority: Priority = Priority.NORMAL,
                         metadata: Dict[str, Any] = None) -> str:
        """Add a judgment processing task to the queue"""
        task_id = f"{operation}_{judgment_id}_{int(time.time())}"

        task = JudgmentTask(
            task_id=task_id,
            judgment_id=judgment_id,
            operation=operation,
            priority=priority,
            metadata=metadata or {}
        )

        self.processing_queue.enqueue(task)
        return task_id

    def start_processing(self) -> None:
        """Start worker threads to process queue"""
        if self.workers_running:
            return

        self.workers_running = True
        self._stats['start_time'] = datetime.now()

        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

        print(f"Started {self.max_workers} workers for judgment processing")

    def stop_processing(self) -> None:
        """Stop all worker threads"""
        self.workers_running = False
        for worker in self.workers:
            worker.join(timeout=5.0)
        self.workers.clear()
        print("Stopped all processing workers")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive queue status"""
        priority_counts = self.processing_queue.get_priority_counts()

        return {
            'pending_tasks': self.processing_queue.size(),
            'priority_breakdown': {p.name: count for p, count in priority_counts.items()},
            'completed_tasks': self.completed_queue.size(),
            'failed_tasks': self.failed_queue.size(),
            'workers_active': self.workers_running,
            'worker_count': len(self.workers),
            'statistics': self._stats.copy()
        }

    def get_recent_completed(self, limit: int = 10) -> List[JudgmentTask]:
        """Get recently completed tasks"""
        completed = self.completed_queue.to_list()
        return completed[-limit:] if len(completed) > limit else completed

    def get_failed_tasks(self) -> List[JudgmentTask]:
        """Get all failed tasks for review"""
        return self.failed_queue.to_list()

    def retry_failed_task(self, task_id: str) -> bool:
        """Retry a specific failed task"""
        failed_tasks = self.failed_queue.to_list()

        for task in failed_tasks:
            if task.task_id == task_id:
                # Reset task for retry
                task.status = TaskStatus.PENDING
                task.attempts = 0
                task.error_message = None

                # Remove from failed queue and add back to processing
                self.failed_queue._queue.remove(task)
                self.processing_queue.enqueue(task)
                return True

        return False

    def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop for processing tasks"""
        print(f"Worker {worker_id} started")

        while self.workers_running:
            try:
                # Get next task from queue
                task = self.processing_queue.dequeue()

                if task is None:
                    # No tasks available, wait briefly
                    time.sleep(0.1)
                    continue

                # Process the task
                self._process_task(task, worker_id)

            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                time.sleep(1.0)

        print(f"Worker {worker_id} stopped")

    def _process_task(self, task: JudgmentTask, worker_id: int) -> None:
        """Process a single judgment task"""
        task.status = TaskStatus.PROCESSING
        task.attempts += 1

        print(f"Worker {worker_id} processing: {task}")

        try:
            # Get appropriate processor
            processor = self._processors.get(task.operation)
            if not processor:
                raise ValueError(f"Unknown operation: {task.operation}")

            # Execute processing
            success = processor(task)

            if success:
                task.status = TaskStatus.COMPLETED
                self.completed_queue.enqueue(task)
                self._stats['processed'] += 1
                print(f" Completed: {task.task_id}")
            else:
                self._handle_task_failure(task)

        except Exception as e:
            task.error_message = str(e)
            self._handle_task_failure(task)

    def _handle_task_failure(self, task: JudgmentTask) -> None:
        """Handle task processing failure with retry logic"""
        if task.attempts < task.max_attempts:
            # Retry the task
            task.status = TaskStatus.RETRYING
            self.processing_queue.enqueue(task)
            self._stats['retries'] += 1
            print(f"  Retrying: {task.task_id} (attempt {task.attempts}/{task.max_attempts})")
        else:
            # Task failed permanently
            task.status = TaskStatus.FAILED
            self.failed_queue.enqueue(task)
            self._stats['failed'] += 1
            print(f" Failed: {task.task_id} - {task.error_message}")

    def _process_import(self, task: JudgmentTask) -> bool:
        """Simulate judgment import processing"""
        # Simulate processing time based on priority
        processing_time = {
            Priority.CRITICAL: 0.1,
            Priority.HIGH: 0.2,
            Priority.NORMAL: 0.5,
            Priority.LOW: 1.0
        }

        time.sleep(processing_time[task.priority])

        # Simulate occasional failures for demonstration
        if "fail" in task.judgment_id.lower():
            return False

        return True

    def _process_reprocess(self, task: JudgmentTask) -> bool:
        """Simulate judgment reprocessing"""
        time.sleep(0.3)
        return True

    def _process_enrichment(self, task: JudgmentTask) -> bool:
        """Simulate judgment enrichment (NLP, citations, etc.)"""
        time.sleep(0.8)
        return True

    def _process_publish(self, task: JudgmentTask) -> bool:
        """Simulate judgment publishing"""
        time.sleep(0.2)
        return True


class CircularBuffer(Queue[T]):
    """
    Fixed-size circular buffer implementation.
    Useful for maintaining recent processing history or logs.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self._buffer: List[Optional[T]] = [None] * capacity
        self._head = 0  # Points to next write position
        self._tail = 0  # Points to next read position
        self._size = 0
        self._lock = threading.Lock()

    def enqueue(self, item: T) -> None:
        """Add item to buffer, overwriting oldest if full"""
        with self._lock:
            self._buffer[self._head] = item
            self._head = (self._head + 1) % self.capacity

            if self._size < self.capacity:
                self._size += 1
            else:
                # Buffer is full, advance tail
                self._tail = (self._tail + 1) % self.capacity

    def dequeue(self) -> Optional[T]:
        """Remove and return oldest item"""
        with self._lock:
            if self._size == 0:
                return None

            item = self._buffer[self._tail]
            self._buffer[self._tail] = None
            self._tail = (self._tail + 1) % self.capacity
            self._size -= 1
            return item

    def peek(self) -> Optional[T]:
        """View oldest item without removing"""
        with self._lock:
            if self._size == 0:
                return None
            return self._buffer[self._tail]

    def size(self) -> int:
        """Get current number of items"""
        with self._lock:
            return self._size

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self._lock:
            return self._size == 0

    def is_full(self) -> bool:
        """Check if buffer is at capacity"""
        with self._lock:
            return self._size == self.capacity

    def to_list(self) -> List[T]:
        """Return buffer contents as list (oldest to newest)"""
        with self._lock:
            if self._size == 0:
                return []

            result = []
            current = self._tail
            for _ in range(self._size):
                result.append(self._buffer[current])
                current = (current + 1) % self.capacity
            return result


def demonstrate_judgment_queues():
    """Demonstrate queue implementations with UK legal data."""

    print("=== Judgment Processing Queues Demo ===\n")

    # Sample UK judgments for processing
    uk_judgments = [
        {
            'id': 'uksc_2023_15',
            'case_name': 'R (Miller) v Prime Minister',
            'citation': '[2023] UKSC 15',
            'court': 'UKSC',
            'priority': Priority.HIGH
        },
        {
            'id': 'ewca_2023_892',
            'case_name': 'Smith v Secretary of State',
            'citation': '[2023] EWCA Civ 892',
            'court': 'EWCA',
            'priority': Priority.NORMAL
        },
        {
            'id': 'ewhc_2023_1456',
            'case_name': 'Jones v Local Authority',
            'citation': '[2023] EWHC 1456 (Admin)',
            'court': 'EWHC',
            'priority': Priority.NORMAL
        },
        {
            'id': 'ukftt_2023_fail',  # This will simulate a failure
            'case_name': 'Brown v HMRC',
            'citation': '[2023] UKFTT 234 (TC)',
            'court': 'UKFTT',
            'priority': Priority.LOW
        }
    ]

    # 1. Simple Queue Demo
    print("1. SIMPLE FIFO QUEUE:")
    simple_queue = SimpleQueue[str]()

    for judgment in uk_judgments:
        simple_queue.enqueue(judgment['case_name'])
        print(f"   Enqueued: {judgment['case_name']}")

    print(f"\n   Queue size: {simple_queue.size()}")
    print(f"   Next item: {simple_queue.peek()}")

    print(f"\n   Processing queue:")
    while not simple_queue.is_empty():
        case = simple_queue.dequeue()
        print(f"   Processing: {case}")

    # 2. Priority Queue Demo
    print(f"\n2. PRIORITY QUEUE:")
    processor = JudgmentProcessor(max_workers=2)

    # Add tasks with different priorities
    for judgment in uk_judgments:
        task_id = processor.add_judgment_task(
            judgment_id=judgment['id'],
            operation='import',
            priority=judgment['priority'],
            metadata={'court': judgment['court'], 'citation': judgment['citation']}
        )
        print(f"   Added task: {task_id} ({judgment['priority'].name})")

    # Add some enrichment tasks
    processor.add_judgment_task('uksc_2023_15', 'enrich', Priority.HIGH)
    processor.add_judgment_task('ewca_2023_892', 'publish', Priority.NORMAL)

    # Show queue status
    status = processor.get_queue_status()
    print(f"\n   Queue status:")
    print(f"     Pending tasks: {status['pending_tasks']}")
    print(f"     Priority breakdown: {status['priority_breakdown']}")

    # 3. Process Tasks
    print(f"\n3. PROCESSING TASKS:")
    processor.start_processing()

    # Wait for processing to complete
    time.sleep(3.0)

    final_status = processor.get_queue_status()
    print(f"\n   Final status:")
    print(f"     Completed: {final_status['completed_tasks']}")
    print(f"     Failed: {final_status['failed_tasks']}")
    print(f"     Statistics: {final_status['statistics']}")

    # Show completed tasks
    completed = processor.get_recent_completed(5)
    if completed:
        print(f"\n   Recent completions:")
        for task in completed:
            print(f"      {task.judgment_id} ({task.operation})")

    # Show failed tasks
    failed = processor.get_failed_tasks()
    if failed:
        print(f"\n   Failed tasks:")
        for task in failed:
            print(f"      {task.judgment_id}: {task.error_message}")

    processor.stop_processing()

    # 4. Circular Buffer Demo
    print(f"\n4. CIRCULAR BUFFER (Recent History):")
    history_buffer = CircularBuffer[str](capacity=3)

    events = [
        "Import started for UKSC 2023/15",
        "Enrichment completed for EWCA 2023/892",
        "Publishing failed for EWHC 2023/1456",
        "Retry successful for EWHC 2023/1456",
        "New batch import initiated"
    ]

    for event in events:
        history_buffer.enqueue(event)
        print(f"   Event: {event}")
        if history_buffer.is_full():
            print(f"   Buffer full - oldest events will be overwritten")

    print(f"\n   Recent history (last {history_buffer.size()} events):")
    history = history_buffer.to_list()
    for i, event in enumerate(history, 1):
        print(f"     {i}. {event}")

    return {
        'simple_queue': simple_queue,
        'processor': processor,
        'history_buffer': history_buffer,
        'final_stats': final_status
    }


if __name__ == "__main__":
    demonstrate_judgment_queues()