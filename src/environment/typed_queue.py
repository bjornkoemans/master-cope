from collections import deque
from typing import Deque, Generic, TypeVar

T = TypeVar("T")


class Queue(Generic[T]):
    """A simple FIFO queue implementation"""

    def __init__(self) -> None:
        self.queue: Deque[T] = deque()

    def enqueue(self, item: T) -> None:
        self.queue.append(item)

    def dequeue(self) -> T | None:
        if self.is_empty():
            return None
        return self.queue.popleft()

    def peek(self, i=0) -> T | None:
        if self.is_empty():
            return None
        return self.queue[i]

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def size(self) -> int:
        return len(self.queue)

    def __len__(self) -> int:
        return self.size()

    def __repr__(self) -> str:
        return f"Queue({list(self.queue)})"

    def __str__(self) -> str:
        return f"Queue with {len(self.queue)} items: {list(self.queue)}"

    def __iter__(self):
        return iter(self.queue)

    def __contains__(self, item: T) -> bool:
        return item in self.queue
