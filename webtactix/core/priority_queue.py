# webtactix/core/priority_queue
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Optional

from webtactix.core.schemas import NodeId


@dataclass
class PriorityQueue:
    """
    Global FIFO queue.
    push appends to the right, pop removes from the left.
    """
    _q: Deque[NodeId] = field(default_factory=deque)

    def push(self, node_id: NodeId) -> None:
        self._q.append(node_id)

    def pop(self) -> Optional[NodeId]:
        if not self._q:
            return None
        return self._q.popleft()

    def __len__(self) -> int:
        return len(self._q)

    def empty(self) -> bool:
        return len(self._q) == 0