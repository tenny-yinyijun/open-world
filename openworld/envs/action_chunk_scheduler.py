from typing import Any, List, Optional


class ActionChunkScheduler:
    """Buffers per-step actions and releases fixed-size chunks for world-model calls.

    The scheduler accumulates individual actions and signals readiness once
    enough actions have been collected to form a complete chunk.  The chunk
    size is configurable (default 15) to support variable-size scheduling
    policies in the future.
    """

    def __init__(self, chunk_size: int = 15):
        self.chunk_size = chunk_size
        self._buffer: List[Any] = []

    def reset(self) -> None:
        """Clear the action buffer."""
        self._buffer = []

    def append(self, action: Any) -> None:
        """Add a single action to the buffer."""
        self._buffer.append(action)

    def is_ready(self) -> bool:
        """Return True when the buffer contains enough actions for one chunk."""
        return len(self._buffer) >= self.chunk_size

    def get_chunk(self) -> List[Any]:
        """Pop and return the next chunk of actions.

        Raises:
            RuntimeError: If fewer than ``chunk_size`` actions are buffered.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Not enough actions buffered: {len(self._buffer)} < {self.chunk_size}"
            )
        chunk = self._buffer[: self.chunk_size]
        self._buffer = self._buffer[self.chunk_size :]
        return chunk

    def num_buffered(self) -> int:
        """Return the number of actions currently in the buffer."""
        return len(self._buffer)
