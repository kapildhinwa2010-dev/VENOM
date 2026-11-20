"""Hard cap on tool calls per answer. Stops tool spam and infinite loops."""
from __future__ import annotations


class Budget:
    def __init__(self, max_calls: int = 3):
        self.max_calls = max_calls
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.max_calls:
            return False
        self.used += 1
        return True

    def reset(self) -> None:
        self.used = 0

    def state(self) -> dict:
        return {"used": self.used, "max": self.max_calls, "left": self.max_calls - self.used}
