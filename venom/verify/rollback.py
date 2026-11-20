"""Snapshot / restore any state so an experiment can always be undone."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional


class RollbackManager:
    def __init__(self, max_snapshots: int = 5):
        self.max_snapshots = max_snapshots
        self._snaps: List[Dict[str, Any]] = []

    @staticmethod
    def _id(state: Any) -> str:
        blob = json.dumps(state, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def snapshot(self, state: Any, tag: str = "") -> str:
        sid = self._id(state)
        self._snaps.append({"id": sid, "tag": tag, "state": copy.deepcopy(state)})
        while len(self._snaps) > self.max_snapshots:
            self._snaps.pop(0)
        return sid

    def restore(self, snap_id: Optional[str] = None) -> Any:
        if snap_id is None:
            if not self._snaps:
                return None
            snap = self._snaps[-1]
        else:
            snap = next((s for s in self._snaps if s["id"] == snap_id), None)
            if snap is None:
                return None
        return copy.deepcopy(snap["state"])

    def latest_id(self) -> Optional[str]:
        return self._snaps[-1]["id"] if self._snaps else None

    def list_snapshots(self) -> List[Dict[str, str]]:
        return [{"id": s["id"], "tag": s["tag"]} for s in self._snaps]

    def clear(self) -> None:
        self._snaps.clear()
