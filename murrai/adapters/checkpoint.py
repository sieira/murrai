import os
import tempfile
from pathlib import Path

import torch

from murrai.application.checkpoints import Snapshot


def _state_to_cpu(state):
    if isinstance(state, torch.Tensor):
        return state.detach().to("cpu")
    if isinstance(state, dict):
        return {key: _state_to_cpu(value) for key, value in state.items()}
    if isinstance(state, list):
        return [_state_to_cpu(value) for value in state]
    if isinstance(state, tuple):
        return tuple(_state_to_cpu(value) for value in state)
    return state


class FileCheckpointStore:
    def __init__(self, root):
        self._root = Path(root)

    def _path(self, name):
        return self._root / f"{name}.pt"

    def save(self, name, snapshot):
        snapshot = Snapshot(
            kind=snapshot.kind,
            merges=snapshot.merges,
            state=_state_to_cpu(snapshot.state),
        )
        self._root.mkdir(parents=True, exist_ok=True)
        descriptor, temp = tempfile.mkstemp(dir=self._root, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as file:
                torch.save(
                    {
                        "kind": snapshot.kind,
                        "merges": snapshot.merges,
                        "state": snapshot.state,
                    },
                    file,
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp, self._path(name))
        except BaseException:
            os.unlink(temp)
            raise

    def load(self, name):
        path = self._path(name)
        if not path.exists():
            raise KeyError(name)
        data = torch.load(path)
        return Snapshot(kind=data["kind"], merges=data["merges"], state=data["state"])

    def names(self):
        return sorted(path.stem for path in self._root.glob("*.pt"))


class InMemoryCheckpointStore:
    def __init__(self):
        self._snapshots = {}

    def save(self, name, snapshot):
        self._snapshots[name] = snapshot

    def load(self, name):
        if name not in self._snapshots:
            raise KeyError(name)
        return self._snapshots[name]

    def names(self):
        return sorted(self._snapshots)
