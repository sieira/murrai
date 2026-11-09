from pathlib import Path

import torch

from murrai.application.checkpoints import Snapshot


class FileCheckpointStore:
    def __init__(self, root):
        self._root = Path(root)

    def _path(self, name):
        return self._root / f"{name}.pt"

    def save(self, name, snapshot):
        self._root.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": snapshot.kind,
                "chars": snapshot.chars,
                "state": snapshot.state,
            },
            self._path(name),
        )

    def load(self, name):
        path = self._path(name)
        if not path.exists():
            raise KeyError(name)
        data = torch.load(path)
        return Snapshot(kind=data["kind"], chars=data["chars"], state=data["state"])

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
