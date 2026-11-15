import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from murrai.application.checkpoints import CheckpointRecord, Snapshot


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


def _written_atomically(root, path, write):
    root.mkdir(parents=True, exist_ok=True)
    descriptor, temp = tempfile.mkstemp(dir=root, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    except BaseException:
        os.unlink(temp)
        raise


class FileCheckpointStore:
    def __init__(self, root):
        self._root = Path(root)

    def _path(self, name):
        return self._root / f"{name}.pt"

    def _record_path(self, name):
        return self._root / f"{name}.json"

    def save(self, name, snapshot):
        snapshot = Snapshot(
            kind=snapshot.kind,
            merges=snapshot.merges,
            state=_state_to_cpu(snapshot.state),
        )
        _written_atomically(
            self._root,
            self._path(name),
            lambda file: torch.save(
                {
                    "kind": snapshot.kind,
                    "merges": snapshot.merges,
                    "state": snapshot.state,
                },
                file,
            ),
        )

    def save_record(self, name, record):
        _written_atomically(
            self._root,
            self._record_path(name),
            lambda file: file.write(json.dumps(asdict(record)).encode()),
        )

    def load(self, name):
        path = self._path(name)
        if not path.exists():
            raise KeyError(name)
        data = torch.load(path, weights_only=True)
        return Snapshot(kind=data["kind"], merges=data["merges"], state=data["state"])

    def record(self, name):
        path = self._record_path(name)
        if not path.exists():
            return CheckpointRecord()
        return CheckpointRecord(**json.loads(path.read_text()))

    def names(self):
        return sorted(path.stem for path in self._root.glob("*.pt"))


class InMemoryCheckpointStore:
    def __init__(self):
        self._snapshots = {}
        self._records = {}

    def save(self, name, snapshot):
        self._snapshots[name] = snapshot

    def save_record(self, name, record):
        self._records[name] = record

    def load(self, name):
        if name not in self._snapshots:
            raise KeyError(name)
        return self._snapshots[name]

    def record(self, name):
        return self._records.get(name, CheckpointRecord())

    def names(self):
        return sorted(self._snapshots)
