import json
import os
import tempfile
from pathlib import Path

import torch

from murrai.application.checkpoints import (
    CheckpointRecord,
    GradientRun,
    RunSettings,
    RunState,
    Snapshot,
)
from murrai.domain.shape import Shape


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
            shape=snapshot.shape,
            weights=_state_to_cpu(snapshot.weights),
            run=snapshot.run,
        )
        _written_atomically(
            self._root,
            self._path(name),
            lambda file: torch.save(
                {
                    "kind": snapshot.kind,
                    "merges": snapshot.merges,
                    "shape": vars(snapshot.shape),
                    "weights": snapshot.weights,
                    "run": (
                        _state_to_cpu(snapshot.run.to_dict()) if snapshot.run else None
                    ),
                },
                file,
            ),
        )

    def save_record(self, name, record):
        data = {}
        if record.settings:
            data.update(
                saved_at=record.settings.saved_at,
                corpus=record.settings.corpus,
                seed=record.settings.seed,
            )
            if record.settings.corpus_chars is not None:
                data["corpus_chars"] = record.settings.corpus_chars
        if record.gradient:
            data.update(
                steps=record.gradient.steps,
                batch=record.gradient.batch,
                learning_rate=record.gradient.learning_rate,
            )
            if record.gradient.final_loss is not None:
                data["final_loss"] = record.gradient.final_loss
        if record.held_out_perplexity is not None:
            data["held_out_perplexity"] = record.held_out_perplexity
        _written_atomically(
            self._root,
            self._record_path(name),
            lambda file: file.write(json.dumps(data).encode()),
        )

    def load(self, name):
        path = self._path(name)
        if not path.exists():
            raise KeyError(name)
        data = torch.load(path, weights_only=True)
        return Snapshot(
            kind=data["kind"],
            merges=data["merges"],
            shape=Shape(**data["shape"]),
            weights=data["weights"],
            run=RunState(**data["run"]) if data.get("run") else None,
        )

    def record(self, name):
        path = self._record_path(name)
        if not path.exists():
            return CheckpointRecord()
        data = json.loads(path.read_text())
        return CheckpointRecord(
            settings=(
                RunSettings(
                    saved_at=data["saved_at"],
                    corpus=data["corpus"],
                    seed=data["seed"],
                    corpus_chars=data.get("corpus_chars"),
                )
                if "saved_at" in data
                else None
            ),
            gradient=(
                GradientRun(
                    steps=data["steps"],
                    batch=data["batch"],
                    learning_rate=data["learning_rate"],
                    final_loss=data.get("final_loss"),
                )
                if "steps" in data
                else None
            ),
            held_out_perplexity=data.get("held_out_perplexity"),
        )

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
