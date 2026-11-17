import json
from pathlib import Path

from murrai.adapters.checkpoint import written_atomically
from murrai.application.sessions import StorySave


def _payload(save):
    return {
        "checkpoint": save.checkpoint,
        "ids": list(save.ids),
        "origins": list(save.origins),
    }


def _save_of(name, data):
    return StorySave(
        ids=tuple(data["ids"]),
        origins=tuple(data["origins"]),
        checkpoint=data["checkpoint"],
    )


class FileStoryStore:
    def __init__(self, root):
        self._root = Path(root)

    def _path(self, name):
        return self._root / f"{name}.json"

    def save(self, name, save):
        payload = _payload(save)
        written_atomically(
            self._root,
            self._path(name),
            lambda file: file.write(json.dumps(payload).encode()),
        )

    def load(self, name):
        path = self._path(name)
        if not path.exists():
            raise KeyError(name)
        return _save_of(name, json.loads(path.read_text()))

    def names(self):
        return sorted(path.stem for path in self._root.glob("*.json"))


class InMemoryStoryStore:
    def __init__(self):
        self._saves: dict[str, StorySave] = {}

    def save(self, name, save):
        self._saves[name] = save

    def load(self, name):
        if name not in self._saves:
            raise KeyError(name)
        return self._saves[name]

    def names(self):
        return sorted(self._saves)
