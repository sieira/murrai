import os
import tempfile
import urllib.request
from pathlib import Path

from murrai.adapters.corpus import DirTextCorpus

TRAIN_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStories-train.txt"
)
VALID_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStories-valid.txt"
)
TRAIN = "tinystories-train"
VALID = "tinystories-valid"
FILE = "tinystories.txt"
SMOKE_CHARS = 2_000_000
CHUNK = 1 << 20


class TinyStories:
    def __init__(self, root, source=urllib.request.urlopen):
        self._root = Path(root)
        self._source = source

    def fetch(self):
        self._split(TRAIN, TRAIN_URL)
        self._split(VALID, VALID_URL)
        self._smoke_slice()

    def train(self):
        return DirTextCorpus(self._root / TRAIN)

    def valid(self):
        return DirTextCorpus(self._root / VALID)

    def smoke(self):
        return DirTextCorpus(self._root)

    def _split(self, name, url):
        target = self._root / name / FILE
        if target.exists():
            return
        self._download(url, target)

    def _smoke_slice(self):
        target = self._root / FILE
        if target.exists():
            return
        valid = (self._root / VALID / FILE).read_text(encoding="utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(valid[:SMOKE_CHARS].strip() + "\n", encoding="utf-8")

    def _download(self, url, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as file, self._source(url) as response:
                while chunk := response.read(CHUNK):
                    file.write(chunk)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp, target)
        except BaseException:
            os.unlink(temp)
            raise
