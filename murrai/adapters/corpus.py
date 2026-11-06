from pathlib import Path


class DirTextCorpus:
    def __init__(self, directory):
        self._directory = Path(directory)

    def text(self):
        return "".join(p.read_text() for p in sorted(self._directory.glob("*.txt")))
