from pathlib import Path


class DirTextCorpus:
    def __init__(self, directory):
        self._directory = Path(directory)

    def text(self):
        return "\n".join(p.read_text() for p in sorted(self._directory.glob("*.txt")))


class TextCorpus:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text
