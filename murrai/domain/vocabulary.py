class Vocabulary:
    def __init__(self, chars):
        self.chars = tuple(chars)
        self._ids = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def size(self):
        return len(self.chars)

    def grown_with(self, chars):
        fresh = sorted(set(chars) - set(self.chars))
        return Vocabulary(self.chars + tuple(fresh))

    def encode(self, text):
        try:
            return [self._ids[c] for c in text]
        except KeyError as e:
            raise ValueError(f"unknown char {e.args[0]!r}") from e

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)
