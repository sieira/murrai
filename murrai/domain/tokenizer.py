from itertools import pairwise


class TokenizerEncoderDecoder:
    def __init__(self):
        self.merges = ()

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def fit(self, text, num_merges):
        ids = list(text.encode("utf-8"))
        merges = []
        for _ in range(num_merges):
            counts = {}
            for pair in pairwise(ids):
                counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            best = min(counts, key=lambda p: (-counts[p], p))
            merges.append(best)
            ids = _replace(ids, best, 256 + len(merges) - 1)
        self.merges = tuple(merges)
        return self

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        for rank, pair in enumerate(self.merges):
            ids = _replace(ids, pair, 256 + rank)
        return ids

    def decode(self, ids):
        vocab = _vocab(self.merges)
        return b"".join(vocab[i] for i in ids).decode("utf-8")


def _replace(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for rank, (a, b) in enumerate(merges):
        vocab[256 + rank] = vocab[a] + vocab[b]
    return vocab
