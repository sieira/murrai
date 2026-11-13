from array import array
from collections import Counter
from itertools import islice


class TokenizerEncoderDecoder:
    def __init__(self, merges=()):
        self.merges = tuple(merges)

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def fit(self, text, num_merges, on_progress=None):
        ids = array("i", iter(text.encode("utf-8")))
        counts = Counter(zip(ids, islice(ids, 1, None), strict=False))
        merges = []
        for _ in range(num_merges):
            if not counts:
                break
            best = min(counts, key=lambda p: (-counts[p], p))
            merges.append(best)
            ids = _merge_best(ids, best, 256 + len(merges) - 1, counts)
            if on_progress is not None:
                on_progress(len(merges), num_merges)
        self.merges = tuple(merges)
        return self

    def encode(self, text, on_progress=None):
        ids = array("i", iter(text.encode("utf-8")))
        for rank, pair in enumerate(self.merges):
            ids = _merge_best(ids, pair, 256 + rank, None)
            if on_progress is not None:
                on_progress(rank + 1, len(self.merges))
        return list(ids)

    @property
    def pieces(self):
        return self._render("replace")

    @property
    def labels(self):
        return self._render("backslashreplace")

    def _render(self, errors):
        vocab = _vocab(self.merges)
        return tuple(vocab[i].decode("utf-8", errors=errors) for i in range(len(vocab)))

    def decode(self, ids):
        vocab = _vocab(self.merges)
        return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


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


def _merge_best(ids, best, new_id, counts):
    # An occurrence's left neighbour may itself be a merged occurrence two
    # positions back; the pair spanning the two is then dropped by the later
    # occurrence, so the earlier one must not drop it twice.
    first, second = best
    starts = []
    pos = 0
    length = len(ids)
    while True:
        try:
            pos = ids.index(first, pos)
        except ValueError:
            break
        if pos + 1 < length and ids[pos + 1] == second:
            starts.append(pos)
            pos += 2
        else:
            pos += 1

    out = ids[:0]
    previous = None
    prev_end = 0
    for pos in starts:
        out.extend(ids[prev_end:pos])
        left_old = ids[pos - 1] if pos > 0 else None
        left_new = new_id if previous == pos - 2 else left_old
        right = ids[pos + 2] if pos + 2 < length else None
        followed = right == first and pos + 3 < length and ids[pos + 3] == second
        if counts is not None:
            if left_old is not None:
                pair = (left_old, first)
                remaining = counts[pair] - 1
                if remaining:
                    counts[pair] = remaining
                else:
                    del counts[pair]
            remaining = counts[best] - 1
            if remaining:
                counts[best] = remaining
            else:
                del counts[best]
            if right is not None and not followed:
                pair = (second, right)
                remaining = counts[pair] - 1
                if remaining:
                    counts[pair] = remaining
                else:
                    del counts[pair]
            if left_new is not None:
                counts[(left_new, new_id)] += 1
            if right is not None and not followed:
                counts[(new_id, right)] += 1
        out.append(new_id)
        previous = pos
        prev_end = pos + 2
    out.extend(ids[prev_end:])
    return out


def _vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for rank, (a, b) in enumerate(merges):
        vocab[256 + rank] = vocab[a] + vocab[b]
    return vocab
