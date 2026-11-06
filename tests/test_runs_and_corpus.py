from murrai.adapters.corpus import DirTextCorpus
from murrai.application.ports import DataSource
from murrai.application.runs import train_bigram
from murrai.domain.bigram import Bigram


def test_run_trains_and_evaluates():
    class FakeSource:
        def text(self):
            return "aabbaabbaabbaabb"

    source: DataSource = FakeSource()
    result = train_bigram(source, holdout=0.2)
    assert isinstance(result.model, Bigram)
    assert result.held_out_chars > 0
    assert result.train_chars > 0
    assert 0.0 < result.held_out_perplexity < float("inf")


def test_run_held_out_is_last_slice():
    class FakeSource:
        def text(self):
            return "aabbaabbaabbaabb"

    class TracingSource:
        def __init__(self, seen):
            self.seen = seen

        def text(self):
            return "abcd"

    result = train_bigram(TracingSource(True), holdout=0.25)
    assert result.train_chars == 3
    assert result.held_out_chars == 1


def test_run_perplexity_matches_domain_math():
    class FakeSource:
        def text(self):
            return "aabbaabbaabbaabb"

    result = train_bigram(FakeSource(), holdout=0.2)
    expected = Bigram.fit(
        "aabbaabbaabbaabb"[: int(len("aabbaabbaabbaabb") * 0.8)],
        result.model.vocabulary,
    ).perplexity("aabbaabbaabbaabb"[int(len("aabbaabbaabbaabb") * 0.8) :])
    assert result.held_out_perplexity == expected


def test_dir_corpus_reads_txt_files(tmp_path):
    (tmp_path / "b.txt").write_text("second")
    (tmp_path / "a.txt").write_text("first")
    assert DirTextCorpus(tmp_path).text() == "firstsecond"


def test_dir_corpus_ignores_other_files(tmp_path):
    (tmp_path / "a.txt").write_text("text")
    (tmp_path / "notes.md").write_text("ignored")
    assert DirTextCorpus(tmp_path).text() == "text"


def test_dir_corpus_accepts_plain_string_path(tmp_path):
    (tmp_path / "a.txt").write_text("text")
    assert DirTextCorpus(str(tmp_path)).text() == "text"
