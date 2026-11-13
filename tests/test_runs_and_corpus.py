from murrai.adapters.corpus import DirTextCorpus
from murrai.adapters.events import ListenerBus
from murrai.application.ports import Corpus, TrainingEvent
from murrai.application.runs import train_bigram, train_transformer_lm
from murrai.domain.bigram import Bigram


def test_run_trains_and_evaluates():
    class FakeCorpus:
        def text(self):
            return "aabbaabbaabbaabb"

    corpus: Corpus = FakeCorpus()
    result = train_bigram(corpus, holdout=0.2)
    assert isinstance(result.model, Bigram)
    assert result.held_out_chars > 0
    assert result.train_chars > 0
    assert 0.0 < result.held_out_perplexity < float("inf")


def test_run_held_out_is_last_slice():
    class FakeCorpus:
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
    class FakeCorpus:
        def text(self):
            return "aabbaabbaabbaabb"

    result = train_bigram(FakeCorpus(), holdout=0.2)
    expected = Bigram.fit(
        "aabbaabbaabbaabb"[: int(len("aabbaabbaabbaabb") * 0.8)],
        result.model.tokenizer,
    ).perplexity("aabbaabbaabbaabb"[int(len("aabbaabbaabbaabb") * 0.8) :])
    assert result.held_out_perplexity == expected


def test_dir_corpus_reads_txt_files(tmp_path):
    (tmp_path / "b.txt").write_text("second")
    (tmp_path / "a.txt").write_text("first")
    assert DirTextCorpus(tmp_path).text() == "first\nsecond"


def test_dir_corpus_ignores_other_files(tmp_path):
    (tmp_path / "a.txt").write_text("text")
    (tmp_path / "notes.md").write_text("ignored")
    assert DirTextCorpus(tmp_path).text() == "text"


def test_dir_corpus_accepts_plain_string_path(tmp_path):
    (tmp_path / "a.txt").write_text("text")
    assert DirTextCorpus(str(tmp_path)).text() == "text"


def test_run_reports_counting_progress_to_a_bus():
    class FakeCorpus:
        def text(self):
            return "aabbaabb" * 20

    bus = ListenerBus()
    seen = []
    bus.subscribe(seen.append)

    result = train_bigram(FakeCorpus(), holdout=0.2, bus=bus)

    assert seen
    assert all(isinstance(event, TrainingEvent) for event in seen)
    counting = [event for event in seen if event.phase == "counting"]
    assert counting[-1].step == counting[-1].total_steps == result.train_chars
    steps = [event.step for event in counting]
    assert steps == sorted(steps)
    assert {event.phase for event in seen} == {"counting", "tokenising"}


def test_the_tokenising_stage_reports_its_own_progress():
    class FakeCorpus:
        def text(self):
            return "aabbaabb" * 20

    bus = ListenerBus()
    seen = []
    bus.subscribe(seen.append)

    train_transformer_lm(FakeCorpus().text(), steps=2, batch=2, seed=0, bus=bus)

    phases = [event.phase for event in seen]
    assert "tokenising" in phases
    assert "training" in phases
    assert phases.index("tokenising") < phases.index("training")


def test_the_tokenising_stage_counts_down_to_its_own_total():
    class FakeCorpus:
        def text(self):
            return "aabbaabb" * 20

    bus = ListenerBus()
    seen = []
    bus.subscribe(seen.append)

    result = train_transformer_lm(
        FakeCorpus().text(),
        steps=2,
        batch=2,
        seed=0,
        bus=bus,
    )

    tokenising = [event for event in seen if event.phase == "tokenising"]
    assert tokenising[-1].total_steps == 300
    assert tokenising[-1].step == len(result.model.tokenizer.merges)
