import pytest

from murrai.adapters.corpus import DirTextCorpus
from murrai.adapters.events import ListenerBus
from murrai.application.ports import Corpus, TrainingEvent
from murrai.application.runs import (
    train_bigram,
    train_mlp_lm,
    train_transformer_lm,
)
from murrai.domain.bigram import Bigram


class FakeCorpus:
    def text(self):
        return "aabbaabbaabbaabb"


def test_the_trainers_take_held_out_text_and_nothing_else():
    with pytest.raises(TypeError):
        train_bigram(FakeCorpus(), holdout=0.2)
    with pytest.raises(TypeError):
        train_mlp_lm("ab" * 40, steps=1, batch=4, seed=0, holdout=0.2)
    with pytest.raises(TypeError):
        train_transformer_lm(
            "ab" * 40,
            steps=1,
            batch=4,
            seed=0,
            depth=1,
            dimension=8,
            max_time=8,
            holdout=0.2,
        )


def test_run_trains_and_evaluates():
    corpus: Corpus = FakeCorpus()
    result = train_bigram(corpus, held_out_text="aabb")
    assert isinstance(result.model, Bigram)
    assert result.train_chars == 16
    assert result.held_out_chars == 4
    assert 0.0 < result.held_out_perplexity < float("inf")


def test_no_held_out_text_means_no_perplexity():
    assert train_bigram(FakeCorpus()).held_out_perplexity is None
    assert (
        train_mlp_lm(
            "aabbaabbaabbaabb",
            steps=2,
            batch=4,
            seed=0,
        ).held_out_perplexity
        is None
    )
    assert (
        train_transformer_lm(
            "aabbaabbaabbaabb",
            steps=2,
            batch=4,
            seed=0,
            depth=1,
            dimension=8,
            max_time=8,
        ).held_out_perplexity
        is None
    )


def test_the_trainer_trains_on_the_whole_corpus_and_evaluates_on_the_held_out_text():
    class TracingSource:
        def __init__(self, seen):
            self.seen = seen

        def text(self):
            return "abcd"

    result = train_bigram(TracingSource(True), held_out_text="d")
    assert result.train_chars == 4
    assert result.held_out_chars == 1


def test_run_perplexity_matches_domain_math():
    result = train_bigram(FakeCorpus(), held_out_text="aabb")
    expected = Bigram.fit(
        "aabbaabbaabbaabb",
        result.model.tokenizer,
    ).perplexity("aabb")
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

    result = train_bigram(FakeCorpus(), bus=bus)
    bus.flush()

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

    train_transformer_lm(
        FakeCorpus().text(),
        steps=2,
        batch=2,
        seed=0,
        bus=bus,
    )
    bus.flush()

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
    bus.flush()

    tokenising = [event for event in seen if event.phase == "tokenising"]
    assert tokenising[-1].total_steps == 300
    assert tokenising[-1].step == len(result.model.tokenizer.merges)


def test_the_encoding_stage_reports_its_own_progress():
    class FakeCorpus:
        def text(self):
            return "aabbaabb" * 20

    bus = ListenerBus()
    seen = []
    bus.subscribe(seen.append)

    train_transformer_lm(
        FakeCorpus().text(),
        steps=2,
        batch=2,
        seed=0,
        held_out_text="ab" * 40,
        bus=bus,
    )

    phases = [event.phase for event in seen]
    assert "encoding" in phases
    assert phases.index("tokenising") < phases.index("encoding")
    assert phases.index("encoding") < phases.index("training")
