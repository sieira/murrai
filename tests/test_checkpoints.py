import pytest
import torch

from murrai.adapters.checkpoint import FileCheckpointStore, InMemoryCheckpointStore
from murrai.application.checkpoints import Snapshot, restore, snapshot_of
from murrai.application.runs import train_mlp_lm, train_neural_bigram
from murrai.domain.bigram import Bigram
from murrai.domain.mlp import MlpLM
from murrai.domain.neural_bigram import NeuralBigram
from murrai.domain.transformer import TransformerLM
from murrai.domain.vocabulary import Vocabulary
from murrai.presentation.chat import ChatSession


@pytest.fixture
def counted():
    return Bigram.fit("abab", Vocabulary.from_text("ab"))


@pytest.fixture
def neural():
    return train_neural_bigram("abab", steps=5, batch=8, seed=0).model


@pytest.fixture
def mlp():
    return train_mlp_lm("abab" * 20, steps=5, batch=8, seed=0).model


@pytest.fixture
def transformer():
    torch.manual_seed(0)
    return TransformerLM(
        Vocabulary.from_text("ababab"),
        depth=2,
        dimension=8,
        max_time=4,
    )


def test_the_counted_bigram_round_trips_through_its_state(counted):
    vocabulary = Vocabulary(counted.vocabulary.chars)
    rebuilt = Bigram.from_state(vocabulary, counted.state())
    torch.testing.assert_close(rebuilt.probs([0]), counted.probs([0]))
    torch.testing.assert_close(rebuilt.probs([]), counted.probs([]))


def test_the_neural_bigram_round_trips_through_its_state(neural):
    vocabulary = Vocabulary(neural.vocabulary.chars)
    rebuilt = NeuralBigram.from_state(vocabulary, neural.state())
    assert rebuilt.probs([0]) == pytest.approx(neural.probs([0]))


def test_the_mlp_round_trips_through_its_state(mlp):
    vocabulary = Vocabulary(mlp.vocabulary.chars)
    rebuilt = MlpLM.from_state(vocabulary, mlp.state())
    torch.testing.assert_close(rebuilt.probs([0, 1]), mlp.probs([0, 1]))


def test_the_transformer_round_trips_through_its_state(transformer):
    vocabulary = Vocabulary(transformer.vocabulary.chars)
    rebuilt = TransformerLM.from_state(vocabulary, transformer.state())
    torch.testing.assert_close(rebuilt.probs([0, 1]), transformer.probs([0, 1]))


def test_a_snapshot_carries_kind_chars_and_state(counted, neural, mlp, transformer):
    for model in (counted, neural, mlp, transformer):
        snapshot = snapshot_of(model)
        assert snapshot.kind == model.kind
        assert snapshot.chars == model.vocabulary.chars
        assert snapshot.state


def test_restore_rebuilds_each_model(counted, neural, mlp, transformer):
    for model in (counted, neural, mlp, transformer):
        rebuilt = restore(snapshot_of(model))
        assert rebuilt.kind == model.kind
        ids = [] if isinstance(model, (Bigram, NeuralBigram)) else [0, 1]
        original = model.probs(ids)
        after = rebuilt.probs(ids)
        assert torch.allclose(torch.as_tensor(after), torch.as_tensor(original))


def test_a_snapshot_starts_with_an_empty_info(counted):
    assert snapshot_of(counted).info == {}


def test_the_file_store_round_trips_the_info(tmp_path, counted):
    store = FileCheckpointStore(tmp_path)
    store.save(
        "counted-bigram",
        snapshot_of(
            counted,
            info={"steps": 3, "saved_at": "2027-01-15T08:00:00+00:00"},
        ),
    )
    assert store.load("counted-bigram").info == {
        "steps": 3,
        "saved_at": "2027-01-15T08:00:00+00:00",
    }


def test_the_file_store_loads_an_old_file_without_info(tmp_path, counted):
    path = tmp_path / "old.pt"
    with open(path, "wb") as file:
        torch.save(
            {
                "kind": counted.kind,
                "chars": counted.vocabulary.chars,
                "state": counted.state(),
            },
            file,
        )
    store = FileCheckpointStore(tmp_path)
    snapshot = store.load("old")
    assert snapshot.info == {}
    torch.testing.assert_close(restore(snapshot).probs([0]), counted.probs([0]))


def test_the_memory_store_carries_the_info(counted):
    store = InMemoryCheckpointStore()
    store.save("counted-bigram", snapshot_of(counted, info={"steps": 3}))
    assert store.load("counted-bigram").info == {"steps": 3}


def test_restore_rejects_an_unknown_kind():
    snapshot = Snapshot(kind="nonsense", chars=("a",), state={})
    with pytest.raises(ValueError):
        restore(snapshot)


def test_the_file_store_round_trips(tmp_path, mlp):
    store = FileCheckpointStore(tmp_path)
    store.save("windowed-mlp", snapshot_of(mlp))
    assert store.names() == ["windowed-mlp"]
    rebuilt = restore(store.load("windowed-mlp"))
    torch.testing.assert_close(rebuilt.probs([0, 1]), mlp.probs([0, 1]))


def test_the_file_store_lists_names_sorted(tmp_path, mlp, neural):
    store = FileCheckpointStore(tmp_path)
    store.save("windowed-mlp", snapshot_of(mlp))
    store.save("neural-bigram", snapshot_of(neural))
    assert store.names() == ["neural-bigram", "windowed-mlp"]


def test_the_file_store_rejects_a_missing_name(tmp_path):
    store = FileCheckpointStore(tmp_path)
    with pytest.raises(KeyError):
        store.load("windowed-mlp")


def test_an_interrupted_save_keeps_the_previous_checkpoint(
    tmp_path,
    neural,
    monkeypatch,
):
    store = FileCheckpointStore(tmp_path)
    store.save("neural-bigram", snapshot_of(neural))

    def interrupted_save(payload, destination):
        if hasattr(destination, "write"):
            destination.write(b"half-written")
        else:
            with open(destination, "wb") as file:
                file.write(b"half-written")
        raise KeyboardInterrupt

    monkeypatch.setattr("murrai.adapters.checkpoint.torch.save", interrupted_save)
    with pytest.raises(KeyboardInterrupt):
        store.save("neural-bigram", snapshot_of(neural))
    assert store.names() == ["neural-bigram"]
    rebuilt = restore(store.load("neural-bigram"))
    torch.testing.assert_close(rebuilt.probs([0]), neural.probs([0]))


def test_names_skips_an_unfinished_file(tmp_path):
    store = FileCheckpointStore(tmp_path)
    (tmp_path / "half-written.pt.tmp").write_bytes(b"junk")
    assert store.names() == []


class FakeCorpus:
    def text(self):
        return "abab" * 40


def test_a_fake_store_drives_a_tiny_training_run_end_to_end():
    corpus = FakeCorpus()
    store = InMemoryCheckpointStore()
    result = train_mlp_lm(corpus.text(), steps=5, batch=8, seed=0)
    store.save(result.model.kind, snapshot_of(result.model))
    model = restore(store.load("windowed-mlp"))
    session = ChatSession(model, seed=0)
    session.extend_user("ab")
    dice = session.dice()
    assert pytest.approx(sum(p for _, p in dice)) == 1.0
    assert session.rollout("ab", 5)
