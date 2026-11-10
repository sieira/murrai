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
        Vocabulary.from_text("ababab"), depth=2, dimension=8, max_time=4
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


class FakeDataSource:
    def text(self):
        return "abab" * 40


def test_a_fake_store_drives_a_tiny_training_run_end_to_end():
    source = FakeDataSource()
    store = InMemoryCheckpointStore()
    result = train_mlp_lm(source.text(), steps=5, batch=8, seed=0)
    store.save(result.model.kind, snapshot_of(result.model))
    model = restore(store.load("windowed-mlp"))
    session = ChatSession(model, seed=0)
    session.extend_user("ab")
    dice = session.dice()
    assert pytest.approx(sum(p for _, p in dice)) == 1.0
    assert session.rollout("ab", 5)
