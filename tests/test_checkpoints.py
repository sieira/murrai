import json
from dataclasses import FrozenInstanceError, dataclass

import pytest
import torch

from murrai.adapters.checkpoint import FileCheckpointStore, InMemoryCheckpointStore
from murrai.application.checkpoints import (
    CheckpointRecord,
    GradientRun,
    RunSettings,
    RunState,
    Snapshot,
    restore,
    snapshot_of,
)
from murrai.application.runs import train_mlp_lm, train_neural_bigram
from murrai.domain.bigram import Bigram
from murrai.domain.mlp import MlpLM
from murrai.domain.neural_bigram import NeuralBigram
from murrai.domain.shape import Shape
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.domain.transformer import TransformerLM
from murrai.presentation.chat import ChatSession


@pytest.fixture
def counted():
    return Bigram.fit("abab", TokenizerEncoderDecoder())


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
        TokenizerEncoderDecoder(),
        depth=2,
        dimension=8,
        max_time=4,
    )


def test_the_counted_bigram_round_trips_through_its_state(counted):
    tokenizer = TokenizerEncoderDecoder(counted.tokenizer.merges)
    rebuilt = Bigram.from_state(tokenizer, *counted.state())
    torch.testing.assert_close(rebuilt.probs([0]), counted.probs([0]))
    torch.testing.assert_close(rebuilt.probs([]), counted.probs([]))


def test_the_neural_bigram_round_trips_through_its_state(neural):
    tokenizer = TokenizerEncoderDecoder(neural.tokenizer.merges)
    rebuilt = NeuralBigram.from_state(tokenizer, *neural.state())
    assert rebuilt.probs([0]) == pytest.approx(neural.probs([0]))


def test_the_mlp_round_trips_through_its_state(mlp):
    tokenizer = TokenizerEncoderDecoder(mlp.tokenizer.merges)
    rebuilt = MlpLM.from_state(tokenizer, *mlp.state())
    torch.testing.assert_close(rebuilt.probs([0, 1]), mlp.probs([0, 1]))


def test_the_transformer_round_trips_through_its_state(transformer):
    tokenizer = TokenizerEncoderDecoder(transformer.tokenizer.merges)
    rebuilt = TransformerLM.from_state(tokenizer, *transformer.state())
    torch.testing.assert_close(rebuilt.probs([0, 1]), transformer.probs([0, 1]))


def _same_weights(left, right):
    if isinstance(left, dict):
        return all(_same_weights(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return all(
            _same_weights(item, other) for item, other in zip(left, right, strict=False)
        )
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    return left == right


def test_a_snapshot_carries_the_shape_and_the_weights(
    counted,
    neural,
    mlp,
    transformer,
):
    for model in (counted, neural, mlp, transformer):
        snapshot = snapshot_of(model)
        shape, weights = model.state()
        assert snapshot.kind == model.kind
        assert snapshot.merges == model.tokenizer.merges
        assert snapshot.shape == shape
        assert _same_weights(snapshot.weights, weights)


def test_the_run_state_is_frozen():
    run = RunState(
        step=1,
        rate=0.1,
        best=1.0,
        best_step=0,
        train_seconds=1.0,
        optimizer=None,
        loss_curve=[0.5],
        rate_curve=[0.1],
        validation_curve=[],
    )
    with pytest.raises(FrozenInstanceError):
        run.step = 2


def test_the_run_state_copies_its_curves():
    loss_curve = [0.5]
    rate_curve = [0.1]
    validation_curve = [(1, 0.5)]
    run = RunState(
        step=1,
        rate=0.1,
        best=1.0,
        best_step=0,
        train_seconds=1.0,
        optimizer=None,
        loss_curve=loss_curve,
        rate_curve=rate_curve,
        validation_curve=validation_curve,
    )
    loss_curve.append(0.4)
    rate_curve.append(0.2)
    validation_curve.append((2, 0.4))
    assert run.loss_curve == [0.5]
    assert run.rate_curve == [0.1]
    assert run.validation_curve == [(1, 0.5)]


def test_the_run_state_copies_its_optimizer():
    optimizer = {"state": {"momentum": torch.tensor([1.0])}}
    run = RunState(
        step=1,
        rate=0.1,
        best=1.0,
        best_step=0,
        train_seconds=1.0,
        optimizer=optimizer,
        loss_curve=[],
        rate_curve=[],
        validation_curve=[],
    )
    optimizer["state"]["momentum"][0] = 99.0
    assert run.optimizer["state"]["momentum"][0] == 1.0


def test_the_run_state_resumes_with_fresh_curves():
    saved = RunState(
        step=10,
        rate=0.1,
        best=0.5,
        best_step=5,
        train_seconds=10.0,
        optimizer={"state": {}},
        loss_curve=[0.5],
        rate_curve=[0.1],
        validation_curve=[],
    )
    resumed = saved.to_dict()
    resumed["loss_curve"].append(0.4)
    assert saved.loss_curve == [0.5]
    assert resumed["loss_curve"] == [0.5, 0.4]


def test_restore_rebuilds_each_model(counted, neural, mlp, transformer):
    for model in (counted, neural, mlp, transformer):
        rebuilt = restore(snapshot_of(model))
        assert rebuilt.kind == model.kind
        ids = [] if isinstance(model, (Bigram, NeuralBigram)) else [0, 1]
        original = model.probs(ids)
        after = rebuilt.probs(ids)
        assert torch.allclose(torch.as_tensor(after), torch.as_tensor(original))


def test_a_snapshot_round_trips_the_tokenizer_merges():
    tokenizer = TokenizerEncoderDecoder().fit("abab", num_merges=2)
    model = Bigram.fit("abab", tokenizer)
    restored = restore(snapshot_of(model))
    assert restored.tokenizer.merges == tokenizer.merges
    assert restored.tokenizer.encode("abab") == tokenizer.encode("abab")
    torch.testing.assert_close(restored.probs([ord("a")]), model.probs([ord("a")]))


def test_a_snapshot_is_frozen(counted):
    snapshot = snapshot_of(counted)
    with pytest.raises(FrozenInstanceError):
        snapshot.kind = "nonsense"


def test_the_file_store_keeps_the_record_beside_the_weights(tmp_path, counted):
    store = FileCheckpointStore(tmp_path)
    record = CheckpointRecord(
        settings=RunSettings(
            saved_at="2027-01-15T08:00:00+00:00",
            corpus="data/corpus",
            seed=0,
        ),
        gradient=GradientRun(steps=3, batch=64, learning_rate=3.0),
    )
    store.save("counted-bigram", snapshot_of(counted))
    store.save_record("counted-bigram", record)

    assert store.record("counted-bigram") == record


def test_the_record_is_frozen():
    record = CheckpointRecord()
    with pytest.raises(FrozenInstanceError):
        record.held_out_perplexity = 2.0


def test_the_store_translates_the_record_to_flat_json(tmp_path, counted):
    store = FileCheckpointStore(tmp_path)
    store.save("counted-bigram", snapshot_of(counted))
    store.save_record(
        "counted-bigram",
        CheckpointRecord(
            settings=RunSettings(
                saved_at="2027-01-15T08:00:00+00:00",
                corpus="data/corpus",
                seed=0,
            ),
            gradient=GradientRun(steps=3, batch=64, learning_rate=3.0, final_loss=0.5),
            held_out_perplexity=1.5,
        ),
    )

    data = json.loads((tmp_path / "counted-bigram.json").read_text())
    assert data == {
        "saved_at": "2027-01-15T08:00:00+00:00",
        "corpus": "data/corpus",
        "seed": 0,
        "steps": 3,
        "batch": 64,
        "learning_rate": 3.0,
        "final_loss": 0.5,
        "held_out_perplexity": 1.5,
    }
    assert store.record("counted-bigram").gradient.final_loss == 0.5


def test_reading_the_record_never_opens_the_weights(tmp_path, counted, monkeypatch):
    store = FileCheckpointStore(tmp_path)
    store.save("counted-bigram", snapshot_of(counted))
    store.save_record(
        "counted-bigram",
        CheckpointRecord(gradient=GradientRun(steps=3, batch=64, learning_rate=3.0)),
    )

    def refuse(path, *args, **kwargs):
        raise AssertionError("reading the record opened the weights")

    monkeypatch.setattr("murrai.adapters.checkpoint.torch.load", refuse)
    assert store.record("counted-bigram").gradient.steps == 3


def test_a_checkpoint_without_a_record_file_reads_an_empty_record(tmp_path, counted):
    store = FileCheckpointStore(tmp_path)
    store.save("counted-bigram", snapshot_of(counted))

    assert store.record("counted-bigram") == CheckpointRecord()


def test_the_memory_store_carries_the_record(counted):
    store = InMemoryCheckpointStore()
    store.save("counted-bigram", snapshot_of(counted))
    store.save_record(
        "counted-bigram",
        CheckpointRecord(gradient=GradientRun(steps=3, batch=64, learning_rate=3.0)),
    )

    assert store.record("counted-bigram").gradient.steps == 3


def test_restore_rejects_an_unknown_kind():
    snapshot = Snapshot(kind="nonsense", merges=(), shape=Shape(), weights=None)
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


@dataclass(frozen=True)
class StreakRun(RunState):
    streak: int = 0


def test_the_store_saves_the_run_state_in_the_value_object_s_own_form(tmp_path):
    run = StreakRun(**RunState.fresh(0.5).to_dict(), streak=7)
    snapshot = Snapshot(
        kind="transformer",
        merges=[],
        shape=Shape(),
        weights={},
        run=run,
    )
    store = FileCheckpointStore(tmp_path / "checkpoints")
    store.save("streak", snapshot)

    loaded = store.load("streak")

    assert type(loaded.run) is RunState
    assert loaded.run == RunState.fresh(0.5)
