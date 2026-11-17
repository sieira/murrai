import pytest

from murrai.adapters.checkpoint import FileCheckpointStore
from murrai.adapters.corpus import TextCorpus
from murrai.composition.train import train_transformer_from_corpus


def test_the_train_composition_targets_the_ten_million_parameter_model():
    result = train_transformer_from_corpus(
        TextCorpus("ab" * 400),
        held_out=TextCorpus("ab" * 80),
        steps=2,
        batch=8,
        seed=0,
    )
    params = sum(p.numel() for p in result.model.parameters())
    assert 9_000_000 <= params <= 11_000_000
    assert result.held_out_perplexity is not None


def test_a_train_run_resumes_from_its_latest_checkpoint(tmp_path):
    whole_store = FileCheckpointStore(tmp_path / "whole")
    part_store = FileCheckpointStore(tmp_path / "part")
    common = {
        "corpus": TextCorpus("ab" * 400),
        "held_out": TextCorpus("ab" * 80),
        "batch": 8,
        "seed": 0,
        "checkpoint_every": 2,
        "checkpoint_name": "run",
        "corpus_name": "smoke-train",
        "validate_every": 2,
    }

    whole = train_transformer_from_corpus(steps=6, store=whole_store, **common)
    train_transformer_from_corpus(steps=4, store=part_store, **common)
    resumed = train_transformer_from_corpus(steps=6, store=part_store, **common)

    assert resumed.steps == 6
    assert resumed.loss_curve == whole.loss_curve[4:]
    assert resumed.validation_curve == whole.validation_curve[3:]


def test_the_train_checkpoint_record_carries_the_run(tmp_path):
    store = FileCheckpointStore(tmp_path / "checkpoints")
    result = train_transformer_from_corpus(
        TextCorpus("ab" * 400),
        held_out=TextCorpus("ab" * 80),
        steps=2,
        batch=8,
        seed=0,
        store=store,
        checkpoint_every=1,
        checkpoint_name="run",
        corpus_name="smoke-train",
    )
    record = store.record("run")
    assert record.settings.seed == 0
    assert record.settings.corpus == "smoke-train"
    assert record.settings.saved_at
    assert record.gradient.steps == 2
    assert record.gradient.batch == 8
    assert record.gradient.learning_rate == pytest.approx(1e-3)
    assert record.gradient.final_loss == pytest.approx(result.loss_curve[-1])
