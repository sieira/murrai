from datetime import UTC, datetime

from murrai.application.checkpoints import (
    CheckpointRecord,
    GradientRun,
    RunSettings,
)
from murrai.application.runs import train_transformer_lm

MODEL = {"depth": 8, "dimension": 320, "max_time": 128, "heads": 5}


def train_transformer_from_corpus(
    corpus,
    steps,
    held_out=None,
    batch=256,
    seed=0,
    learning_rate=1e-3,
    validate_every=200,
    checkpoint_every=500,
    store=None,
    checkpoint_name="tinystories",
    corpus_name="tinystories-train",
    merges=300,
    device=None,
    bus=None,
):
    settings = RunSettings(
        saved_at=datetime.now(UTC).isoformat(timespec="seconds"),
        corpus=corpus_name,
        seed=seed,
    )
    checkpoint = None
    snapshots = None
    keeper = None
    if store is not None:
        if checkpoint_name in store.names():
            checkpoint = store.load(checkpoint_name)

        def snapshots(snapshot):
            store.save(checkpoint_name, snapshot)

        def keeper(model):
            store.save_record(
                checkpoint_name,
                CheckpointRecord(
                    settings=settings,
                    gradient=GradientRun(
                        steps=steps,
                        batch=batch,
                        learning_rate=learning_rate,
                    ),
                ),
            )

    result = train_transformer_lm(
        corpus.text(),
        steps=steps,
        batch=batch,
        seed=seed,
        holdout=0,
        learning_rate=learning_rate,
        validate_every=validate_every,
        checkpoint_every=checkpoint_every,
        snapshots=snapshots,
        held_out_text=held_out.text() if held_out is not None else None,
        merges=merges,
        device=device,
        bus=bus,
        keeper=keeper,
        checkpoint=checkpoint,
        **MODEL,
    )
    if store is not None:
        store.save_record(
            checkpoint_name,
            CheckpointRecord(
                settings=settings,
                gradient=GradientRun(
                    steps=result.steps,
                    batch=batch,
                    learning_rate=learning_rate,
                    final_loss=result.loss_curve[-1],
                ),
                held_out_perplexity=result.held_out_perplexity,
            ),
        )
    return result
