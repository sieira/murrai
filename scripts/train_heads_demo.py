import argparse
import json
import time
from pathlib import Path

import torch

from murrai.adapters.checkpoint import FileCheckpointStore
from murrai.adapters.corpus import DirTextCorpus
from murrai.adapters.events import ListenerBus
from murrai.application.checkpoints import snapshot_of
from murrai.application.runs import train_transformer_lm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--chars", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save")
    parser.add_argument("--store", default="checkpoints")
    args = parser.parse_args()

    torch.set_default_device(args.device)
    text = DirTextCorpus(args.corpus).text()[: args.chars]
    started = time.time()

    def show(event):
        if event.step % 100 == 0:
            print(
                f"step {event.step}/{event.total_steps} {time.time() - started:.0f}s",
                flush=True,
            )

    bus = ListenerBus()
    bus.subscribe(show)
    result = train_transformer_lm(
        text,
        steps=args.steps,
        batch=args.batch,
        seed=0,
        depth=args.depth,
        dimension=args.dimension,
        max_time=args.window,
        heads=args.heads,
        bus=bus,
    )
    curve = [
        sum(chunk) / len(chunk)
        for chunk in (
            result.loss_curve[i : i + 100]
            for i in range(0, len(result.loss_curve), 100)
        )
    ]

    torch.set_default_device("cpu")
    model = result.model.to("cpu")
    parameters = sum(p.numel() for p in model.parameters())
    print(f"heads {args.heads}, {parameters} parameters")
    print(f"held-out perplexity {result.held_out_perplexity:.3f}")
    print(f"train loss {result.loss_curve[0]:.3f} -> {result.loss_curve[-1]:.3f}")
    print("loss per 100 steps", " ".join(f"{value:.3f}" for value in curve))
    print("--- sample (seed 17) ---")
    print(model.generate(200, seed=17))

    if args.save:
        FileCheckpointStore(args.store).save(args.save, snapshot_of(model))
        notes = {
            "heads": args.heads,
            "steps": args.steps,
            "parameters": parameters,
            "held_out_perplexity": result.held_out_perplexity,
            "loss_per_100_steps": curve,
        }
        Path(args.store, f"{args.save}.json").write_text(json.dumps(notes, indent=2))
        print(f"saved {args.store}/{args.save}.pt")


if __name__ == "__main__":
    main()
