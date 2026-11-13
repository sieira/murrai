import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_demo_module():
    path = Path(__file__).parent.parent / "scripts" / "train_heads_demo.py"
    spec = importlib.util.spec_from_file_location("train_heads_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def generate(self, n_chars, seed):
        return "x"


def test_the_demo_trains_on_the_requested_device(tmp_path, monkeypatch):
    (tmp_path / "story.txt").write_text("aabbaabb" * 20)
    module = _load_demo_module()
    captured = {}
    monkeypatch.setattr(
        module,
        "train_transformer_lm",
        lambda text, **kwargs: (
            captured.update(kwargs),
            SimpleNamespace(
                model=FakeModel(),
                loss_curve=[1.0, 0.5],
                held_out_perplexity=1.5,
            ),
        )[1],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_heads_demo.py", str(tmp_path), "--steps", "1", "--device", "cpu"],
    )

    module.main()

    assert captured["device"] == "cpu"
