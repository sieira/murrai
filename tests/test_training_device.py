import torch

from murrai.application.runs import (
    chip_profile,
    suggested_defaults,
    training_device,
)


def _torch_with(monkeypatch, cuda, mps):
    def fake(available):
        return type("_", (), {"is_available": staticmethod(lambda: available)})

    monkeypatch.setattr(torch, "cuda", fake(cuda))
    monkeypatch.setattr(torch.backends, "mps", fake(mps))


def test_without_a_gpu_training_happens_on_cpu(monkeypatch):
    _torch_with(monkeypatch, cuda=False, mps=False)
    assert training_device(prefer_mps=True) == torch.device("cpu")
    assert training_device(prefer_mps=False) == torch.device("cpu")


def test_cuda_wins_wherever_it_exists(monkeypatch):
    _torch_with(monkeypatch, cuda=True, mps=True)
    assert training_device(prefer_mps=True) == torch.device("cuda")
    assert training_device(prefer_mps=False) == torch.device("cuda")


def test_mps_only_trains_the_gpt_but_not_the_mlp(monkeypatch):
    _torch_with(monkeypatch, cuda=False, mps=True)
    assert training_device(prefer_mps=True) == torch.device("mps")
    assert training_device(prefer_mps=False) == torch.device("cpu")


def test_the_choice_can_force_the_cpu(monkeypatch):
    _torch_with(monkeypatch, cuda=True, mps=True)
    assert training_device(choice="cpu") == torch.device("cpu")


def test_prefer_gpu_prefers_the_mps_even_for_the_mlp(monkeypatch):
    _torch_with(monkeypatch, cuda=False, mps=True)
    assert training_device(prefer_mps=False, choice="prefer gpu") == torch.device(
        "mps",
    )


def test_the_chip_reports_cuda_when_it_exists(monkeypatch):
    _torch_with(monkeypatch, cuda=True, mps=True)
    assert chip_profile() == "cuda"


def test_the_chip_reports_mps_on_apple_silicon(monkeypatch):
    _torch_with(monkeypatch, cuda=False, mps=True)
    assert chip_profile() == "mps"


def test_the_chip_reports_cpu_without_a_gpu(monkeypatch):
    _torch_with(monkeypatch, cuda=False, mps=False)
    assert chip_profile() == "cpu"


def test_the_suggestions_scale_with_the_chip():
    assert suggested_defaults("cpu") == {
        "batch": 64,
        "dimension": 64,
        "depth": 4,
        "max_time": 64,
        "hidden": 128,
    }
    assert suggested_defaults("mps") == {
        "batch": 128,
        "dimension": 128,
        "depth": 6,
        "max_time": 128,
        "hidden": 256,
    }
    assert suggested_defaults("cuda") == {
        "batch": 256,
        "dimension": 128,
        "depth": 6,
        "max_time": 128,
        "hidden": 256,
    }


def test_an_unknown_chip_suggests_nothing():
    assert suggested_defaults("exotic") == {}


def test_the_suggestions_survive_being_edited_by_a_caller():
    tuned = suggested_defaults("cpu")
    tuned["batch"] = 99999

    assert suggested_defaults("cpu")["batch"] == 64
