import torch

from murrai.application.runs import training_device


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
