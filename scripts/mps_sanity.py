import time

import torch

N = 2048


def go(device):
    torch.manual_seed(0)
    x = torch.randn(N, N, device=device)
    y = torch.randn(N, N, device=device)
    for _ in range(3):
        torch.mm(x, y)
    if device.type == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        z = torch.mm(x, y)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    print(
        f"mm {device.type}: 10 x {N}x{N} matmuls in {elapsed:.3f}s "
        f"({2 * N**3 * 10 / elapsed / 1e9:.1f} GFLOP/s), sum={float(z.sum()):.2f}"
    )


def main():
    assert torch.mps.is_available(), "MPS must be available on the target machine"
    go(torch.device("cpu"))
    go(torch.device("mps"))


if __name__ == "__main__":
    main()
