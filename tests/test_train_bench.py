import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "train_bench.py"
spec = importlib.util.spec_from_file_location("train_bench", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_the_correctness_check_runs_end_to_end(capsys):
    bench.correctness("cpu", "ab" * 2000)

    printed = capsys.readouterr().out
    assert "held-out perplexity" in printed
    assert "checkpoint devices ['cpu']" in printed
