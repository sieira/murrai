import io
import os
from pathlib import Path

import pytest

from murrai.adapters.tinystories import (
    SMOKE_CHARS,
    TRAIN_URL,
    VALID_URL,
    TinyStories,
)


def _source(payloads):
    calls = []

    def open_url(url):
        calls.append(url)
        return io.BytesIO(payloads[url])

    open_url.calls = calls
    return open_url


def _fetched(tmp_path, payloads):
    source = _source(payloads)
    adapter = TinyStories(tmp_path, source=source)
    adapter.fetch()
    return adapter, source


def test_fetch_downloads_both_splits_into_the_corpus_root(tmp_path):
    _adapter, source = _fetched(
        tmp_path,
        {TRAIN_URL: b"train story", VALID_URL: b"valid story"},
    )

    assert (tmp_path / "tinystories-train" / "tinystories.txt").read_bytes() == (
        b"train story"
    )
    assert (tmp_path / "tinystories-valid" / "tinystories.txt").read_bytes() == (
        b"valid story"
    )
    assert sorted(source.calls) == [TRAIN_URL, VALID_URL]


def test_the_fetched_splits_come_back_through_the_corpus_port(tmp_path):
    adapter, _ = _fetched(
        tmp_path,
        {TRAIN_URL: b"train story", VALID_URL: b"valid story"},
    )

    assert adapter.train().text() == "train story"
    assert adapter.valid().text() == "valid story"


def test_a_second_fetch_touches_no_network(tmp_path):
    adapter, source = _fetched(
        tmp_path,
        {TRAIN_URL: b"train story", VALID_URL: b"valid story"},
    )

    adapter.fetch()

    assert sorted(source.calls) == [TRAIN_URL, VALID_URL]


def test_fetch_rebuilds_the_smoke_slice_from_the_head_of_the_valid_split(tmp_path):
    valid = b"x" * SMOKE_CHARS + b"tail that stays out of the slice"
    _fetched(tmp_path, {TRAIN_URL: b"train story", VALID_URL: valid})

    slice_text = (tmp_path / "tinystories.txt").read_text(encoding="utf-8")

    assert slice_text == "x" * SMOKE_CHARS + "\n"


def test_fetch_keeps_an_existing_smoke_slice(tmp_path):
    (tmp_path / "tinystories.txt").write_text("existing fixture", encoding="utf-8")
    _fetched(tmp_path, {TRAIN_URL: b"train story", VALID_URL: b"valid story"})

    assert (tmp_path / "tinystories.txt").read_text(encoding="utf-8") == (
        "existing fixture"
    )


def test_a_download_that_fails_midway_leaves_no_split_behind(tmp_path):
    def failing_source(url):
        class Response:
            def read(self, size):
                raise OSError("connection reset")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return Response()

    adapter = TinyStories(tmp_path, source=failing_source)
    try:
        adapter.fetch()
    except OSError:
        pass
    else:
        raise AssertionError("the fetch should propagate the failure")

    assert not (tmp_path / "tinystories-train" / "tinystories.txt").exists()


@pytest.mark.skipif(
    not os.environ.get("MURRAI_REAL_DOWNLOAD"),
    reason="downloads 2GB from Hugging Face; opt in with MURRAI_REAL_DOWNLOAD=1",
)
def test_the_real_download_fills_both_splits():
    root = Path("data/corpus")
    adapter = TinyStories(root)
    adapter.fetch()

    train_file = root / "tinystories-train" / "tinystories.txt"
    assert train_file.stat().st_size > 1_000_000_000
    assert len(adapter.valid().text()) > 1_000_000
