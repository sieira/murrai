import json
from dataclasses import FrozenInstanceError

import pytest

from murrai.adapters.story import FileStoryStore, InMemoryStoryStore
from murrai.application.sessions import StorySave

SAVE = StorySave(
    ids=(1, 2, 3),
    origins=("entered", "generated", "entered"),
    checkpoint="gpt-corpus",
)


def test_a_story_save_is_frozen():
    with pytest.raises(FrozenInstanceError):
        SAVE.ids = ()


def test_the_file_story_store_round_trips(tmp_path):
    store = FileStoryStore(tmp_path)
    store.save("gpt-corpus-story", SAVE)

    assert store.names() == ["gpt-corpus-story"]
    assert store.load("gpt-corpus-story") == SAVE


def test_the_file_story_store_lists_names_sorted(tmp_path):
    store = FileStoryStore(tmp_path)
    store.save("zebra", SAVE)
    store.save("antelope", SAVE)

    assert store.names() == ["antelope", "zebra"]


def test_the_file_story_store_rejects_a_missing_name(tmp_path):
    store = FileStoryStore(tmp_path)

    with pytest.raises(KeyError):
        store.load("gpt-corpus-story")


def test_the_file_story_store_translates_the_save_to_flat_json(tmp_path):
    store = FileStoryStore(tmp_path)
    store.save("gpt-corpus-story", SAVE)

    data = json.loads((tmp_path / "gpt-corpus-story.json").read_text())
    assert data == {
        "checkpoint": "gpt-corpus",
        "ids": [1, 2, 3],
        "origins": ["entered", "generated", "entered"],
    }


def test_the_memory_story_store_carries_the_save():
    store = InMemoryStoryStore()
    store.save("gpt-corpus-story", SAVE)

    assert store.load("gpt-corpus-story") == SAVE
    assert store.names() == ["gpt-corpus-story"]


def test_names_skips_an_unfinished_file(tmp_path):
    store = FileStoryStore(tmp_path)
    (tmp_path / "half-written.json.tmp").write_text("junk")

    assert store.names() == []


def test_an_interrupted_save_keeps_the_previous_story(tmp_path, monkeypatch):
    store = FileStoryStore(tmp_path)
    store.save("gpt-corpus-story", SAVE)

    def interrupted_dumps(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("murrai.adapters.story.json.dumps", interrupted_dumps)
    with pytest.raises(KeyboardInterrupt):
        store.save("gpt-corpus-story", SAVE)

    assert store.load("gpt-corpus-story") == SAVE
