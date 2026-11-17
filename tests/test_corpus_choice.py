from types import SimpleNamespace

import murrai.composition.chat as murrai_composition_chat
from murrai.adapters.checkpoint import InMemoryCheckpointStore
from murrai.composition.chat import _sliced_text, chat_menu
from murrai.domain.tokenizer import TokenizerEncoderDecoder
from murrai.presentation.chat.form import TrainingForm


def _corpus_root(tmp_path, themes):
    data = tmp_path / "data"
    corpus = data / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "base.txt").write_text("ab" * 40)
    for name, text in themes.items():
        theme = data / name
        theme.mkdir()
        (theme / f"{name}.txt").write_text(text)
    return corpus


def _command(directory, name, **kwargs):
    tree = chat_menu(directory, **kwargs)()
    return next(c for group in tree.values() for c in group if c.name == name)


def _field(directory, command_name, field_name, **kwargs):
    command = _command(directory, command_name, **kwargs)
    return next(f for f in command.fields if f.name == field_name)


def _values(command, **overrides):
    values = {field.name: field.default for field in command.fields}
    values.update(overrides)
    return values


def test_the_text_field_lists_the_directories_with_texts(tmp_path):
    corpus = _corpus_root(tmp_path, {"philosophy": "cddc" * 20, "science": "effe" * 20})
    (tmp_path / "data" / "empty").mkdir()
    (tmp_path / "data" / "notes.txt").write_text("not a directory")

    field = _field(corpus, "counted bigram", "text", seed=0)

    assert field.kind == "choice"
    assert field.label == "text"
    assert field.choices == ("corpus", "philosophy", "science")
    assert field.default == "corpus"


def test_the_text_field_survives_a_corpus_without_neighbors(tmp_path):
    corpus = _corpus_root(tmp_path, {})

    field = _field(corpus, "counted bigram", "text", seed=0)

    assert field.choices == ("corpus",)


def test_a_newly_fetched_theme_shows_up_without_a_restart(tmp_path):
    corpus = _corpus_root(tmp_path, {})

    before = _field(corpus, "counted bigram", "text", seed=0).choices
    (tmp_path / "data" / "politics").mkdir()
    (tmp_path / "data" / "politics" / "prince.txt").write_text("cddc" * 20)
    after = _field(corpus, "counted bigram", "text", seed=0).choices

    assert before == ("corpus",)
    assert after == ("corpus", "politics")


def test_the_chosen_text_lands_in_the_run(tmp_path):
    corpus = _corpus_root(tmp_path, {"philosophy": "cddc" * 20})
    store = InMemoryCheckpointStore()
    counted = _command(corpus, "counted bigram", seed=0, store=store)

    counted.bake(
        _values(counted, text="philosophy", name="counted-bigram-philosophy"),
    ).build()

    record = store.record("counted-bigram-philosophy")
    assert record.settings.corpus == str(tmp_path / "data" / "philosophy")


def test_the_form_hands_the_chosen_text_to_the_bake(tmp_path):
    corpus = _corpus_root(tmp_path, {"philosophy": "cddc" * 20})
    command = _command(corpus, "counted bigram", seed=0)
    form = TrainingForm(command)
    form.open()
    form.move_choice("text", 1)

    values = form.accept()

    assert values["text"] == "philosophy"


def test_the_chosen_text_trains_on_that_directory(tmp_path):
    corpus = _corpus_root(tmp_path, {"philosophy": "cddc" * 20})
    store = InMemoryCheckpointStore()
    counted = _command(corpus, "counted bigram", seed=0, store=store)

    counted.bake(
        _values(counted, text="philosophy", name="counted-bigram-philosophy"),
    ).build()

    record = store.record("counted-bigram-philosophy")
    assert record.settings.corpus == str(tmp_path / "data" / "philosophy")


def test_the_default_text_keeps_the_configured_corpus(tmp_path):
    corpus = _corpus_root(tmp_path, {"philosophy": "cddc" * 20})
    store = InMemoryCheckpointStore()
    counted = _command(corpus, "counted bigram", seed=0, store=store)

    counted.bake(_values(counted)).build()

    assert store.record("counted-bigram-corpus").settings.corpus == str(corpus)


def test_the_size_field_offers_the_caps(tmp_path):
    corpus = _corpus_root(tmp_path, {})
    commands = (
        ("counted bigram", {}),
        ("neural bigram", {"neural_steps": 20}),
        ("windowed mlp", {"mlp_steps": 20}),
        ("gpt", {"gpt_steps": 20}),
    )

    for name, kwargs in commands:
        field = _field(corpus, name, "size", seed=0, **kwargs)
        assert field.kind == "choice", name
        assert field.label == "size", name
        assert field.choices == ("all", "100k", "500k"), name
        assert field.default == "all", name


def test_the_chosen_size_lands_in_the_baked_values(tmp_path):
    corpus = _corpus_root(tmp_path, {})
    counted = _command(corpus, "counted bigram", seed=0)
    form = TrainingForm(counted)
    form.open()
    form.move_choice("size", 1)

    values = form.accept()

    assert values["size"] == "100k"


def _single_text_corpus(tmp_path, text):
    corpus = tmp_path / "data" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "base.txt").write_text(text)
    return corpus


def test_the_100k_size_slices_the_text_to_its_first_100000_chars(tmp_path):
    corpus = _single_text_corpus(tmp_path, "ab" * 55000 + "z" * 10000)

    sliced = _sliced_text(corpus, {"size": "100k"})

    assert sliced == "ab" * 50000


def test_the_500k_size_slices_the_text_to_its_first_500000_chars(tmp_path):
    corpus = _single_text_corpus(tmp_path, "ab" * 300000)

    sliced = _sliced_text(corpus, {"size": "500k"})

    assert sliced == "ab" * 250000


def test_the_all_size_keeps_the_whole_text(tmp_path):
    text = "ab" * 55000 + "z" * 10000
    corpus = _single_text_corpus(tmp_path, text)

    assert _sliced_text(corpus, {"size": "all"}) == text


def test_a_capped_counted_run_only_sees_the_capped_chars(tmp_path):
    corpus = _single_text_corpus(tmp_path, "ab" * 55000 + "z" * 10000)
    store = InMemoryCheckpointStore()
    counted = _command(corpus, "counted bigram", seed=0, store=store)

    counted.bake(_values(counted, size="100k")).build()

    record = store.record("counted-bigram-corpus")
    snapshot = store.load("counted-bigram-corpus")
    pieces = TokenizerEncoderDecoder(snapshot.merges).pieces
    assert all("z" not in piece for piece in pieces[256:])
    assert record.settings.corpus_chars == 100000


def _captured_text(monkeypatch):
    captured = {}

    def stub(*args, **kwargs):
        captured["value"] = args[0]
        return SimpleNamespace(model=SimpleNamespace(kind="stub"), loss_curve=[1.0])

    for name in (
        "train_bigram",
        "train_neural_bigram",
        "train_mlp_lm",
        "train_transformer_lm",
    ):
        monkeypatch.setattr(murrai_composition_chat, name, stub)
    return captured


def test_all_four_runners_train_on_the_sliced_text(tmp_path, monkeypatch):
    full = "ab" * 55000 + "z" * 10000
    corpus = _single_text_corpus(tmp_path, full)
    captured = _captured_text(monkeypatch)

    for name in (
        "counted bigram",
        "neural bigram",
        "windowed mlp",
        "gpt",
    ):
        captured.clear()
        command = _command(
            corpus,
            name,
            seed=0,
            neural_steps=20,
            mlp_steps=20,
            gpt_steps=20,
        )
        command.bake(_values(command, size="100k")).build()
        handed = captured["value"]
        if callable(getattr(handed, "text", None)):
            handed = handed.text()
        assert handed == full[:100000], name


def test_a_capped_counted_run_records_the_char_count_it_trained_on(tmp_path):
    corpus = _single_text_corpus(tmp_path, "ab" * 55000 + "z" * 10000)
    store = InMemoryCheckpointStore()
    counted = _command(corpus, "counted bigram", seed=0, store=store)

    counted.bake(_values(counted, size="100k")).build()

    assert store.record("counted-bigram-corpus").settings.corpus_chars == 100000


def test_a_capped_neural_run_records_the_char_count_it_trained_on(tmp_path):
    corpus = _single_text_corpus(tmp_path, "ab" * 55000 + "z" * 10000)
    store = InMemoryCheckpointStore()
    neural = _command(
        corpus,
        "neural bigram",
        seed=0,
        neural_steps=20,
        store=store,
    )

    neural.bake(_values(neural, size="100k")).build()

    assert store.record("neural-bigram-corpus").settings.corpus_chars == 100000


def test_an_uncapped_run_leaves_the_char_count_out_of_the_story(tmp_path):
    corpus = _single_text_corpus(tmp_path, "ab" * 55000 + "z" * 10000)
    store = InMemoryCheckpointStore()
    counted = _command(corpus, "counted bigram", seed=0, store=store)

    counted.bake(_values(counted)).build()

    assert store.record("counted-bigram-corpus").settings.corpus_chars is None


def test_the_form_accepts_a_chosen_text_and_size(tmp_path):
    corpus = _corpus_root(tmp_path, {"philosophy": "cddc" * 20})
    command = _command(corpus, "counted bigram", seed=0)
    form = TrainingForm(command)
    form.open()
    form.move_choice("text", 1)
    form.move(1)
    form.move_choice("size", 1)

    values = form.accept()

    assert values["text"] == "philosophy"
    assert values["size"] == "100k"


def test_a_missing_corpus_directory_still_builds_the_text_field(tmp_path):
    lonely = tmp_path / "data" / "corpus"

    field = _text_field_of(lonely)

    assert field.choices == ("corpus",)


def _text_field_of(directory):
    counted = next(
        c for c in chat_menu(directory)()["new"] if c.name == "counted bigram"
    )
    return next(f for f in counted.fields if f.name == "text")


def test_a_checkpoint_s_name_carries_its_text(tmp_path):
    corpus = _corpus_root(tmp_path, {"philosophy": "cddc" * 20})
    store = InMemoryCheckpointStore()
    counted = _command(corpus, "counted bigram", seed=0, store=store)

    counted.bake(_values(counted, text="philosophy", name="a philosophy run")).build()

    assert store.names() == ["a philosophy run"]
