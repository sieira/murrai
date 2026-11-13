from murrai.domain.tokenizer import TokenizerEncoderDecoder

CORPUS = """\
The old lighthouse keeper climbed the spiral stairs every evening, carrying a
lantern of whale oil and a small book of weather notes. He had kept the notes
for forty years, and in that time he had learned to read the sea the way a
farmer reads the sky. A green tint at the horizon meant rain before morning.
A sudden stillness in the gulls meant a storm was gathering far offshore. The
village children thought him strange, but the fishermen trusted him more than
they trusted the forecast office in the city, because the lighthouse keeper
had never once been wrong about the wind.

One autumn, a bright orange balloon drifted out of the clouds and came to
rest against the railing of the lantern room. Tied to its string was a folded
letter, written in careful purple ink, asking whoever found it to write back
to a girl in a mountain town who had never seen the ocean. The keeper wrote
back the same night. He described the sound of the waves on the rocks below,
the smell of salt and cold stone, and the way the light swept the water in
long slow strokes, once every eight seconds, all night long.

The letter took three weeks to arrive, and the reply took three weeks to
return, and so a correspondence began that lasted seven years. She sent
pressed alpine flowers and questions about the sea. He sent feathers from
the cliffs and answers that filled entire pages. When he finally climbed
down the stairs for the last time, an old man with tired knees, he mailed
her the book of weather notes, forty years of small observations, and a
single line at the end: the sea is only ever a mirror, and you were the
first person who ever looked into it with me."""

UNSEEN = """\
In the city she kept a small balcony garden and thought about the ocean
every single day, wondering whether the waves sounded the way he had
described them, or whether words had simply made the memory gentler."""

if __name__ == "__main__":
    for num_merges in (100, 300):
        t = TokenizerEncoderDecoder().fit(CORPUS, num_merges=num_merges)
        print(f"merges: {num_merges}  vocab size: {t.vocab_size}")
        for name, text in (("training corpus", CORPUS), ("unseen sample", UNSEEN)):
            char_tokens = len(text)
            bpe_tokens = len(t.encode(text))
            print(
                f"  {name:16s} char-level: {char_tokens:5d}  bpe: {bpe_tokens:5d}"
                f"  density gain: {char_tokens / bpe_tokens:.2f}x",
            )
