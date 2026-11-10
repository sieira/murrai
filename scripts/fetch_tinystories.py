import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStories-valid.txt"
)


def main():
    raw = urllib.request.urlopen(URL).read().decode("utf-8")
    sliced = raw[:2_000_000].strip() + "\n"
    out = Path("data/corpus/tinystories.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sliced)
    print(f"wrote {out} ({len(sliced)} chars)")


if __name__ == "__main__":
    main()
