import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/21/pg21.txt"


def main():
    text = urllib.request.urlopen(URL).read().decode("utf-8-sig")
    start = text.find("LIFE OF AESOP")
    end = text.find("*** END")
    sliced = text[start:end].strip()[:60000] + "\n"
    out = Path("data/corpus/aesop.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sliced)
    print(f"wrote {out} ({len(sliced)} chars)")


if __name__ == "__main__":
    main()
