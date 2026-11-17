from pathlib import Path

from murrai.adapters.tinystories import TinyStories


def main():
    root = Path("data/corpus")
    TinyStories(root).fetch()
    for name in ("tinystories-train", "tinystories-valid"):
        file = root / name / "tinystories.txt"
        print(f"{file}: {file.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
