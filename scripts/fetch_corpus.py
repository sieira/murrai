from pathlib import Path

from murrai.adapters.texts import fetch_text, strip_between

START = "START OF THE PROJECT GUTENBERG EBOOK"
END = "END OF THE PROJECT GUTENBERG EBOOK"

BOOKS = {
    1497: "republic",
    4280: "critique-of-pure-reason",
    6762: "politics",
    8438: "nicomachean-ethics",
    9662: "enquiry-human-understanding",
    4583: "dialogues-natural-religion",
    3800: "spinoza-ethics",
    1998: "thus-spake-zarathustra",
    4363: "beyond-good-and-evil",
    2680: "meditations",
    5827: "problems-of-philosophy",
    132: "art-of-war",
    216: "tao-teh-king",
    3207: "leviathan",
    815: "democracy-in-america-1",
    816: "democracy-in-america-2",
    1404: "federalist-papers",
    34901: "on-liberty",
    1232: "the-prince",
    147: "common-sense",
    205: "walden",
    61: "communist-manifesto",
    46333: "social-contract-discourses",
    3300: "wealth-of-nations",
    33310: "ricardo-principles",
    833: "theory-leisure-class",
    15776: "economic-consequences-peace",
    4239: "essay-population",
    1228: "origin-of-species",
    2300: "descent-of-man",
    621: "varieties-religious-experience",
    28233: "principia",
    30155: "relativity",
    15489: "dream-psychology",
    10615: "locke-essay-1",
    10616: "locke-essay-2",
    21: "aesop",
}


def main():
    total = 0
    for book_id, slug in BOOKS.items():
        out = Path("data/corpus") / f"{slug}.txt"
        if out.exists():
            count = len(out.read_text())
            print(f"{slug}: cached, {count} chars")
        else:
            raw = fetch_text(
                f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            )
            body = strip_between(raw, START, END)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(body)
            count = len(body)
            print(f"{slug}: fetched, {count} chars")
        total += count
    print(f"total: {total} chars")


if __name__ == "__main__":
    main()
