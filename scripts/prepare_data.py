#!/usr/bin/env python3
"""Clean raw author texts into a single training-ready corpus.

Usage:
    python scripts/prepare_data.py <author>     # e.g. weil, dickinson, le_guin
    python scripts/prepare_data.py --all

Reads:  authors/<author>/data/raw/*.{txt,muse}
Writes: authors/<author>/data/processed/<author>.txt   (cleaned, concatenated)

The goal is a corpus that is *only the author's own voice* — boilerplate,
editorial apparatus, footnotes, and (for anthologies) sections written by other
people are stripped, because anything left in gets learned as "their voice".
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUTHORS_DIR = ROOT / "authors"

# --- per-file overrides ----------------------------------------------------
# For .muse anthology files, keep ONLY the sections actually written by the
# author. Sections are delimited by "** Heading" lines in Muse format.
# key = filename, value = (keep_from_heading, stop_before_heading) substrings.
MUSE_KEEP_SECTIONS = {
    "abolition_political_parties.muse": (
        "On the Abolition of All Political Parties",  # Weil's essay starts here
        "The Importance of Simone Weil",              # Milosz appendix starts here
    ),
}

# Files whose editorial apparatus is too entangled with the text to separate
# automatically — excluded so the corpus stays author-voice-only.
SKIP_FILES = {
    # Todd's 1894 letters weave biographical narration around the letters;
    # excluded for now (the poems are clean Dickinson on their own).
    "dickinson": {"letters_todd_1894.txt"},
}

# Authors whose raw files are Gutenberg prose novels (cleaned by strip_gutenberg_novel).
NOVEL_AUTHORS = {"hugo", "dostoevsky", "tolstoy"}


# anchor: a structural heading at column 0 (tables of contents indent their entries,
# so requiring no leading whitespace skips the TOC)
_HEADING = re.compile(r"(?i)^(VOLUME|BOOK|CHAPTER|PART|PROLOGUE|EPILOGUE)\b")
# title-case structural headings like "Chapter I. ..." / "Book One: ..." — required
# numeral after the word so prose ("Part of him...") is never matched
_NUM_HEADING = re.compile(
    r"(?im)^\s*(VOLUME|BOOK|CHAPTER|PART|PROLOGUE|EPILOGUE)\s+"
    r"([IVXLC]+|\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|"
    r"FIRST|SECOND|THIRD|FOURTH|FIFTH)\b.*$")


def strip_gutenberg_novel(text: str) -> str:
    """Clean a Gutenberg prose novel: skip front matter / translator's intro to the
    first real chapter, then drop structural headings and illustration markers.

    The novel's start is the first VOLUME/BOOK/CHAPTER/PART heading that is followed
    closely by real prose — this skips both the table of contents (headings followed
    by more headings) and any translator's preface (prose with no heading)."""
    text = re.sub(r"\[[^\]]*\]", "", text)                          # [Illustration: ...]
    # Gutenberg endnote back-matter: "<n> (return) <gloss>" blocks (the "(return)"
    # is the footnote's link-back text). Drop each from its marker to the next
    # blank line so multi-line glosses go too — all translator apparatus, not voice.
    text = re.sub(r"(?ms)^\s*\d+ \(return\).*?(?=\n\s*\n|\Z)", "", text)
    # novel start = first structural heading followed (within ~30 lines) by a real
    # PROSE PARAGRAPH (>=50 words in one block). A table of contents is many short
    # lines (no single long paragraph), so TOC headings don't trigger it; a
    # translator's preface has no structural heading, so it's skipped too.
    lines = text.split("\n")

    def find_start(strip_indent: bool) -> int | None:
        for i, ln in enumerate(lines):
            if _HEADING.match(ln.lstrip() if strip_indent else ln):
                chunk = "\n".join(lines[i + 1:i + 30])
                if any(len(p.split()) >= 50 for p in re.split(r"\n\s*\n", chunk)):
                    return i
        return None

    # Prefer column-0 headings: real chapter heads sit at col 0 while the table of
    # contents is indented, so strict matching skips the TOC. Fall back to
    # indent-tolerant matching only if that finds nothing — some editions (e.g.
    # Notre-Dame de Paris) indent the ENTIRE body, so no heading is ever at col 0.
    start = find_start(strip_indent=False)
    if start is None:
        start = find_start(strip_indent=True)
    if start is not None:
        lines = lines[start:]
    text = "\n".join(lines)
    # strip structural headings (UPPERCASE incl. em-dashes) + title-case numbered ones
    text = re.sub(r"(?m)^\s*(VOLUME|BOOK|CHAPTER|PART|PROLOGUE|EPILOGUE)\b.*$", "", text)
    text = _NUM_HEADING.sub("", text)
    # standalone ALL-CAPS titles / list-of-illustrations captions
    text = re.sub(r"(?m)^\s*[A-ZÀ-Þ][A-ZÀ-Þ0-9'’.,;:!?()\-—– ]{3,}\s*$", "", text)
    return text


def strip_dickinson_poems(text: str) -> str:
    """Keep only Dickinson's poems from the 3-series Gutenberg edition: drop the
    title page + each series' editorial PREFACE, and bracketed publication notes."""
    # each of the 3 series opens with a title page ("POEMS / by EMILY DICKINSON /
    # <Series> / Edited by ... / PREFACE / <preface>") that ends at the first poem
    # section "I. LIFE." — drop that whole front-matter block for every series.
    text = re.sub(r"(?ms)^POEMS\s*$.*?(?=^I\.\s*LIFE\.)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)                       # [Published in ...] notes
    # drop editor scaffolding: Roman-numeral poem indices and ALL-CAPS titles
    # (Dickinson's poems were untitled; titles/sections were added by editors)
    text = re.sub(r"(?m)^\s*[IVXLC]+\.\s*$", "", text)
    text = re.sub(r"(?m)^\s*[A-Z][A-Z'’ .,;:!?-]{2,}\s*$", "", text)
    return text


# --- cleaning primitives ---------------------------------------------------
def strip_gutenberg(text: str) -> str:
    """Drop Project Gutenberg license header/footer if present."""
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I)
    if start:
        text = text[start.end():]
    # search END *after* trimming the front — end.start() must index the current
    # string, not the original, or the cut lands start.end() chars too late and
    # leaks the END marker + trailing license boilerplate.
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I)
    if end:
        text = text[:end.start()]
    return text


def strip_muse(text: str, fname: str) -> str:
    """Strip Muse markup, metadata header, footnotes, and non-author sections."""
    # 1) drop the #title/#author/#date... metadata header block
    lines = text.splitlines()
    lines = [ln for ln in lines if not ln.lstrip().startswith("#")]
    text = "\n".join(lines)

    # 2) keep only the author's own section(s) if configured
    if fname in MUSE_KEEP_SECTIONS:
        keep_from, stop_before = MUSE_KEEP_SECTIONS[fname]
        s = text.find("** " + keep_from)
        e = text.find("** " + stop_before)
        if s != -1:
            text = text[s:e] if e != -1 else text[s:]

    # 3) drop the section heading lines themselves and footnote definitions
    out = []
    for ln in text.splitlines():
        if ln.startswith("** ") or ln.startswith("* "):
            continue                       # section headings
        if re.match(r"^\[\d+\]", ln.strip()):
            continue                       # footnote definitions: "[1] ibid."
        out.append(ln)
    text = "\n".join(out)

    # 4) inline markup: <em>..</em>, <br>, <sup>..</sup>, [[link][label]], [3]
    text = re.sub(r"\[\[[^\]]*\]\[([^\]]*)\]\]", r"\1", text)  # wiki link -> label
    text = re.sub(r"<[^>]+>", "", text)                        # html-ish tags
    text = re.sub(r"\[\d+\]", "", text)                        # inline footnote refs
    text = text.replace("/", "")                               # stray muse emphasis
    return text


def normalize(text: str) -> str:
    """Whitespace, hyphenation, OCR punctuation-spacing, and quote cleanup."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")            # soft hyphen
    text = text.replace("﻿", "")            # BOM

    # join words broken across a line by a hyphen:  "signal-\nized" -> "signalized"
    text = re.sub(r"(\w)[-‐]\s*\n\s*(\w)", r"\1\2", text)

    # OCR artifact: spaces *around* punctuation -> "word , then" -> "word, then"
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=\w)", r"\1 ", text)   # ensure a space after

    # spaced/curly quotes:  “ word ”  ->  “word”
    text = re.sub(r"([“‘])\s+", r"\1", text)
    text = re.sub(r"\s+([”’])", r"\1", text)

    # rejoin contractions/possessives split by an apostrophe: "Isn' t" -> "Isn't"
    text = re.sub(r"\s*([’'])\s*(t|s|ll|re|ve|d|m)\b", r"\1\2", text, flags=re.I)

    # collapse runs of spaces/tabs and excess blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_file(path: Path, author: str = "") -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".muse":
        raw = strip_muse(raw, path.name)
    else:
        raw = strip_gutenberg(raw)
        if path.name == "gutenberg_poems_complete.txt":
            raw = strip_dickinson_poems(raw)
        elif author in NOVEL_AUTHORS:
            raw = strip_gutenberg_novel(raw)
    return normalize(raw)


# --- driver ----------------------------------------------------------------
def process_author(author: str) -> None:
    raw_dir = AUTHORS_DIR / author / "data" / "raw"
    out_dir = AUTHORS_DIR / author / "data" / "processed"
    skip = SKIP_FILES.get(author, set())
    files = sorted(p for p in raw_dir.iterdir()
                   if p.is_file() and p.suffix in {".txt", ".muse"} and p.name not in skip)
    if not files:
        print(f"  [{author}] no raw files in {raw_dir} — skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    parts, rows = [], []
    for f in files:
        cleaned = clean_file(f, author)
        parts.append(cleaned)
        rows.append((f.name, len(f.read_text(encoding="utf-8", errors="replace").split()),
                     len(cleaned.split())))

    corpus = "\n\n".join(parts) + "\n"
    out_path = out_dir / f"{author}.txt"
    out_path.write_text(corpus, encoding="utf-8")

    print(f"\n[{author}] -> {out_path.relative_to(ROOT)}")
    print(f"  {'file':<40} {'raw words':>10} {'clean words':>12}")
    for name, rw, cw in rows:
        print(f"  {name:<40} {rw:>10,} {cw:>12,}")
    print(f"  {'TOTAL':<40} {sum(r[1] for r in rows):>10,} {len(corpus.split()):>12,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("author", nargs="?", help="author dir name (e.g. weil)")
    ap.add_argument("--all", action="store_true", help="process every author")
    args = ap.parse_args()

    if args.all:
        authors = sorted(p.name for p in AUTHORS_DIR.iterdir()
                         if (p / "data" / "raw").is_dir())
    elif args.author:
        authors = [args.author]
    else:
        ap.error("give an author name or --all")

    for a in authors:
        process_author(a)


if __name__ == "__main__":
    main()
