"""Lesson 1: retrieve fictional policy passages using keyword overlap.

Python standard library only. This is a retrieval baseline, not full RAG.
"""

import argparse
import json
import re
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent / "data" / "policies"
STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "at",
    "to",
    "for",
    "of",
    "and",
    "in",
    "i",
    "my",
    "what",
    "if",
    "can",
    "does",
    "do",
    "it",
}


def tokenize(text):
    """Lowercase text and keep meaningful words for exact word matching."""
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_WORDS


def parse_document(document, path):
    """Read simple front-matter metadata and return it with Markdown content."""
    metadata = {}
    content = document
    if document.startswith("---\n"):
        try:
            raw_metadata, content = document[4:].split("\n---\n", 1)
        except ValueError as error:
            raise ValueError(f"Unclosed metadata block in {path}") from error
        for line in raw_metadata.splitlines():
            if ":" not in line:
                raise ValueError(f"Invalid metadata line in {path}: {line}")
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata, content


def section_id(value):
    """Create a readable identifier fragment from a section heading."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def chunk_page_text(text, max_words=180, overlap_words=30):
    """Split extracted page text into overlapping, fixed-size word windows."""
    if max_words <= overlap_words:
        raise ValueError("max_words must be greater than overlap_words")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


def load_markdown_chunks(path):
    document = path.read_text(encoding="utf-8")
    metadata, content = parse_document(document, path)
    title = content.splitlines()[0].removeprefix("# ")
    document_id = metadata.get("document_id", path.stem)
    chunks = []
    for section in content.split("\n## ")[1:]:
        heading, body = section.split("\n", 1)
        chunks.append(
            {
                "chunk_id": f"{document_id}::{section_id(heading)}",
                "title": title,
                "section": heading,
                "text": body.strip(),
                "source": path.relative_to(POLICY_DIR.parent.parent).as_posix(),
                "metadata": metadata,
            }
        )
    return chunks


def load_pdf_chunks(path):
    """Extract page-aware chunks from a PDF and its optional metadata sidecar."""
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise SystemExit(
            "PDF support requires pypdf. Run:\n"
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from error

    sidecar = path.with_suffix(".metadata.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    metadata.setdefault("document_id", path.stem)
    metadata.setdefault("supplier", "Unknown")
    metadata.setdefault("rate", "Unknown")

    reader = PdfReader(path)
    pdf_title = reader.metadata.title if reader.metadata else None
    title = metadata.get("title") or pdf_title or path.stem.replace("_", " ").title()
    chunks = []
    for page_number, page in enumerate(reader.pages, 1):
        extracted = page.extract_text() or ""
        for chunk_number, text in enumerate(chunk_page_text(extracted), 1):
            chunk_metadata = {**metadata, "page_number": page_number}
            chunks.append(
                {
                    "chunk_id": (
                        f"{metadata['document_id']}::page-{page_number}::chunk-{chunk_number}"
                    ),
                    "title": title,
                    "section": f"Page {page_number}",
                    "text": text,
                    "source": path.relative_to(POLICY_DIR.parent.parent).as_posix(),
                    "metadata": chunk_metadata,
                }
            )
    return chunks


def load_chunks():
    """Load section-based Markdown chunks and page-aware PDF chunks."""
    chunks = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        chunks.extend(load_markdown_chunks(path))
    for path in sorted(POLICY_DIR.glob("*.pdf")):
        chunks.extend(load_pdf_chunks(path))
    return chunks


def retrieve(question, chunks, top_k=3):
    """Rank passages by the number of distinct query words they contain."""
    query_words = tokenize(question)
    results = []
    for chunk in chunks:
        metadata_text = " ".join(str(value) for value in chunk["metadata"].values())
        searchable_text = f"{metadata_text} {chunk['title']} {chunk['section']} {chunk['text']}"
        matched_words = query_words & tokenize(searchable_text)
        if matched_words:
            results.append(
                {**chunk, "score": len(matched_words), "matched_words": sorted(matched_words)}
            )
    return sorted(results, key=lambda result: result["score"], reverse=True)[:top_k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question to search for")
    args = parser.parse_args()
    question = args.question if args.question is not None else input("Your question: ")
    chunks = load_chunks()
    print(f"\nLoaded {len(chunks)} passages. All policies are fictional demo data.")
    results = retrieve(question, chunks)
    if not results:
        print("No keyword matches found. This does not prove the answer is absent.")
        return
    for position, result in enumerate(results, 1):
        print(f"\n{position}. {result['title']} / {result['section']}")
        print(f"   Keyword score: {result['score']} (not a confidence percentage)")
        print(f"   Matched words: {', '.join(result['matched_words'])}")
        print(f"   Source: {result['source']}")
        print(f"   Passage: {result['text']}")
    print("\nThese are search results, not an AI-generated answer.")


if __name__ == "__main__":
    main()
