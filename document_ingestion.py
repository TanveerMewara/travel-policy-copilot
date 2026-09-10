"""Validate and store an uploaded policy PDF with retrieval metadata."""

import json
import re
from io import BytesIO

from search import POLICY_DIR

MAX_PDF_BYTES = 10 * 1024 * 1024


def validate_document_id(document_id):
    value = document_id.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,79}", value):
        raise ValueError("Document ID must be 3-80 lowercase letters, numbers, or hyphens.")
    return value


def ingest_pdf_bytes(content, metadata):
    """Save a validated PDF and a JSON sidecar without overwriting a document."""
    document_id = validate_document_id(metadata.get("document_id", ""))
    if not content.startswith(b"%PDF-"):
        raise ValueError("The uploaded file does not have a valid PDF header.")
    if len(content) > MAX_PDF_BYTES:
        raise ValueError("PDF files must be 10 MB or smaller in this demo.")

    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        has_text = any((page.extract_text() or "").strip() for page in reader.pages)
    except Exception as error:
        raise ValueError(f"The PDF could not be read: {error}") from error
    if not has_text:
        raise ValueError(
            "No extractable text was found. Scanned PDFs require OCR, which this demo "
            "does not implement yet."
        )

    pdf_path = POLICY_DIR / f"{document_id}.pdf"
    metadata_path = POLICY_DIR / f"{document_id}.metadata.json"
    if pdf_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Document ID '{document_id}' already exists. Choose another ID.")

    clean_metadata = {
        "document_id": document_id,
        "title": str(metadata.get("title", "")).strip() or document_id,
        "supplier": str(metadata.get("supplier", "")).strip() or "Unknown",
        "rate": str(metadata.get("rate", "")).strip() or "Unknown",
        "effective_from": str(metadata.get("effective_from", "")),
        "effective_to": str(metadata.get("effective_to", "")),
    }
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(content)
    try:
        metadata_path.write_text(json.dumps(clean_metadata, indent=2), encoding="utf-8")
    except Exception:
        pdf_path.unlink(missing_ok=True)
        raise
    return pdf_path, metadata_path
