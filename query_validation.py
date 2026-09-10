"""Checks that must run before trusting retrieval or calling generation."""

import re


def missing_policy_year(question, candidates):
    """Return available years when otherwise-identical policy versions conflict."""
    if re.search(r"\b(?:19|20)\d{2}\b", question):
        return []
    if not candidates:
        return []

    top = candidates[0]
    top_metadata = top["metadata"]
    versions = [
        candidate
        for candidate in candidates
        if candidate["metadata"].get("supplier") == top_metadata.get("supplier")
        and candidate["metadata"].get("rate") == top_metadata.get("rate")
        and candidate["section"] == top["section"]
    ]
    years = sorted(
        {
            candidate["metadata"].get("effective_from", "")[:4]
            for candidate in versions
            if candidate["metadata"].get("effective_from", "")[:4].isdigit()
        }
    )
    return years if len(years) > 1 else []
