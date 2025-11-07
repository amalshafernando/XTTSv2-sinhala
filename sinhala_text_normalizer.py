"""Utility helpers for normalizing Sinhala text prior to XTTS training.

These helpers focus on:

* Unicode canonicalization (NFC) to keep composed Sinhala glyphs stable.
* Removing zero-width joiners/non-joiners that frequently appear in scraped corpora.
* Collapsing repeated whitespace and standardized punctuation spacing.
* Converting ASCII digits and common punctuation variants to Sinhala-friendly forms.

The goal is to ensure the transcripts consumed by XTTS fine-tuning are
clean, deduplicated, and free of spurious artifacts that hurt tokenizer
training.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

# Zero-width characters frequently present in Sinhala corpora
ZERO_WIDTH_CHARS = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE (BOM)
}

# Common punctuation variants to normalize. Sinhala uses both Sinhala and
# Western punctuation. We keep a conservative mapping and collapse repeated
# punctuation later.
PUNCTUATION_REPLACEMENTS = {
    "“": '"',
    "”": '"',
    "’": "'",
    "‘": "'",
    "–": "-",
    "—": "-",
    "…": "...",
}

# Map ASCII digits to Sinhala digits. XTTS can cope with ASCII digits as well,
# but Sinhala text sources frequently mix both styles; normalizing improves
# consistency of the tokenizer.
ASCII_TO_SINHALA_DIGITS = str.maketrans("0123456789", "෦෧෨෩෪෫෬෭෮෯")

# Sinhala spacing-sensitive punctuation (keep these characters when cleaning)
ALLOWED_PUNCTUATION = set(".,;:!?" "'\"" "()[]{}" "-" "…")


def remove_zero_width(text: str) -> str:
    """Strip zero-width characters that often pollute Sinhala corpora."""

    return "".join(ch for ch in text if ch not in ZERO_WIDTH_CHARS)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and trim the string."""

    # Replace any carriage returns or tabs with spaces before collapsing
    text = text.replace("\r", " ").replace("\t", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sanitize_punctuation(text: str) -> str:
    """Normalize punctuation variants and collapse excessive repeats."""

    for src, dest in PUNCTUATION_REPLACEMENTS.items():
        text = text.replace(src, dest)

    # Remove spaces before sentence-ending punctuation
    text = re.sub(r"\s+([,;:!?])", r"\1", text)
    # Ensure a single space follows comma/semicolon when appropriate
    text = re.sub(r"([,;:])(?=[^\s])", r"\1 ", text)
    text = re.sub(r"\s+\.", ".", text)

    # Collapse repeated punctuation marks (e.g., "!!!" -> "!")
    text = re.sub(r"([.?!])\1{2,}", r"\1", text)

    return text


def strip_disallowed_symbols(text: str) -> str:
    """Remove control characters while keeping Sinhala letters and allowed punctuation."""

    cleaned: list[str] = []
    for ch in text:
        if ch in ALLOWED_PUNCTUATION:
            cleaned.append(ch)
            continue

        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("M") or cat.startswith("N") or cat == "Zs":
            cleaned.append(ch)
            continue

        # Keep Sinhala danda (end of sentence marker) if present
        if ch in {"\u0df4"}:  # Sinhala danda
            cleaned.append(ch)
            continue

        # Drop any remaining control marks/emojis/etc.
    return "".join(cleaned)


def normalize_sinhala_text(text: str) -> str:
    """Apply full normalization pipeline for a Sinhala transcript."""

    if not text:
        return ""

    # Unicode canonical form
    text = unicodedata.normalize("NFC", text)

    # Remove byte order marks / zero-width artifacts
    text = remove_zero_width(text)

    # Translate punctuation variants
    text = sanitize_punctuation(text)

    # Translate ASCII digits to Sinhala digits
    text = text.translate(ASCII_TO_SINHALA_DIGITS)

    # Remove stray control characters
    text = strip_disallowed_symbols(text)

    # Collapse whitespace and trim
    text = normalize_whitespace(text)

    return text


@dataclass
class NormalizationStats:
    """Statistics summarizing the effect of normalization on a corpus."""

    total_samples: int
    total_characters: int
    unique_characters: int
    removed_zero_width: int
    removed_control_chars: int
    sample_original: str | None
    sample_normalized: str | None

    @classmethod
    def from_pairs(cls, pairs: Sequence[tuple[str, str]]) -> "NormalizationStats":
        total_chars = 0
        char_counter: Counter[str] = Counter()
        removed_zero = 0
        removed_control = 0
        sample_original = None
        sample_normalized = None

        for original, normalized in pairs:
            if sample_original is None and original and original != normalized:
                sample_original = original
                sample_normalized = normalized

            total_chars += len(normalized)
            char_counter.update(normalized)

            removed_zero += sum(1 for ch in original if ch in ZERO_WIDTH_CHARS)
            removed_control += sum(1 for ch in original if unicodedata.category(ch).startswith("C"))

        return cls(
            total_samples=len(pairs),
            total_characters=total_chars,
            unique_characters=len(char_counter),
            removed_zero_width=removed_zero,
            removed_control_chars=removed_control,
            sample_original=sample_original,
            sample_normalized=sample_normalized,
        )


def summarize_corpus(texts: Sequence[str]) -> dict[str, int]:
    """Return counts of basic corpus statistics after normalization."""

    char_counter: Counter[str] = Counter()
    total_length = 0
    for text in texts:
        total_length += len(text)
        char_counter.update(text)

    return {
        "total_samples": len(texts),
        "total_characters": total_length,
        "unique_characters": len(char_counter),
    }


__all__ = [
    "normalize_sinhala_text",
    "NormalizationStats",
    "summarize_corpus",
]


