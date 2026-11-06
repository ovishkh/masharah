"""
Test Suite for Masharah Pipeline
=================================
Tests:
- Tier 1 Deterministic processing (block parsing, footnote mapping, TOC extraction)
- Tier 2 Arabic normalization & gazetteer detection
- Tier 3 Citation verification logic & schema enforcement
"""

import sys
import os
import json
from pathlib import Path
import pytest

# Add poc directory to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'poc'))

from tier1_deterministic import (
    DeterministicProcessor,
    detect_hadith_transmission,
    detect_book_citation,
    normalize_arabic_simple,
    Block,
    BlockRef
)
from tier3_llm_extraction import (
    verify_citations_in_text,
    CitationVerification,
    BookMetadata,
    BiographyEntry,
    CitedField
)


@pytest.fixture
def sample_json_path():
    path = ROOT / 'input' / 'demo_book_daraqutni.json'
    assert path.exists(), f"Sample file not found at {path}"
    return str(path)


# ─────────────────────────────────────────────
# Tier 1 Tests
# ─────────────────────────────────────────────

def test_tier1_processor_initialization(sample_json_path):
    proc = DeterministicProcessor(sample_json_path)
    assert proc.raw is not None
    assert 'content' in proc.raw
    assert 'pub' in proc.raw


def test_tier1_block_parsing_and_counts(sample_json_path):
    proc = DeterministicProcessor(sample_json_path)
    result = proc.process()

    # Known facts about demo_book_daraqutni.json
    assert result.page_count == 68
    assert result.block_count == 594
    assert 'text' in result.blocks_by_kind
    assert 'footnote' in result.blocks_by_kind
    assert result.blocks_by_kind['text'] > 200
    assert result.blocks_by_kind['footnote'] > 300


def test_tier1_footnote_mapping(sample_json_path):
    proc = DeterministicProcessor(sample_json_path)
    result = proc.process()

    assert len(result.footnote_links) == 300
    # Check structure of mapped link
    first_link = result.footnote_links[0]
    assert first_link.anchor_block_id.startswith('b')
    assert first_link.footnote_block_id.startswith('b')
    assert first_link.anchor_char_end >= first_link.anchor_char_start
    assert len(first_link.footnote_text_preview) > 0


def test_tier1_toc_extraction(sample_json_path):
    proc = DeterministicProcessor(sample_json_path)
    result = proc.process()

    assert len(result.toc_entries) == 17
    # First TOC heading should be muhaqqiq intro
    assert "مقدمة" in result.toc_entries[0].text
    # Author intro should exist
    author_intro = [t for t in result.toc_entries if "مقدمة المؤلف" in t.text]
    assert len(author_intro) == 1


# ─────────────────────────────────────────────
# Tier 2 Tests (Arabic Linguistic Processing)
# ─────────────────────────────────────────────

def test_tier2_hadith_transmission_detection():
    assert detect_hadith_transmission("حدثنا عبد الله بن يوسف قال أخبرنا مالك") is True
    assert detect_hadith_transmission("أخبرنا الشيخ الحافظ أبو طاهر") is True
    assert detect_hadith_transmission("كتاب الطهارة وباب الوضوء") is False


def test_tier2_classical_book_citation_detection():
    sample_text = "قال ابن حجر في الإصابة ورواه البخاري في صحيحه وقال الذهبي في السير"
    cited = detect_book_citation(sample_text)
    assert "الإصابة" in cited
    assert "السير" in cited


def test_tier2_arabic_normalization():
    original = "الأُخوة والأَخوات"
    normalized = normalize_arabic_simple(original)
    # Check diacritics stripped
    assert "ُ" not in normalized
    assert "َ" not in normalized
    # Hamza normalized to alef
    assert "الأُ" not in normalized


# ─────────────────────────────────────────────
# Tier 3 Citation Verification & Anti-Hallucination
# ─────────────────────────────────────────────

def test_tier3_citation_verification_success():
    source_text = "[b000004][TEXT] أبي الحسن علي بن عمر بن أحمد الدارقطني"
    citations = [
        ("author", "أبي الحسن علي بن عمر بن أحمد الدارقطني", "أبي الحسن علي بن عمر بن أحمد الدارقطني")
    ]
    results = verify_citations_in_text(source_text, citations)
    assert len(results) == 1
    assert results[0].quote_found_in_source is True


def test_tier3_citation_verification_catches_hallucination():
    source_text = "[b000004][TEXT] أبي الحسن علي بن عمر بن أحمد الدارقطني"
    # LLM hallucinates an author not in source text
    hallucinated_citations = [
        ("author", "محمد بن إدريس الشافعي", "محمد بن إدريس الشافعي")
    ]
    results = verify_citations_in_text(source_text, hallucinated_citations)
    assert len(results) == 1
    assert results[0].quote_found_in_source is False
    assert "Quote NOT found" in (results[0].verification_note or "")


def test_tier3_pydantic_schema_validation():
    # Verify strict schema contract
    entry = BiographyEntry(
        block_id="b000105",
        name="فاطمة",
        source_quote="١ - فاطمة.",
        relation_type="sibling_of_prophet",
        confidence=0.95
    )
    assert entry.block_id == "b000105"
    assert entry.name == "فاطمة"
    assert entry.confidence == 0.95
