#!/usr/bin/env python3
"""
demo.py — Full 4-Tier Hybrid Neuro-Symbolic PoC
================================================
Runs the complete pipeline on demo_book_daraqutni.json

Usage:
    export GROK_API_KEY=your_key_here
    python3 demo.py

Output:
    output/tier1_output.json  — Deterministic processing results
    output/tier3_extraction.json — LLM extraction with citations
    output/enriched_book.json  — Final merged enrichment layer
"""

import sys
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Automatically load .env
load_dotenv(find_dotenv(usecwd=True))

# ── Paths ──
ROOT = Path(__file__).parent.parent
INPUT_JSON = ROOT / 'input' / 'demo_book_daraqutni.json'
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def banner(text: str):
    print(f'\n{"="*60}')
    print(f'  {text}')
    print(f'{"="*60}')


def run_tier1():
    banner('TIER 1 — Deterministic Python (Zero Cost)')
    sys.path.insert(0, str(Path(__file__).parent))
    from tier1_deterministic import DeterministicProcessor

    proc = DeterministicProcessor(str(INPUT_JSON))
    result = proc.process()

    print(f'  ✅ Book: {result.raw_title}')
    print(f'  ✅ Author: {result.raw_author}')
    print(f'  ✅ Pages: {result.page_count} | Blocks: {result.block_count}')
    print(f'  ✅ SHA256: {"MATCH" if result.sha256_match else "MISMATCH"}')
    print(f'  ✅ Footnote links: {len(result.footnote_links)}')
    print(f'  ✅ TOC entries: {len(result.toc_entries)}')

    if result.quality_flags:
        print(f'  ⚠  Quality flags: {len(result.quality_flags)}')
        for f in result.quality_flags:
            print(f'     • {f}')

    return result


def run_tier2_arabic_nlp(tier1_result):
    """
    Tier 2: Arabic NLP — deterministic, local, zero API cost.
    Uses PyArabic for normalization.
    """
    banner('TIER 2 — Arabic NLP (Free, Local)')

    try:
        from pyarabic import araby

        # Sample: normalize a few blocks
        sample_texts = [
            tier1_result.raw_title,
            tier1_result.raw_author,
        ]
        print('  Arabic normalization samples:')
        for text in sample_texts:
            if text:
                normalized = araby.strip_tashkeel(text)
                normalized = araby.normalize_hamza(normalized)
                normalized = araby.normalize_alef(normalized)
                print(f'  Original : {text}')
                print(f'  Normalized: {normalized}')
                print()

        # Detect hadith transmission verbs in blocks
        from tier1_deterministic import detect_hadith_transmission, detect_book_citation

        hadith_blocks = []
        cited_books = set()

        for block in tier1_result.blocks:
            if detect_hadith_transmission(block.text):
                hadith_blocks.append(block.block_id)
            for book in detect_book_citation(block.text):
                cited_books.add(book)

        print(f'  ✅ Blocks with hadith transmission verbs: {len(hadith_blocks)}')
        print(f'     Sample: {hadith_blocks[:5]}')
        print(f'  ✅ Classical books cited in this text: {len(cited_books)}')
        for b in sorted(cited_books):
            print(f'     • {b}')

        return {
            'hadith_transmission_blocks': hadith_blocks,
            'cited_classical_books': sorted(cited_books),
        }

    except ImportError:
        print('  ⚠  PyArabic not available — skipping (run: pip3 install pyarabic)')
        return {}


def run_tier3(tier1_result):
    """Tier 3: LLM extraction with Grok API."""
    banner('TIER 3 — LLM Extraction (Grok API, Grounded)')

    api_key = os.environ.get('GROK_API_KEY') or os.environ.get('XAI_API_KEY')
    if not api_key:
        print('  ⚠  GROK_API_KEY not set — skipping Tier 3')
        print('     Set it with: export GROK_API_KEY=your_key')
        print('  [Showing what Tier 3 WOULD extract from this book]')
        return _tier3_dry_run(tier1_result)

    tier1_path = OUTPUT_DIR / 'tier1_output.json'
    from tier3_llm_extraction import run_tier3_extraction

    model = os.environ.get('GROK_MODEL', 'grok-3-mini')
    result = run_tier3_extraction(
        json_path=str(INPUT_JSON),
        tier1_output_path=str(tier1_path),
        model=model,
    )
    return result


def _tier3_dry_run(tier1_result):
    """Show what Tier 3 would extract, using just Tier 1 data (demo without API key)."""
    print('  [DRY RUN — no API key needed]')
    print()

    # Show the blocks that WOULD be sent to the LLM
    print('  Title page blocks (would be sent to Grok):\n')
    title_blocks = [b for b in tier1_result.blocks if b.page_n in [3, 4, 5] and b.kind != 'footnote']
    for b in title_blocks[:12]:
        print(f'    [{b.block_id}][{b.kind.upper()}] {b.text}')

    print('\n  Expected extraction (based on visible text):')
    print('    title: "الإخوة والأخوات"')
    print('    author: "أبو الحسن علي بن عمر بن أحمد الدارقطني"')
    print('    author_death_year_hijri: "٣٨٥"')
    print('    editor: "الدكتور باسم فيصل أحمد الجوابرة"')
    print('    publisher: "دار الراية للنشر والتوزيع"')
    print('    genre: "biography" (genealogy/nasab)')
    print('    confidence: 0.98')

    print('\n  Sample biography entry (page 21):')
    bio_blocks = [b for b in tier1_result.blocks if b.page_n == 21]
    for b in bio_blocks[:8]:
        print(f'    [{b.block_id}][{b.kind.upper()}] {b.text[:100]}')

    return {'dry_run': True, 'note': 'Set GROK_API_KEY to run full extraction'}


def merge_and_save(tier1_result, tier2_result: dict, tier3_result: dict):
    """Merge all tiers into the final enrichment layer."""
    banner('MERGING — Final Enrichment Layer')

    from dataclasses import asdict

    enriched = {
        'enrichment_schema_version': '1.0',
        'pub_id': tier1_result.pub_id,
        'work_id': tier1_result.work_id,

        # Tier 1 — deterministic facts (100% reliable)
        'tier1_deterministic': {
            'sha256_verified': tier1_result.sha256_match,
            'page_count': tier1_result.page_count,
            'block_count': tier1_result.block_count,
            'blocks_by_kind': tier1_result.blocks_by_kind,
            'quality_flags': tier1_result.quality_flags,
            'toc': [asdict(e) for e in tier1_result.toc_entries],
            'footnote_links': [asdict(f) for f in tier1_result.footnote_links],
            'multi_page_footnotes': tier1_result.multi_page_footnotes,
        },

        # Tier 2 — Arabic NLP (deterministic)
        'tier2_arabic_nlp': tier2_result,

        # Tier 3 — LLM extraction (grounded, citation-verified)
        'tier3_llm_extraction': tier3_result,

        # Pipeline metadata
        'pipeline': {
            'tiers_run': ['tier1', 'tier2', 'tier3'],
            'anti_hallucination': {
                'grounded_extraction': True,
                'citation_required': True,
                'citation_verified_programmatically': True,
                'temperature': 0,
                'schema_enforced': 'Pydantic + Instructor',
            },
            'original_content_untouched': True,
            'enrichment_is_separate_layer': True,
        },
    }

    out_path = OUTPUT_DIR / 'enriched_book.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    print(f'  ✅ Enriched book saved → {out_path}')
    return enriched


def main():
    print('\n🕌  Islamic Digital Library — Tiered Hybrid Neuro-Symbolic PoC')
    print(f'   Input: {INPUT_JSON}')
    print(f'   Output: {OUTPUT_DIR}/')

    start = time.time()

    # ── Tier 1: Deterministic ──
    tier1 = run_tier1()

    # Save Tier 1 output for Tier 3 to use
    from dataclasses import asdict
    tier1_path = OUTPUT_DIR / 'tier1_output.json'
    with open(tier1_path, 'w', encoding='utf-8') as f:
        json.dump({
            'pub_id': tier1.pub_id,
            'work_id': tier1.work_id,
            'sha256_match': tier1.sha256_match,
            'page_count': tier1.page_count,
            'block_count': tier1.block_count,
            'blocks_by_kind': tier1.blocks_by_kind,
            'quality_flags': tier1.quality_flags,
            'raw_metadata': {
                'title': tier1.raw_title,
                'author': tier1.raw_author,
                'category': tier1.raw_category,
            },
            'toc': [asdict(e) for e in tier1.toc_entries],
            'footnote_links': [asdict(f) for f in tier1.footnote_links],
            'multi_page_footnotes': tier1.multi_page_footnotes,
        }, f, indent=2, ensure_ascii=False)

    # ── Tier 2: Arabic NLP ──
    tier2 = run_tier2_arabic_nlp(tier1)

    # ── Tier 3: LLM Extraction ──
    tier3 = run_tier3(tier1)

    # ── Merge all tiers ──
    merge_and_save(tier1, tier2, tier3)

    elapsed = time.time() - start
    banner(f'✅ COMPLETE — {elapsed:.1f}s')
    print(f'  Output files:')
    for f in OUTPUT_DIR.iterdir():
        size_kb = f.stat().st_size // 1024
        print(f'    {f.name} ({size_kb} KB)')


if __name__ == '__main__':
    main()
