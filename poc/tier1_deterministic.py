"""
TIER 1 — Deterministic Python Processing
=========================================
Zero LLM calls. Zero cost. Zero hallucination.
Handles: schema validation, block indexing, footnote stitching,
         ref mapping, TOC extraction, quality checks.

This runs BEFORE any LLM and anchors all block_id + char offsets.
The LLM never touches the original block structure.
"""

from __future__ import annotations
import json
import hashlib
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# Data structures (pure Python, no LLM)
# ─────────────────────────────────────────────

@dataclass
class BlockRef:
    """An inline reference within a text block pointing to a footnote block."""
    start: int          # char offset in source block text
    end: int            # char offset in source block text
    to: str             # target block_id (e.g. "b000045")


@dataclass
class Block:
    block_id: str                        # e.g. "b000041"
    kind: str                            # text | heading | footnote | toc_heading
    text: str
    page_n: int
    volume: int
    scan_page: int
    char_len: int = 0
    refs: list[BlockRef] = field(default_factory=list)
    continues: Optional[bool] = None     # footnote spans multiple pages
    is_continuation: bool = False        # this block continues from previous page


@dataclass
class FootnoteLink:
    """Bidirectional mapping: anchor block ↔ footnote block."""
    anchor_block_id: str
    anchor_char_start: int
    anchor_char_end: int
    footnote_block_id: str
    footnote_text_preview: str           # first 80 chars


@dataclass
class TocEntry:
    block_id: str
    kind: str                            # toc_heading | heading
    text: str
    page_n: int
    level: int                           # 1 = top-level, 2 = sub, etc.


@dataclass
class DeterministicOutput:
    """Everything Tier 1 produces — no LLM involved."""
    pub_id: str
    work_id: str
    content_sha256: str
    computed_sha256: str
    sha256_match: bool
    page_count: int
    block_count: int
    blocks_by_kind: dict[str, int]
    blocks: list[Block]
    footnote_links: list[FootnoteLink]
    toc_entries: list[TocEntry]
    multi_page_footnotes: list[dict]     # footnotes that span pages
    quality_flags: list[str]
    # Raw metadata from JSON (not LLM-generated)
    raw_title: str
    raw_author: str
    raw_category: str
    raw_page_count: int
    raw_volume_count: int
    source_code: str


# ─────────────────────────────────────────────
# Arabic text utilities (rule-based, no model)
# ─────────────────────────────────────────────

HADITH_TRANSMISSION_VERBS = {
    'حدثنا', 'حدثني', 'أخبرنا', 'أخبرني', 'ثنا', 'نا',
    'أنبأنا', 'أنبأني', 'سمعت', 'روى', 'قال',
}

KNOWN_BOOK_TITLES = {
    'الطبقات الكبرى', 'السير', 'الإصابة', 'أسد الغابة',
    'الاستيعاب', 'تاريخ بغداد', 'الأنساب', 'سير أعلام النبلاء',
    'تذكرة الحفاظ', 'تهذيب التهذيب', 'صحيح البخاري',
    'صحيح مسلم', 'سنن أبي داود', 'سنن الترمذي',
    'سنن النسائي', 'سنن ابن ماجه', 'المستدرك',
}

ARABIC_ORDINAL_PATTERN = re.compile(r'^[٠-٩\d]+\s*[-–]\s*|^[٠-٩]+\.')
SECTION_MARKER_PATTERN = re.compile(r'^\*\s*\*\s*\*')
CONTINUES_MARKER = '='    # Shamela convention: footnote block ending with = continues


def detect_hadith_transmission(text: str) -> bool:
    """Rule-based: does this block start a hadith narration chain?"""
    first_tokens = set(text.split()[:4])
    return bool(first_tokens & HADITH_TRANSMISSION_VERBS)


def detect_book_citation(text: str) -> list[str]:
    """Rule-based: find known classical book titles referenced in text."""
    found = []
    for title in KNOWN_BOOK_TITLES:
        if title in text:
            found.append(title)
    return found


def is_biography_entry(text: str) -> bool:
    """Heuristic: numbered entries like '١ - فاطمة' are biography list items."""
    return bool(ARABIC_ORDINAL_PATTERN.match(text.strip()))


def normalize_arabic_simple(text: str) -> str:
    """
    Minimal normalization without CAMeL Tools dependency.
    For production, replace with pyarabic.araby calls.
    """
    # Normalize hamza forms
    text = re.sub(r'[أإآ]', 'ا', text)
    # Normalize ta marbuta
    text = re.sub(r'ة', 'ه', text)
    # Strip common diacritics (harakat)
    diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    text = diacritics.sub('', text)
    return text


# ─────────────────────────────────────────────
# Core Tier 1 processor
# ─────────────────────────────────────────────

class DeterministicProcessor:
    """
    Processes the digitized book JSON entirely with deterministic code.
    No LLM. No network calls. Pure Python.
    """

    def __init__(self, json_path: str):
        self.path = Path(json_path)
        with open(self.path) as f:
            self.raw = json.load(f)

    def _verify_sha256(self) -> tuple[str, str, bool]:
        """Verify content integrity against stored SHA256."""
        claimed = self.raw['content']['content_sha256']
        content_str = json.dumps(self.raw['content'], separators=(',', ':'), ensure_ascii=False)
        computed = hashlib.sha256(content_str.encode()).hexdigest()
        return claimed, computed, (claimed == computed)

    def _parse_all_blocks(self) -> list[Block]:
        """Extract every block from every page into flat indexed list."""
        blocks = []
        pages = self.raw['content']['pages']
        for page in pages:
            for raw_block in page.get('blocks', []):
                refs = [
                    BlockRef(r['start'], r['end'], r['to'])
                    for r in raw_block.get('refs', [])
                ]
                text = raw_block.get('text', '')
                # Detect footnote continuation (Shamela uses trailing '=')
                continues = raw_block.get('continues')
                is_continuation = text.strip().startswith('=')
                if is_continuation:
                    text = text.lstrip('=').strip()

                blocks.append(Block(
                    block_id=raw_block['b'],
                    kind=raw_block['kind'],
                    text=text,
                    page_n=page['n'],
                    volume=page['volume'],
                    scan_page=page['scan_page'],
                    char_len=len(text),
                    refs=refs,
                    continues=continues,
                    is_continuation=is_continuation,
                ))
        return blocks

    def _build_footnote_links(self, blocks: list[Block]) -> list[FootnoteLink]:
        """
        Map every inline ref marker in text blocks to its footnote block.
        100% deterministic — uses the refs[] arrays already in the JSON.
        """
        block_map = {b.block_id: b for b in blocks}
        links = []
        for block in blocks:
            if block.kind in ('text', 'heading') and block.refs:
                for ref in block.refs:
                    target = block_map.get(ref.to)
                    if target:
                        links.append(FootnoteLink(
                            anchor_block_id=block.block_id,
                            anchor_char_start=ref.start,
                            anchor_char_end=ref.end,
                            footnote_block_id=ref.to,
                            footnote_text_preview=target.text[:80],
                        ))
        return links

    def _build_toc(self, blocks: list[Block]) -> list[TocEntry]:
        """Extract hierarchical TOC from heading and toc_heading blocks."""
        toc = []
        for block in blocks:
            if block.kind in ('toc_heading', 'heading'):
                # Simple level heuristic: toc_heading=1, heading=2
                level = 1 if block.kind == 'toc_heading' else 2
                toc.append(TocEntry(
                    block_id=block.block_id,
                    kind=block.kind,
                    text=block.text,
                    page_n=block.page_n,
                    level=level,
                ))
        return toc

    def _detect_multi_page_footnotes(self, blocks: list[Block]) -> list[dict]:
        """
        Stitch footnote blocks that continue across pages.
        Shamela marks them with continues=true and next block starts with '='.
        """
        footnote_blocks = {b.block_id: b for b in blocks if b.kind == 'footnote'}
        multi = []
        for bid, block in footnote_blocks.items():
            if block.continues:
                # Find the continuation block on the next page
                # It will be the footnote starting with '=' on page_n + 1
                continuations = [
                    b for b in blocks
                    if b.kind == 'footnote'
                    and b.is_continuation
                    and b.page_n == block.page_n + 1
                ]
                if continuations:
                    full_text = block.text + ' ' + continuations[0].text
                    multi.append({
                        'primary_block': bid,
                        'continuation_block': continuations[0].block_id,
                        'full_text': full_text,
                        'page_start': block.page_n,
                        'page_end': continuations[0].page_n,
                    })
        return multi

    def _quality_checks(self, blocks: list[Block], claimed_pages: int) -> list[str]:
        """Run deterministic quality gates."""
        flags = []
        actual_pages = len(set(b.page_n for b in blocks))

        if actual_pages < claimed_pages:
            diff_pct = (claimed_pages - actual_pages) / claimed_pages * 100
            flags.append(f"PAGE_COUNT_SHORT: claimed={claimed_pages}, actual={actual_pages} ({diff_pct:.1f}% short)")

        # Check for orphaned refs (refs pointing to non-existent blocks)
        block_ids = {b.block_id for b in blocks}
        for block in blocks:
            for ref in block.refs:
                if ref.to not in block_ids:
                    flags.append(f"ORPHANED_REF: {block.block_id} refs {ref.to} which doesn't exist")

        # Check for empty blocks
        empty = [b.block_id for b in blocks if not b.text.strip()]
        if empty:
            flags.append(f"EMPTY_BLOCKS: {len(empty)} blocks have no text: {empty[:5]}")

        return flags

    def process(self) -> DeterministicOutput:
        """Run the full deterministic pipeline. Returns structured output."""
        pub = self.raw['pub']
        metadata = self.raw['metadata']

        # 1. Integrity check
        claimed_sha, computed_sha, sha_match = self._verify_sha256()

        # 2. Parse all blocks into flat structure
        blocks = self._parse_all_blocks()

        # 3. Block counts by kind
        kind_counts: dict[str, int] = {}
        for b in blocks:
            kind_counts[b.kind] = kind_counts.get(b.kind, 0) + 1

        # 4. Footnote links (deterministic from refs[] arrays)
        footnote_links = self._build_footnote_links(blocks)

        # 5. TOC
        toc = self._build_toc(blocks)

        # 6. Multi-page footnotes
        multi_page = self._detect_multi_page_footnotes(blocks)

        # 7. Quality checks
        quality_flags = self._quality_checks(blocks, pub['page_count'])

        return DeterministicOutput(
            pub_id=pub['id'],
            work_id=pub['work_id'],
            content_sha256=claimed_sha,
            computed_sha256=computed_sha,
            sha256_match=sha_match,
            page_count=len(set(b.page_n for b in blocks)),
            block_count=len(blocks),
            blocks_by_kind=kind_counts,
            blocks=blocks,
            footnote_links=footnote_links,
            toc_entries=toc,
            multi_page_footnotes=multi_page,
            quality_flags=quality_flags,
            raw_title=pub.get('title_as_source', ''),
            raw_author=pub.get('author_as_source', ''),
            raw_category=pub.get('category', {}).get('name', ''),
            raw_page_count=pub.get('page_count', 0),
            raw_volume_count=pub.get('volume_count', 0),
            source_code=pub.get('source_code', ''),
        )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    json_path = sys.argv[1] if len(sys.argv) > 1 else '../input/demo_book_daraqutni.json'

    console.print(Panel('[bold cyan]TIER 1 — Deterministic Processing[/bold cyan]\nZero LLM · Zero Cost · Zero Hallucination'))

    proc = DeterministicProcessor(json_path)
    result = proc.process()

    # Summary table
    table = Table(title='Tier 1 Results', show_header=True)
    table.add_column('Field', style='cyan')
    table.add_column('Value', style='green')

    table.add_row('Book', result.raw_title)
    table.add_row('Author', result.raw_author)
    table.add_row('Category', result.raw_category)
    table.add_row('Pages processed', str(result.page_count))
    table.add_row('Total blocks', str(result.block_count))
    table.add_row('SHA256 match', '✅ YES' if result.sha256_match else '❌ NO')
    table.add_row('Footnote links mapped', str(len(result.footnote_links)))
    table.add_row('TOC entries', str(len(result.toc_entries)))
    table.add_row('Multi-page footnotes', str(len(result.multi_page_footnotes)))

    for kind, count in result.blocks_by_kind.items():
        table.add_row(f'  blocks[{kind}]', str(count))

    console.print(table)

    # Quality flags
    if result.quality_flags:
        console.print('\n[yellow]⚠ Quality Flags:[/yellow]')
        for flag in result.quality_flags:
            console.print(f'  • {flag}')
    else:
        console.print('\n[green]✅ All quality checks passed[/green]')

    # TOC
    console.print('\n[bold]Table of Contents (detected):[/bold]')
    for entry in result.toc_entries:
        indent = '  ' * (entry.level - 1)
        console.print(f'  {indent}[{entry.block_id}] p{entry.page_n}: {entry.text}')

    # Sample footnote links
    console.print(f'\n[bold]Sample Footnote Links (first 5 of {len(result.footnote_links)}):[/bold]')
    for link in result.footnote_links[:5]:
        console.print(
            f'  [{link.anchor_block_id}] chars {link.anchor_char_start}-{link.anchor_char_end} '
            f'→ [{link.footnote_block_id}]: {link.footnote_text_preview[:60]}…'
        )

    # Save to output file
    output_path = Path('../output/tier1_output.json')
    output_path.parent.mkdir(exist_ok=True)

    # Serialize (convert dataclasses to dicts)
    output_data = {
        'pub_id': result.pub_id,
        'work_id': result.work_id,
        'sha256_match': result.sha256_match,
        'page_count': result.page_count,
        'block_count': result.block_count,
        'blocks_by_kind': result.blocks_by_kind,
        'quality_flags': result.quality_flags,
        'raw_metadata': {
            'title': result.raw_title,
            'author': result.raw_author,
            'category': result.raw_category,
            'page_count': result.raw_page_count,
            'volume_count': result.raw_volume_count,
            'source_code': result.source_code,
        },
        'toc': [asdict(e) for e in result.toc_entries],
        'footnote_links': [asdict(f) for f in result.footnote_links],
        'multi_page_footnotes': result.multi_page_footnotes,
        # Blocks index (without full text to keep output manageable)
        'block_index': [
            {
                'block_id': b.block_id,
                'kind': b.kind,
                'page_n': b.page_n,
                'volume': b.volume,
                'char_len': b.char_len,
                'has_refs': len(b.refs) > 0,
                'is_continuation': b.is_continuation,
            }
            for b in result.blocks
        ],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    console.print(f'\n[bold green]✅ Tier 1 output saved → {output_path}[/bold green]')
    console.print('[dim]This output is passed to Tier 2 (Arabic NLP) and Tier 3 (LLM extraction)[/dim]')

