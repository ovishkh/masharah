# AI-Powered Islamic Digital Library — Masharah

Automated publishing and knowledge enrichment system for a digital library containing thousands of classical Islamic books in Arabic and other languages.

This project implements a **Tiered Hybrid Neuro-Symbolic Architecture** designed for high precision, zero hallucination, source traceability, and minimal cost at scale (~$0.10–$0.15 per book).

For the complete technical assessment, diagrams, and architecture analysis, see **[SYSTEM.md](SYSTEM.md)**.

---

## 🏛️ Architecture Overview

The system addresses the core assessment requirement by distinguishing which tasks belong to deterministic programming, classical NLP, single-pass LLMs, or multi-agent reasoning:

```
[Raw Book JSON + PDF]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Tier 1: Deterministic Python Engine (Free, ~0.1s)     │
│  • Block indexing (b000001 → b000594)                  │
│  • Maps 300 footnotes deterministically from refs[]    │
│  • Extracts 17 TOC entries from heading blocks         │
│  • Stitches multi-page footnotes across page breaks    │
└──────────────────────────┬─────────────────────────────┘
                           │ Anchored Block IDs & Footnote Map
                           ▼
┌────────────────────────────────────────────────────────┐
│  Tier 2: Arabic Classical NLP Engine (Local, Free)     │
│  • Normalizes diacritics & letter variants (PyArabic)  │
│  • Scans Hadith transmission verbs (حدثنا, أخبرنا)     │
│  • Gazetteer matching for 13+ cited classical titles   │
└──────────────────────────┬─────────────────────────────┘
                           │ Pre-filtered context & clean spans
                           ▼
┌────────────────────────────────────────────────────────┐
│  Tier 3: Grounded LLM Extraction (Groq + Qwen 27B)     │
│  • Strict Pydantic JSON schemas via Instructor         │
│  • Mandatory verbatim Arabic quotes for every field    │
│  • "Null over guess" policy (never hallucinates facts) │
│  • Programmatic substring verification of quotes       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Enrichment Layer (Stand-off JSON)                     │
│  • Original book JSON/PDF remains 100% untouched       │
│  • Full bi-directional links to source block IDs       │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart & PoC

### 1. Requirements & Setup

Ensure Python 3.10+ is installed:

```bash
git clone https://github.com/ovishkh/masharah.git
cd masharah
pip3 install instructor pyarabic openai pydantic rich python-dotenv
```

### 2. Environment Configuration

Copy the template and configure your API key:

```bash
cp .env.example .env
```

Inside `.env`:
```env
GROK_API_KEY=gsk_your_groq_api_key_here
GROK_BASE_URL=https://api.groq.com/openai/v1
GROK_MODEL=qwen/qwen3.8-27b
```

### 3. Run the Pipeline

To run the complete 4-tier pipeline on the sample book (`demo_book_daraqutni.json`):

```bash
cd poc
python3 demo.py
```

*Note: Even without an API key, Tier 1 and Tier 2 run deterministically at zero cost.*

---

## 📁 Repository Structure

```
masharah/
├── SYSTEM.md                  # Complete Technical Assessment & Architecture Plan
├── README.md                  # Project overview & quickstart guide
├── .env.example               # Environment variables template
├── input/
│   ├── demo_book_daraqutni.json    # Sample digitized book (Shamela format)
│   └── demo_book_daraqutni_v1.pdf  # Sample source manuscript PDF
├── output/
│   ├── tier1_output.json      # Deterministic block index & 300 footnote links
│   ├── tier3_extraction.json  # Grounded LLM extractions with verified citations
│   └── enriched_book.json     # Final stand-off enrichment layer
├── poc/
│   ├── demo.py                # Main end-to-end execution script
│   ├── tier1_deterministic.py # Tier 1: Zero-cost structural indexing & footnotes
│   └── tier3_llm_extraction.py# Tier 3: Grounded Qwen extraction via Groq + Instructor
└── docs/
    └── Agentic AI_ Technical Assignment.pdf # Original assignment specification
```

---

## 🔍 Where to Find More Islamic Books to Test

The sample input (`demo_book_daraqutni.json`) is based on the **Al-Maktaba Al-Shamela (المكتبة الشاملة)** catalog schema (`source: src-shamela`, book key `26`). You can obtain thousands of books in this domain from the following resources:

1. **Al-Maktaba Al-Shamela Official Web API & Catalog**:
   - Website: [shamela.ws](https://shamela.ws/)
   - Over 8,000+ classical and contemporary digitized Islamic books across Fiqh, Hadith, Tafseer, Aqeedah, and Biographies.
   - Any book URL follows `https://shamela.ws/book/{id}` (e.g. book `26` is our sample *Al-Ikhwa wal-Akhawat*).

2. **OpenITI (Open Islamicate Texts Initiative)**:
   - GitHub: [github.com/OpenITI](https://github.com/OpenITI)
   - Machine-readable corpus containing over 10,000+ classical Arabic texts with metadata, structural markdown headers (`### |EDITOR|`), and biographical records.

3. **Shamela SQLite / Bok Archives**:
   - Available via [shamela.ws/downloads](https://shamela.ws/) or open-source parsers on GitHub (e.g. `shamela-parser` or `openiti-corpus-converter`) which export to the exact block/page JSON structure used here.
