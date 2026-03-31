"""
chunker.py — Structure-aware chunking for RAG.

Takes the extracted-page output from content_extractor.py and produces
embeddable chunks with context headers. Chunking strategy adapts to
block type:

  schedule  → keep each entity as a self-contained chunk; also produce a
              combined "all schedules" chunk for broad queries.
  faq       → question-only + full Q&A pair (two chunks per pair).
  table     → split by rows, keep header with every chunk.
  list      → split by items, keep groups together.
  text      → sentence-boundary splitting with overlap.

Every chunk carries:
  • A context header (Page / Section / Subsection) so embedding captures
    hierarchical context even when the chunk is small.
  • Metadata for the DB (page_url, page_title, section_title, page_type,
    chunk_index).

No content is truncated before chunking.
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Chunk size targets (characters).
# 800 keeps related info together (e.g. a full schedule block, a policy section)
# while staying small enough for precise retrieval.
MAX_CHUNK_SIZE = 800
MIN_CHUNK_SIZE = 60
OVERLAP_SIZE = 75       # overlap between consecutive text chunks

# Safety cap: if a single page produces more than this many chunks, it's
# almost certainly a bloated newsletter/report — keep only the first N.
MAX_CHUNKS_PER_PAGE = 40


# ============================================================================
# Low-level text splitting
# ============================================================================

def _split_text(
    text: str,
    max_size: int = MAX_CHUNK_SIZE,
    overlap: int = OVERLAP_SIZE,
) -> List[str]:
    """Split text at sentence boundaries with overlap between chunks."""
    if len(text) <= max_size:
        return [text] if len(text) >= MIN_CHUNK_SIZE else []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_size:
            chunks.append(current.strip())
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + " " + sentence
            else:
                current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence

    if current.strip() and len(current.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(current.strip())

    return chunks


def _split_table(text: str, max_size: int = MAX_CHUNK_SIZE) -> List[str]:
    """Split table text by rows, keeping the header with every chunk."""
    lines = text.split("\n")
    if len(lines) < 2:
        return _split_text(text, max_size)

    header_lines: List[str] = []
    data_lines: List[str] = []
    for i, line in enumerate(lines):
        if i == 0:
            header_lines.append(line)
        elif re.match(r"^[-\s|]+$", line):
            header_lines.append(line)
        else:
            data_lines.append(line)

    header = "\n".join(header_lines)
    chunks: List[str] = []
    current_rows: List[str] = []
    current_size = len(header)

    for row in data_lines:
        if current_size + len(row) + 1 > max_size and current_rows:
            chunks.append(header + "\n" + "\n".join(current_rows))
            current_rows = []
            current_size = len(header)
        current_rows.append(row)
        current_size += len(row) + 1

    if current_rows:
        chunks.append(header + "\n" + "\n".join(current_rows))

    return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE]


def _split_list(text: str, max_size: int = MAX_CHUNK_SIZE) -> List[str]:
    """Split list text by item boundaries, keeping related items together."""
    items = re.split(r"\n(?=[-\d]+[.)]\s|\- )", text)
    chunks: List[str] = []
    current = ""

    for item in items:
        if current and len(current) + len(item) + 1 > max_size:
            chunks.append(current.strip())
            current = item
        else:
            current = (current + "\n" + item).strip() if current else item

    if current.strip() and len(current.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(current.strip())

    return chunks


def _split_by_format(
    text: str, content_format: str, max_size: int = MAX_CHUNK_SIZE
) -> List[str]:
    """Dispatch to the right splitter based on content format."""
    if len(text) <= max_size:
        return [text] if len(text) >= MIN_CHUNK_SIZE else []
    if content_format == "table":
        return _split_table(text, max_size)
    if content_format == "list":
        return _split_list(text, max_size)
    return _split_text(text, max_size)


# ============================================================================
# Context header builder
# ============================================================================

def _build_context_header(page_title: str, heading_path: List[str]) -> str:
    """Build a context prefix for a chunk so the embedding captures hierarchy.

    Example output:
        Page: Opening Hours
        Section: Jafet Library
        Subsection: Computer Labs
    """
    parts: List[str] = []
    if page_title:
        parts.append(f"Page: {page_title}")
    labels = ["Section", "Subsection", "Sub-subsection"]
    for i, h in enumerate(heading_path):
        label = labels[i] if i < len(labels) else f"Level-{i + 1}"
        parts.append(f"{label}: {h}")
    return "\n".join(parts)


# ============================================================================
# Block-type–specific chunking
# ============================================================================

def _chunk_schedule_blocks(
    blocks: List[Dict], page_title: str, url: str
) -> List[Dict]:
    """Chunk schedule blocks: one combined + one per entity.

    A combined chunk lets a broad query ("what are the hours?") retrieve
    everything.  Individual chunks let specific queries ("Jafet hours?")
    retrieve targeted info.
    """
    chunks: List[Dict] = []
    idx = 0

    # Collect per-entity texts
    entity_texts: List[tuple] = []  # (heading, full_text)
    for block in blocks:
        heading = block.get("heading", "")
        content = block["content"]
        entity_texts.append((heading, content))

    # ── Combined chunk (all entities) ──
    if len(entity_texts) > 1:
        combined_parts = [f"{page_title} — Opening Hours (All Locations)"]
        for heading, text in entity_texts:
            if heading:
                combined_parts.append(f"\n{heading}:\n{text}")
            else:
                combined_parts.append(f"\n{text}")
        combined = "\n".join(combined_parts)

        # Allow up to 4× max for combined schedules — a single retrieval
        # should return ALL locations.  6 libraries × ~150 chars ≈ 900 chars
        # plus closures can push to 2-3K.  Embedding models handle this fine.
        for sub in _split_text(combined, MAX_CHUNK_SIZE * 4):
            chunks.append({
                "chunk_text": sub,
                "page_url": url,
                "page_title": page_title,
                "section_title": "Hours: All Locations",
                "page_type": "hours_contact",
                "chunk_index": idx,
            })
            idx += 1

    # ── Per-entity chunks ──
    for heading, text in entity_texts:
        header = _build_context_header(page_title, [heading] if heading else [])
        full = f"{header}\n\n{text}" if header else text

        for sub in _split_by_format(full, "text"):
            chunks.append({
                "chunk_text": sub,
                "page_url": url,
                "page_title": page_title,
                "section_title": f"Hours: {heading}" if heading else "Hours",
                "page_type": "hours_contact",
                "chunk_index": idx,
            })
            idx += 1

    return chunks


def _chunk_faq_blocks(
    blocks: List[Dict], page_title: str, url: str
) -> List[Dict]:
    """Chunk FAQ blocks: question-only + full Q&A per pair.

    The question-only chunk embeds close to how users phrase their queries.
    The full Q&A chunk provides the answer for generation.
    """
    chunks: List[Dict] = []
    idx = 0

    for block in blocks:
        content = block["content"]
        heading = block.get("heading", "")

        # Try to split into Q / A
        m = re.match(r"Q:\s*(.+?)\nA:\s*(.+)", content, re.S)
        if m:
            question = m.group(1).strip()
            answer = m.group(2).strip()
            section = f"FAQ: {question[:80]}"

            header = _build_context_header(page_title, [heading] if heading else [])

            # Chunk 1: question only (retrieval-optimised)
            q_text = f"{header}\n\n{question}" if header else question
            chunks.append({
                "chunk_text": q_text,
                "page_url": url,
                "page_title": page_title,
                "section_title": section,
                "page_type": "faq",
                "chunk_index": idx,
            })
            idx += 1

            # Chunk 2: full Q+A (answer extraction)
            qa_text = f"{header}\n\nQ: {question}\nA: {answer}" if header else content
            chunks.append({
                "chunk_text": qa_text,
                "page_url": url,
                "page_title": page_title,
                "section_title": section,
                "page_type": "faq",
                "chunk_index": idx,
            })
            idx += 1
        else:
            # Non-pair FAQ content — chunk as text
            for sub in _split_text(content):
                chunks.append({
                    "chunk_text": sub,
                    "page_url": url,
                    "page_title": page_title,
                    "section_title": heading or "FAQ",
                    "page_type": "faq",
                    "chunk_index": idx,
                })
                idx += 1

    return chunks


def _chunk_generic_blocks(
    blocks: List[Dict], page_title: str, url: str, page_type: str
) -> List[Dict]:
    """Chunk generic, contact, policy, and other block types.

    Each block is split according to its content_format, with a context
    header prepended.
    """
    chunks: List[Dict] = []
    idx = 0

    for block in blocks:
        content = block["content"]
        heading = block.get("heading", "")
        heading_path = block.get("heading_path", [])
        fmt = block.get("content_format", "text")

        header = _build_context_header(page_title, heading_path)
        full = f"{header}\n\n{content}" if header else content

        sub_chunks = _split_by_format(full, fmt)
        for sub in sub_chunks:
            chunks.append({
                "chunk_text": sub,
                "page_url": url,
                "page_title": page_title,
                "section_title": heading,
                "page_type": page_type,
                "chunk_index": idx,
            })
            idx += 1

    return chunks


# ============================================================================
# Public API
# ============================================================================

def chunk_page(extracted_page: Dict) -> List[Dict]:
    """Convert an extracted page (from content_extractor) into RAG chunks.

    Routes blocks to type-specific chunkers:
      • schedule blocks → combined + per-entity
      • faq blocks      → question + full-QA pairs
      • everything else → generic section-based
    """
    url = extracted_page["url"]
    title = extracted_page["page_title"]
    page_type = extracted_page.get("page_type", "general")
    blocks = extracted_page.get("blocks", [])

    if not blocks:
        return []

    # Separate blocks by type
    schedule_blocks = [b for b in blocks if b.get("block_type") == "schedule"]
    faq_blocks = [b for b in blocks if b.get("block_type") == "faq"]
    other_blocks = [
        b for b in blocks
        if b.get("block_type") not in ("schedule", "faq")
    ]

    all_chunks: List[Dict] = []

    if schedule_blocks:
        all_chunks.extend(
            _chunk_schedule_blocks(schedule_blocks, title, url)
        )
    if faq_blocks:
        all_chunks.extend(
            _chunk_faq_blocks(faq_blocks, title, url)
        )
    if other_blocks:
        all_chunks.extend(
            _chunk_generic_blocks(other_blocks, title, url, page_type)
        )

    # If nothing was produced (all blocks too small), try a single-blob fallback
    if not all_chunks and blocks:
        combined = "\n\n".join(b["content"] for b in blocks)
        for sub in _split_text(combined):
            all_chunks.append({
                "chunk_text": sub,
                "page_url": url,
                "page_title": title,
                "section_title": "",
                "page_type": page_type,
                "chunk_index": len(all_chunks),
            })

    # Safety cap: bloated pages (newsletters, reports) can produce thousands
    # of tiny chunks.  Keep only the first N — if a page needs more than 40
    # chunks it's not useful chatbot content.
    if len(all_chunks) > MAX_CHUNKS_PER_PAGE:
        logger.warning(
            f"Page '{title}' produced {len(all_chunks)} chunks, "
            f"capping at {MAX_CHUNKS_PER_PAGE}"
        )
        all_chunks = all_chunks[:MAX_CHUNKS_PER_PAGE]

    # Renumber chunk indices sequentially
    for i, c in enumerate(all_chunks):
        c["chunk_index"] = i

    return all_chunks


def chunk_pages(extracted_pages: List[Dict]) -> List[Dict]:
    """Chunk a batch of extracted pages.

    Returns a flat list of all chunks with metadata, ready for embedding.
    """
    all_chunks: List[Dict] = []
    for page in extracted_pages:
        page_chunks = chunk_page(page)
        all_chunks.extend(page_chunks)

    type_counts: Dict[str, int] = {}
    for c in all_chunks:
        pt = c.get("page_type", "general")
        type_counts[pt] = type_counts.get(pt, 0) + 1

    n_pages = len(extracted_pages)
    logger.info(
        f"Chunked {n_pages} pages → {len(all_chunks)} chunks "
        f"(avg {len(all_chunks) / max(n_pages, 1):.1f}/page). "
        f"By type: {type_counts}"
    )
    return all_chunks
