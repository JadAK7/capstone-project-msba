"""
content_extractor.py — Generic, page-agnostic content extraction from HTML/text.

Replaces the previous scraper_cleaner + document_processor pipeline with a
unified layered strategy that adapts to whatever structure the page provides:

  Layer 1  Semantic HTML — headings, <table>, <ul>/<ol>, <section>, <article>
  Layer 2  DOM block detection — content-density scoring, repeated-sibling
           patterns (cards, accordion items, div-based tables)
  Layer 3  Text pattern detection — schedules, FAQ pairs, contacts, policies
  Layer 4  Fallback — paragraph-boundary splitting

Each layer enriches or replaces the previous when it finds richer structure.
No site-specific selectors or hardcoded page names.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from html import unescape

try:
    from bs4 import BeautifulSoup, NavigableString, Tag, Comment
except ImportError:
    BeautifulSoup = None

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration — all generic, nothing site-specific
# ============================================================================

# Tags whose entire subtree is never useful content
REMOVE_TAGS: Set[str] = {
    "script", "style", "noscript", "iframe", "svg", "path",
    "button", "input", "select", "textarea", "form", "map",
}

# CSS selectors for common non-content regions.
# Ordered roughly by specificity so the broadest patterns come last.
NOISE_SELECTORS: List[str] = [
    # Semantic landmarks
    "nav", "[role='navigation']", "[role='banner']", "[role='search']",
    "footer", "[role='contentinfo']",
    "header",
    # Breadcrumbs
    "[class*='breadcrumb']", "[id*='breadcrumb']",
    # Side / top navigation (common class patterns)
    "[class*='sidenav']", "[class*='side-nav']", "[class*='sideNav']",
    "[class*='topnav']", "[class*='top-nav']", "[class*='topNav']",
    "[class*='quicklaunch']", "[class*='quickLaunch']",
    "[class*='quick-launch']",
    # Footers / site info
    "[class*='footer']", "[id*='footer']",
    # Cookie / consent
    "[class*='cookie']", "[id*='cookie']",
    "[class*='consent']", "[id*='consent']",
    # Social / share
    "[class*='social-share']", "[class*='share-button']",
    "[class*='socialShare']", "[class*='shareButton']",
    # Skip links
    "[class*='skip-nav']", "[class*='skip-to']",
    # Explicitly hidden
    "[aria-hidden='true']",
    "[style*='display:none']", "[style*='display: none']",
    # SharePoint common chrome
    "[class*='ms-dialogHidden']", "[class*='ms-rtestate-notify']",
]

# Selectors tried in order to find the main content area.
MAIN_CONTENT_SELECTORS: List[str] = [
    "main", "[role='main']",
    "article", "[role='article']",
    "#content", "#main-content", "#mainContent", "#main_content",
    ".content", ".main-content", ".page-content", ".mainContent",
    # SharePoint patterns
    "#contentRow", ".ms-rte-layoutszone-inner",
    "#DeltaPlaceHolderMain", ".ms-webpartPage-root",
]

# Generic text-level noise patterns (truly universal boilerplate).
_NOISE_TEXT_PATTERNS = [
    re.compile(r"^\s*skip to (?:main )?content\s*$", re.I | re.M),
    re.compile(r"^\s*sign in\s*$", re.I | re.M),
    re.compile(r"\u00a9\s*\d{4}.*$", re.M),                      # © 2024 …
    re.compile(r"all rights reserved.*$", re.I | re.M),
    re.compile(r"this site uses cookies.*?(?:accept|ok|got it)", re.I | re.S),
    re.compile(r"^\s*\d+\s+shares?\s+\d+.*$", re.I | re.M),      # social counts
]

# Heading tags in order
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# Block-level tags whose children should be walked individually
_BLOCK_TAGS = {
    "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol",
    "p", "div", "section", "article", "blockquote", "pre", "details",
    "figure", "figcaption", "main", "aside", "dl",
}

# Tags that wrap inline content (not block containers)
_INLINE_PARENTS = {"table", "ul", "ol", "li", "tr", "td", "th", "dl", "dt", "dd"}


# ============================================================================
# Pattern regexes — generic, not site-specific
# ============================================================================

_TIME_RANGE_RE = re.compile(
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)\s*[-–—to]+\s*"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)"
)
_TIME_LOOSE_RE = re.compile(
    r"(?:\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)|closed|24\s*hours|24/7)", re.I
)
_DAY_RE = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"Mon\.?|Tue\.?|Wed\.?|Thu\.?|Fri\.?|Sat\.?|Sun\.?|"
    r"Weekday|Weekend|Daily|"
    r"Monday\s*[-–—to]+\s*\w+day|Mon\s*[-–]\s*\w+|M\s*[-–]\s*F|"
    r"Saturday\s*[-–—&]\s*Sunday)", re.I
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
)
_QUESTION_RE = re.compile(
    r"(?:^|\n)\s*(?:Q[:.]?\s*|Question[:.]?\s*|\d+[.)]\s*)"
    r"(.+?\?)\s*\n+"
    r"\s*(?:A[:.]?\s*|Answer[:.]?\s*)?"
    r"(.+?)(?=\n\s*(?:Q[:.]?\s*|Question[:.]?\s*|\d+[.)]\s*)|$)",
    re.S | re.I,
)
_POLICY_WORDS = {
    "policy", "policies", "regulation", "guideline", "rule",
    "code of conduct", "prohibited", "shall", "must not",
    "violation", "compliance",
}


# ============================================================================
# Low-level helpers
# ============================================================================

def _clean_whitespace(text: str) -> str:
    """Normalise whitespace: collapse horizontal spaces, limit blank lines."""
    text = unescape(text)
    text = re.sub(r"[^\S\n]+", " ", text)        # horizontal ws → single space
    text = re.sub(r"\n{3,}", "\n\n", text)        # 3+ newlines → 2
    lines = [l.strip() for l in text.split("\n")]
    return "\n".join(lines).strip()


def _strip_noise_text(text: str) -> str:
    """Remove universal boilerplate patterns from text."""
    for pat in _NOISE_TEXT_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


def _visible_text_length(element) -> int:
    """Approximate visible text length of a BS4 element."""
    return len(element.get_text(strip=True))


def _link_text_length(element) -> int:
    """Total length of anchor text inside an element."""
    return sum(len(a.get_text(strip=True)) for a in element.find_all("a"))


def _content_score(element) -> float:
    """Score an element by text density and low link-density.

    High score = lots of text content with few navigation links.
    """
    text_len = _visible_text_length(element)
    if text_len < 30:
        return 0.0
    link_len = _link_text_length(element)
    link_ratio = link_len / max(text_len, 1)
    # Penalise heavily if >50% of text is links (nav bars, footers)
    return text_len * max(0.0, 1.0 - link_ratio * 1.5)


# ============================================================================
# Table / list → text serialisation
# ============================================================================

def _table_to_text(table_tag) -> str:
    """Convert <table> to pipe-delimited text preserving rows/columns."""
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            cell_text = td.get_text(separator=" ", strip=True)
            cells.append(cell_text if cell_text else "-")
        if cells and any(c.strip() and c != "-" for c in cells):
            rows.append(" | ".join(cells))
    if not rows:
        return ""
    # Add separator after first row if it looks like a header
    if len(rows) > 1 and _looks_like_header(rows[0]):
        rows.insert(1, "-" * min(len(rows[0]), 60))
    return "\n".join(rows)


def _looks_like_header(row: str) -> bool:
    header_kw = [
        "name", "title", "description", "library", "location", "day",
        "time", "hours", "schedule", "service", "date", "type", "category",
        "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday",
    ]
    low = row.lower()
    return any(kw in low for kw in header_kw)


def _list_to_text(list_tag) -> str:
    """Convert <ul>/<ol> to readable bulleted/numbered text."""
    items = []
    ordered = list_tag.name == "ol"
    for i, li in enumerate(list_tag.find_all("li", recursive=False)):
        text = li.get_text(separator=" ", strip=True)
        if text:
            prefix = f"{i + 1}. " if ordered else "- "
            items.append(f"{prefix}{text}")
    return "\n".join(items)


def _dl_to_text(dl_tag) -> str:
    """Convert <dl> (definition list) to readable text."""
    parts = []
    current_term = ""
    for child in dl_tag.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "dt":
            current_term = child.get_text(strip=True)
        elif child.name == "dd":
            desc = child.get_text(separator=" ", strip=True)
            if current_term:
                parts.append(f"{current_term}: {desc}")
            else:
                parts.append(desc)
            current_term = ""
    return "\n".join(parts)


# ============================================================================
# HTML cleaning and main-content isolation
# ============================================================================

def _clean_html(html: str) -> Optional["BeautifulSoup"]:
    """Parse HTML and remove all noise elements.  Returns cleaned soup."""
    if not BeautifulSoup or not html:
        return None
    soup = BeautifulSoup(html, "html.parser")

    # Remove comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Remove non-content tags
    for tag_name in REMOVE_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()

    # Remove noise regions
    for selector in NOISE_SELECTORS:
        try:
            for el in soup.select(selector):
                el.decompose()
        except Exception:
            pass

    return soup


def _find_main_content(soup) -> "Tag":
    """Locate the main content container.

    Strategy:
      1. Try known semantic selectors (main, article, #content, …).
      2. Fall back to the highest-scoring container by content density.
      3. Last resort: use <body> itself.
    """
    root = soup.body or soup

    for selector in MAIN_CONTENT_SELECTORS:
        try:
            el = soup.select_one(selector)
        except Exception:
            continue
        if el and _visible_text_length(el) > 50:
            return el

    # Fallback: score direct children of body
    best, best_score = root, _content_score(root)
    for child in root.find_all(["div", "section", "article", "main"], recursive=False):
        score = _content_score(child)
        if score > best_score:
            best, best_score = child, score
    return best


# ============================================================================
# Layer 1 — Semantic extraction (headings, tables, lists, sections)
# ============================================================================

def _extract_accordion_blocks(soup) -> List[Dict]:
    """Extract content from accordion / details-summary / toggle patterns."""
    blocks: List[Dict] = []

    # Pattern A: .accordion-item with header + body
    for item in soup.select(
        ".accordion-item, .accordion, [data-toggle='collapse'], "
        ".panel, .card"
    ):
        header = item.select_one(
            ".accordion-header, .accordion-toggle, .panel-heading, "
            ".card-header, [data-toggle='collapse'], summary, "
            "[class*='toggle'], [class*='header']"
        )
        body = item.select_one(
            ".accordion-body, .accordion-content, .panel-body, "
            ".card-body, .panel-collapse, .collapse, "
            "[class*='content'], [class*='body']"
        )
        if header and body:
            h = header.get_text(strip=True)
            c = body.get_text(separator="\n", strip=True)
            if h and c and len(c) > 10:
                blocks.append({
                    "content": c,
                    "heading": h,
                    "heading_path": [h],
                    "content_format": "text",
                    "level": 3,
                    "block_type": "general",
                })

    # Pattern B: HTML5 <details> / <summary>
    for details in soup.find_all("details"):
        summary = details.find("summary")
        if not summary:
            continue
        h = summary.get_text(strip=True)
        summary.decompose()
        c = details.get_text(separator="\n", strip=True)
        if h and c and len(c) > 10:
            blocks.append({
                "content": c,
                "heading": h,
                "heading_path": [h],
                "content_format": "text",
                "level": 3,
                "block_type": "general",
            })

    return blocks


def _extract_tabbed_blocks(soup) -> List[Dict]:
    """Extract content from tabbed interfaces."""
    blocks: List[Dict] = []
    panels = soup.select("[role='tabpanel'], .tab-pane")
    labels = soup.select("[role='tab'], .nav-tabs .nav-link")
    if panels and len(labels) >= len(panels):
        for i, panel in enumerate(panels):
            label = labels[i].get_text(strip=True) if i < len(labels) else ""
            content = panel.get_text(separator="\n", strip=True)
            if content and len(content) > 10:
                blocks.append({
                    "content": content,
                    "heading": label,
                    "heading_path": [label] if label else [],
                    "content_format": "text",
                    "level": 3,
                    "block_type": "general",
                })
    return blocks


def _is_pseudo_heading(element) -> Optional[str]:
    """Detect <p><strong>text</strong></p> acting as a heading.

    Returns the heading text if detected, else None.
    """
    non_empty = [
        c for c in element.children
        if not (isinstance(c, NavigableString) and not c.strip())
    ]
    if (
        len(non_empty) == 1
        and isinstance(non_empty[0], Tag)
        and non_empty[0].name in ("strong", "b")
    ):
        text = non_empty[0].get_text(strip=True)
        if text and 3 < len(text) < 100:
            return text
    return None


def _semantic_extract(root) -> List[Dict]:
    """Walk DOM under *root* and build content blocks from HTML semantics.

    Handles headings (h1-h6), tables, lists, definition lists, and
    paragraph / div leaf content.  Maintains a heading-path stack so every
    block knows its full heading ancestry.
    """
    blocks: List[Dict] = []
    heading_stack: List[Tuple[int, str]] = []   # (level, text)
    current_heading = ""
    current_level = 0
    current_parts: List[str] = []

    def _heading_path() -> List[str]:
        return [h[1] for h in heading_stack]

    def _flush():
        nonlocal current_parts
        if current_parts:
            combined = "\n\n".join(p for p in current_parts if p.strip())
            if combined.strip() and len(combined.strip()) > 10:
                blocks.append({
                    "content": combined.strip(),
                    "heading": current_heading,
                    "heading_path": _heading_path(),
                    "content_format": "text",
                    "level": current_level,
                    "block_type": "general",
                })
            current_parts = []

    for element in root.descendants:
        if not isinstance(element, Tag):
            continue
        # Skip elements nested inside structures we handle atomically
        if element.parent and element.parent.name in _INLINE_PARENTS:
            continue

        tag = element.name

        # ── Headings ──
        if tag in _HEADING_TAGS:
            _flush()
            text = element.get_text(separator=" ", strip=True)
            if text and len(text) > 1:
                level = int(tag[1])
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
                current_heading = text
                current_level = level
            continue

        # ── Tables ──
        if tag == "table":
            table_text = _table_to_text(element)
            if table_text and len(table_text) > 10:
                _flush()
                blocks.append({
                    "content": table_text,
                    "heading": current_heading,
                    "heading_path": _heading_path(),
                    "content_format": "table",
                    "level": current_level,
                    "block_type": "general",
                })
            continue

        # ── Lists ──
        if tag in ("ul", "ol"):
            list_text = _list_to_text(element)
            if list_text and len(list_text) > 10:
                _flush()
                blocks.append({
                    "content": list_text,
                    "heading": current_heading,
                    "heading_path": _heading_path(),
                    "content_format": "list",
                    "level": current_level,
                    "block_type": "general",
                })
            continue

        # ── Definition lists ──
        if tag == "dl":
            dl_text = _dl_to_text(element)
            if dl_text and len(dl_text) > 10:
                _flush()
                blocks.append({
                    "content": dl_text,
                    "heading": current_heading,
                    "heading_path": _heading_path(),
                    "content_format": "list",
                    "level": current_level,
                    "block_type": "general",
                })
            continue

        # ── Paragraph / div / span (leaf content) ──
        if tag in ("p", "div", "span", "blockquote", "section",
                    "article", "figcaption"):
            # Only process leaf-level containers (no nested block children)
            has_block_child = any(
                isinstance(c, Tag) and c.name in _BLOCK_TAGS
                for c in element.children
            )
            if has_block_child:
                continue

            # Detect pseudo-headings (<p><b>Heading</b></p>)
            pseudo = _is_pseudo_heading(element)
            if pseudo:
                _flush()
                level = max(current_level, 3)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, pseudo))
                current_heading = pseudo
                current_level = level
                continue

            text = element.get_text(separator=" ", strip=True)
            if text and len(text) > 5:
                current_parts.append(text)

    _flush()
    return blocks


# ============================================================================
# Layer 2 — DOM block detection (repeated siblings, content density)
# ============================================================================

def _class_signature(el) -> Tuple:
    """Fingerprint an element's class list, ignoring instance-specific tokens."""
    classes = el.get("class", [])
    # Drop classes that look like unique IDs (contain long numbers / hex)
    generic = sorted(
        c for c in classes
        if not re.search(r"\d{4,}|[0-9a-f]{8,}", c, re.I)
    )
    return (el.name, tuple(generic))


def _child_structure(el) -> Tuple:
    """Describe the internal structure of an element (child tag names)."""
    return tuple(
        c.name for c in el.children
        if isinstance(c, Tag) and c.name not in REMOVE_TAGS
    )


def _detect_repeated_siblings(container) -> List[List["Tag"]]:
    """Find groups of ≥3 sibling elements that share the same structure.

    Returns list of groups (each group is a list of Tag elements).
    These represent repeated content units: cards, rows, directory entries, etc.
    """
    direct_children = [c for c in container.children if isinstance(c, Tag)]
    if len(direct_children) < 3:
        return []

    # Group by (tag+class signature, child-structure fingerprint)
    groups: Dict[Tuple, List] = {}
    for child in direct_children:
        sig = (_class_signature(child), _child_structure(child))
        groups.setdefault(sig, []).append(child)

    # Return all groups with ≥ 3 members and meaningful text
    result = []
    for group in groups.values():
        if len(group) < 3:
            continue
        # At least some members should have real content
        has_content = sum(1 for g in group if _visible_text_length(g) > 20)
        if has_content >= len(group) * 0.5:
            result.append(group)
    return result


def _extract_pseudo_table(container) -> Optional[str]:
    """Detect div-based tables (grid of aligned divs) and serialise to text.

    Looks for a container whose direct children each hold the same number of
    sub-children, arranged like table rows with cells.
    """
    rows = [c for c in container.children if isinstance(c, Tag)]
    if len(rows) < 2:
        return None

    # Check structural uniformity: each row has similar child count
    child_counts = [
        len([gc for gc in r.children if isinstance(gc, Tag)])
        for r in rows
    ]
    if not child_counts or child_counts[0] < 2:
        return None

    # At least 60% of rows should have the same child count
    mode_count = max(set(child_counts), key=child_counts.count)
    uniform = sum(1 for c in child_counts if c == mode_count)
    if uniform < len(rows) * 0.6:
        return None

    # Serialise as pipe-delimited table
    text_rows = []
    for row in rows:
        cells = [
            gc.get_text(separator=" ", strip=True)
            for gc in row.children if isinstance(gc, Tag)
        ]
        if any(c.strip() for c in cells):
            text_rows.append(" | ".join(c if c else "-" for c in cells))

    if len(text_rows) < 2:
        return None

    # Insert separator after possible header row
    if _looks_like_header(text_rows[0]):
        text_rows.insert(1, "-" * min(len(text_rows[0]), 60))
    return "\n".join(text_rows)


def _dom_block_extract(root, semantic_blocks: List[Dict]) -> List[Dict]:
    """Layer 2: detect meaningful blocks from DOM structure when Layer 1 is weak.

    Examines direct children of the content root for:
      • Repeated sibling groups (cards, rows, directory items)
      • Pseudo-tables built from nested divs
      • Content-dense containers missed by heading-based extraction
    """
    blocks: List[Dict] = []

    # Check for repeated siblings at root level
    for group in _detect_repeated_siblings(root):
        for item in group:
            text = item.get_text(separator="\n", strip=True)
            text = _clean_whitespace(text)
            if len(text) > 30:
                # Try to pull a heading from the first child element
                heading = ""
                first_child = next(
                    (c for c in item.children if isinstance(c, Tag)), None
                )
                if first_child:
                    candidate = first_child.get_text(strip=True)
                    if candidate and len(candidate) < 100:
                        heading = candidate
                blocks.append({
                    "content": text,
                    "heading": heading,
                    "heading_path": [heading] if heading else [],
                    "content_format": "text",
                    "level": 0,
                    "block_type": "general",
                })

    # Check children for pseudo-tables
    for child in root.find_all(["div", "section"], recursive=False):
        ptable = _extract_pseudo_table(child)
        if ptable:
            blocks.append({
                "content": ptable,
                "heading": "",
                "heading_path": [],
                "content_format": "table",
                "level": 0,
                "block_type": "general",
            })

    # If no repeated structures found, recurse one level deeper
    if not blocks:
        for child in root.find_all(["div", "section", "article"], recursive=False):
            for group in _detect_repeated_siblings(child):
                for item in group:
                    text = item.get_text(separator="\n", strip=True)
                    text = _clean_whitespace(text)
                    if len(text) > 30:
                        blocks.append({
                            "content": text,
                            "heading": "",
                            "heading_path": [],
                            "content_format": "text",
                            "level": 0,
                            "block_type": "general",
                        })

    # Finally, find large content-dense divs that semantic extraction missed
    semantic_text = {b["content"][:80] for b in semantic_blocks}
    for child in root.find_all(["div", "section"], recursive=False):
        score = _content_score(child)
        text = child.get_text(separator="\n", strip=True)
        text = _clean_whitespace(text)
        # Only add if this content isn't already covered and is substantial
        if score > 200 and len(text) > 100 and text[:80] not in semantic_text:
            blocks.append({
                "content": text,
                "heading": "",
                "heading_path": [],
                "content_format": "text",
                "level": 0,
                "block_type": "general",
            })

    return blocks


# ============================================================================
# Layer 3 — Text-pattern detection
# ============================================================================

def _has_schedule_pattern(text: str) -> bool:
    """Check whether text contains schedule-like day+time patterns."""
    times = len(_TIME_RANGE_RE.findall(text)) + len(_TIME_LOOSE_RE.findall(text))
    days = len(_DAY_RE.findall(text))
    return times >= 2 and days >= 1


def _preprocess_schedule_text(text: str) -> str:
    """Insert line breaks into a single-line schedule blob.

    When innerText collapses all whitespace, a schedule like:
      "Library A Monday-Friday: 8am-5pm Saturday: Closed Library B Monday..."
    becomes one long line.  This function detects day-name boundaries
    preceded by non-schedule text (entity names) and inserts newlines so
    the line-by-line parser can work.

    Generic: does not rely on specific location names.
    """
    # If text already has enough lines, leave it alone
    non_empty = [l for l in text.split("\n") if l.strip()]
    if len(non_empty) > 3:
        return text

    # Find every position where a day-name pattern starts
    day_positions = [m.start() for m in _DAY_RE.finditer(text)]
    if len(day_positions) < 2:
        return text

    # Walk through day positions.  If there's significant non-schedule text
    # (>8 chars) between the end of one time-block and the next day name,
    # that gap is likely an entity header — insert a newline before it.
    result = []
    prev_end = 0
    for pos in day_positions:
        # Look backward from day-name to find where the previous schedule ended
        # (last time pattern or "Closed")
        preceding = text[prev_end:pos]
        # Find the last time or "Closed" in the preceding segment
        time_ends = [
            m.end() for m in _TIME_RANGE_RE.finditer(preceding)
        ] + [
            m.end() for m in _TIME_LOOSE_RE.finditer(preceding)
        ]
        if time_ends:
            last_time_end = max(time_ends) + prev_end
            gap = text[last_time_end:pos].strip()
            if len(gap) > 8:
                # There's an entity name between the last time and this day
                result.append(text[prev_end:last_time_end])
                result.append("\n")
                result.append(text[last_time_end:pos])
                prev_end = pos
                continue
        result.append(text[prev_end:pos])
        prev_end = pos
    result.append(text[prev_end:])
    return "".join(result)


def _split_schedule_entities(text: str) -> List[Dict]:
    """Split a schedule blob into one block per entity/location.

    Generic algorithm — no hardcoded location names:
      1. Pre-process single-line blobs to insert line breaks at entity boundaries.
      2. Split text into lines.
      3. Lines that contain day/time info are schedule lines.
      4. Lines that are short, lack time info, and precede schedule lines
         are treated as entity headers.
      5. Group schedule lines under their nearest header.
    """
    text = _preprocess_schedule_text(text)
    lines = text.split("\n")
    entities: List[Dict] = []
    current_header = ""
    current_schedule: List[str] = []

    for line in lines:
        stripped = line.strip().replace("\u200B", "").strip()
        if not stripped:
            if current_schedule:
                entities.append({
                    "header": current_header,
                    "schedule": "\n".join(current_schedule),
                })
                current_schedule = []
            continue

        has_time = bool(
            _TIME_RANGE_RE.search(stripped) or _TIME_LOOSE_RE.search(stripped)
        )
        has_day = bool(_DAY_RE.search(stripped))

        if has_time or has_day:
            # Line is schedule data — but it might also start with an entity name
            # Detect "Entity Name Monday - Friday: 8am-5pm" pattern
            day_match = _DAY_RE.search(stripped)
            if day_match and day_match.start() > 10:
                # Significant text before the first day name → inline entity header
                pre = stripped[:day_match.start()].rstrip(": -–—")
                post = stripped[day_match.start():]
                if pre and len(pre) < 80:
                    if current_schedule:
                        entities.append({
                            "header": current_header,
                            "schedule": "\n".join(current_schedule),
                        })
                        current_schedule = []
                    current_header = pre
                    current_schedule.append(post)
                    continue
            current_schedule.append(stripped)
        elif len(stripped) < 80:
            # Short line without time → could be an entity header OR a
            # date-specific closure note (e.g. "March 30:").
            # Closure notes are short date references that should stay with
            # the current schedule, not start a new entity.
            is_date_note = bool(re.match(
                r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*"
                r"\s+\d{1,2}\b",
                stripped, re.I,
            ))
            if is_date_note and current_schedule:
                # Fold date-specific closure into current schedule
                current_schedule.append(stripped)
            else:
                # New entity header
                if current_schedule:
                    entities.append({
                        "header": current_header,
                        "schedule": "\n".join(current_schedule),
                    })
                    current_schedule = []
                current_header = stripped.rstrip(":")
        else:
            # Long line without time — add as note
            current_schedule.append(stripped)

    if current_schedule:
        entities.append({
            "header": current_header,
            "schedule": "\n".join(current_schedule),
        })

    if len(entities) <= 1:
        return []   # No useful split — caller keeps the original block

    # If none of the entities have a header, the split isn't meaningful —
    # these are just day/time lines belonging to one unnamed entity.
    # Return empty so the caller keeps the block as-is (with its heading).
    if not any(e["header"] for e in entities):
        return []

    return [
        {
            "content": (
                f"{e['header']}\n{e['schedule']}" if e["header"]
                else e["schedule"]
            ),
            "heading": e["header"],
            "heading_path": [e["header"]] if e["header"] else [],
            "content_format": "text",
            "level": 0,
            "block_type": "schedule",
        }
        for e in entities
        if e["schedule"].strip()
    ]


def _extract_faq_pairs(text: str) -> List[Dict]:
    """Extract Q&A pairs from text using generic patterns."""
    pairs = []

    # Strategy 1: Explicit Q/A markers
    for match in _QUESTION_RE.finditer(text):
        q = match.group(1).strip()
        a = match.group(2).strip()
        if len(q) > 10 and len(a) > 10:
            pairs.append({"q": q, "a": a})
    if pairs:
        return pairs

    # Strategy 2: Bold questions (**question?** answer)
    for m in re.finditer(
        r"(?:\*\*|__)(.*?\?)(?:\*\*|__)\s*\n+(.+?)(?=(?:\*\*|__)|$)",
        text, re.S,
    ):
        q, a = m.group(1).strip(), m.group(2).strip()
        if len(q) > 10 and len(a) > 10:
            pairs.append({"q": q, "a": a})
    if pairs:
        return pairs

    # Strategy 3: Sentences ending in '?' followed by non-question text
    segments = re.split(r"(\?)", text)
    i = 0
    while i < len(segments) - 1:
        if segments[i + 1] == "?":
            question = segments[i].strip()
            nl = question.rfind("\n")
            if nl > 0:
                question = question[nl:].strip()
            question += "?"
            answer = segments[i + 2].strip() if i + 2 < len(segments) else ""
            nl = answer.find("\n\n")
            if nl > 0:
                answer = answer[:nl].strip()
            if len(question) > 15 and len(answer) > 15:
                pairs.append({"q": question, "a": answer})
            i += 2
        else:
            i += 1

    return pairs


def _detect_patterns(blocks: List[Dict]) -> List[Dict]:
    """Layer 3: detect and refine blocks based on text patterns.

    May sub-split blocks (e.g. a schedule blob → per-entity blocks)
    and always sets block_type based on detected content.
    """
    result: List[Dict] = []

    for block in blocks:
        text = block["content"]
        heading = block.get("heading", "").lower()

        # ── Schedule detection ──
        if _has_schedule_pattern(text):
            # Tables are already structured — don't try to re-split them.
            # Just tag the block as schedule and keep it intact.
            if block.get("content_format") == "table":
                block["block_type"] = "schedule"
                result.append(block)
                continue
            sub = _split_schedule_entities(text)
            if sub:
                result.extend(sub)
                continue
            block["block_type"] = "schedule"
            result.append(block)
            continue

        # ── FAQ detection ──
        q_count = text.count("?")
        if q_count >= 2 or "faq" in heading or "frequently" in heading:
            pairs = _extract_faq_pairs(text)
            if pairs:
                for p in pairs:
                    result.append({
                        "content": f"Q: {p['q']}\nA: {p['a']}",
                        "heading": block.get("heading", ""),
                        "heading_path": block.get("heading_path", []),
                        "content_format": "faq_pair",
                        "level": block.get("level", 0),
                        "block_type": "faq",
                    })
                continue

        # ── Contact detection ──
        emails = _EMAIL_RE.findall(text)
        phones = _PHONE_RE.findall(text)
        if len(emails) + len(phones) >= 2 or "contact" in heading:
            block["block_type"] = "contact"
            result.append(block)
            continue

        # ── Policy detection ──
        low = text.lower()
        policy_hits = sum(1 for w in _POLICY_WORDS if w in low)
        if policy_hits >= 2 or any(w in heading for w in _POLICY_WORDS):
            block["block_type"] = "policy"
            result.append(block)
            continue

        # ── No special pattern ──
        result.append(block)

    return result


# ============================================================================
# Layer 4 — Fallback paragraph splitting
# ============================================================================

def _fallback_paragraph_blocks(text: str, target_size: int = 600) -> List[Dict]:
    """Split plain text into blocks at paragraph boundaries.

    Consecutive short paragraphs are merged so each block reaches a
    reasonable size for embedding (avoiding tiny fragments).
    """
    text = _strip_noise_text(text)
    text = _clean_whitespace(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    blocks: List[Dict] = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > target_size:
            if len(current) > 30:
                blocks.append({
                    "content": current,
                    "heading": "",
                    "heading_path": [],
                    "content_format": "text",
                    "level": 0,
                    "block_type": "general",
                })
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current and len(current) > 30:
        blocks.append({
            "content": current,
            "heading": "",
            "heading_path": [],
            "content_format": "text",
            "level": 0,
            "block_type": "general",
        })
    return blocks


# ============================================================================
# Classification
# ============================================================================

def _classify_block(block: Dict) -> str:
    """Classify a single block's type from content patterns + heading."""
    # Respect pre-assigned type from pattern detection
    existing = block.get("block_type", "general")
    if existing != "general":
        return existing

    text = block["content"].lower()
    heading = block.get("heading", "").lower()

    if _has_schedule_pattern(block["content"]):
        return "schedule"
    if block["content"].count("?") >= 2 or "faq" in heading:
        return "faq"
    emails = _EMAIL_RE.findall(block["content"])
    phones = _PHONE_RE.findall(block["content"])
    if len(emails) + len(phones) >= 2 or "contact" in heading:
        return "contact"
    policy_hits = sum(1 for w in _POLICY_WORDS if w in text)
    if policy_hits >= 2 or any(w in heading for w in _POLICY_WORDS):
        return "policy"
    return "general"


def _classify_page(blocks: List[Dict], url: str, title: str) -> str:
    """Determine the dominant page type."""
    # Count block types (weighted by content length)
    type_weight: Dict[str, float] = {}
    for b in blocks:
        t = b.get("block_type", "general")
        type_weight[t] = type_weight.get(t, 0) + len(b.get("content", ""))

    # Pick the heaviest non-general type
    non_general = {k: v for k, v in type_weight.items() if k != "general"}
    if non_general:
        best = max(non_general, key=non_general.get)
        # Only override "general" if the typed blocks are significant
        total = sum(type_weight.values())
        if non_general[best] > total * 0.25:
            return best

    # URL / title keyword fallback
    combined = (url + " " + title).lower()
    if any(kw in combined for kw in ["faq", "frequently", "question"]):
        return "faq"
    if any(kw in combined for kw in [
        "hour", "schedule", "timing", "opening", "closing",
        "contact", "direction", "location",
    ]):
        return "hours_contact"
    if any(kw in combined for kw in ["database", "eresource", "e-resource"]):
        return "database_page"
    if any(kw in combined for kw in ["polic", "regulation", "guideline", "rule"]):
        return "policy"
    if any(kw in combined for kw in [
        "service", "borrowing", "interlibrary", "reserve", "loan",
        "printing", "circulation",
    ]):
        return "service"
    if any(kw in combined for kw in ["about", "mission", "history"]):
        return "about"
    return "general"


# ============================================================================
# Extraction quality assessment
# ============================================================================

def _is_weak_extraction(
    blocks: List[Dict], root, min_blocks: int = 2, coverage: float = 0.3
) -> bool:
    """Return True if semantic extraction looks incomplete.

    Checks:
      • Too few blocks found
      • Blocks cover less than `coverage` of the root's visible text
      • Single giant block (blob)
    """
    if not blocks:
        return True
    if len(blocks) < min_blocks:
        return True

    root_len = _visible_text_length(root)
    if root_len == 0:
        return True

    blocks_len = sum(len(b["content"]) for b in blocks)
    if blocks_len / root_len < coverage:
        return True

    # Giant-blob check: one block has > 80% of all extracted content
    if len(blocks) > 1:
        max_block = max(len(b["content"]) for b in blocks)
        if max_block > blocks_len * 0.8:
            return True

    return False


# ============================================================================
# Block quality filter
# ============================================================================

def _is_useful_block(block: Dict) -> bool:
    """Return False for blocks that are noise / navigation remnants.

    Heuristics (all generic, no site-specific strings):
      • Too short (< 25 chars of real text)
      • Link-heavy (> 60% of words are typical nav link text)
      • Pure menu / breadcrumb patterns
    """
    content = block.get("content", "").strip()
    if len(content) < 25:
        return False

    words = content.split()
    if len(words) < 4:
        return False

    # High link-density heuristic: if the text is mostly short fragments
    # separated by pipes/bullets, it's probably a navigation bar.
    # e.g. "Home | About | Services | Contact | FAQ"
    pipe_segments = re.split(r"\s*[|•·]\s*", content)
    if len(pipe_segments) > 4 and all(len(s.split()) <= 3 for s in pipe_segments):
        return False

    # List where every item is ≤ 3 words is probably a nav menu
    if content.startswith("- "):
        items = [l for l in content.split("\n") if l.strip().startswith("- ")]
        if len(items) > 4 and all(len(it.split()) <= 4 for it in items):
            return False

    return True


# ============================================================================
# Public API
# ============================================================================

def extract_page(
    url: str,
    title: str,
    html: str = "",
    text: str = "",
    extra_noise_selectors: List[str] = None,
) -> Dict:
    """Extract structured content blocks from a single page.

    Parameters
    ----------
    url   : Page URL (used for classification, stored in output).
    title : Page title.
    html  : Raw innerHTML.  Preferred source for extraction.
    text  : Raw innerText.  Used as fallback when HTML is unavailable or weak.
    extra_noise_selectors : Additional CSS selectors to strip (site-specific
                            config passed in by the caller, not hard-coded).

    Returns
    -------
    dict with keys: url, page_title, page_type, blocks (list of block dicts),
    metadata.
    """
    blocks: List[Dict] = []

    # ── HTML-based extraction ──
    if html:
        soup = _clean_html(html)
        if soup:
            # Apply any caller-supplied extra noise selectors
            if extra_noise_selectors:
                for sel in extra_noise_selectors:
                    try:
                        for el in soup.select(sel):
                            el.decompose()
                    except Exception:
                        pass

            root = _find_main_content(soup)

            # Extract accordion / tabbed content first (they often hold
            # the real content on SharePoint / CMS sites)
            accordion_blocks = _extract_accordion_blocks(root)
            tabbed_blocks = _extract_tabbed_blocks(root)

            # Layer 1: Semantic extraction
            blocks = _semantic_extract(root)

            # Merge accordion/tab content not already covered
            covered_headings = {b["heading"].lower() for b in blocks if b["heading"]}
            for ab in accordion_blocks + tabbed_blocks:
                if ab["heading"].lower() not in covered_headings:
                    blocks.append(ab)

            # Layer 2: DOM block detection when semantic is weak
            if _is_weak_extraction(blocks, root):
                dom_blocks = _dom_block_extract(root, blocks)
                if dom_blocks:
                    # If DOM found more / richer structure, prefer it
                    dom_len = sum(len(b["content"]) for b in dom_blocks)
                    sem_len = sum(len(b["content"]) for b in blocks)
                    if dom_len > sem_len * 1.3 or len(dom_blocks) > len(blocks) * 2:
                        blocks = dom_blocks
                    else:
                        # Merge non-duplicate DOM blocks
                        existing = {b["content"][:80] for b in blocks}
                        for db in dom_blocks:
                            if db["content"][:80] not in existing:
                                blocks.append(db)

    # ── Text fallback ──
    if not blocks and text:
        clean_text = _strip_noise_text(text)
        clean_text = _clean_whitespace(clean_text)
        if len(clean_text.split()) >= 20:
            blocks = _fallback_paragraph_blocks(clean_text)

    # ── Layer 3: pattern detection on every block ──
    blocks = _detect_patterns(blocks)

    # ── Final classification ──
    for block in blocks:
        block["block_type"] = _classify_block(block)

    # ── Quality filter: drop blocks that are noise remnants ──
    blocks = [b for b in blocks if _is_useful_block(b)]

    page_type = _classify_page(blocks, url, title)
    word_count = sum(len(b["content"].split()) for b in blocks)

    # Clean page title
    clean_title = re.sub(r"<[^>]+>", "", title).strip()
    clean_title = re.sub(r"\s+", " ", clean_title).strip() or title.strip()

    return {
        "url": url,
        "page_title": clean_title,
        "page_type": page_type,
        "blocks": blocks,
        "metadata": {
            "word_count": word_count,
            "block_count": len(blocks),
            "has_tables": any(b["content_format"] == "table" for b in blocks),
            "has_lists": any(b["content_format"] == "list" for b in blocks),
            "has_faq": any(b["block_type"] == "faq" for b in blocks),
            "has_schedule": any(b["block_type"] == "schedule" for b in blocks),
        },
    }


def extract_pages_batch(
    scraped_data: List[Dict],
    extra_noise_selectors: List[str] = None,
) -> List[Dict]:
    """Process a batch of scraped pages into structured extracted pages.

    Input:  [{"url": ..., "title": ..., "content": ..., "html": ...}, ...]
    Output: [extracted_page_dict, ...]  (pages with no blocks are filtered out)
    """
    results = []
    for item in scraped_data:
        page = extract_page(
            url=item.get("url", ""),
            title=item.get("title", ""),
            html=item.get("html", ""),
            text=item.get("content", ""),
            extra_noise_selectors=extra_noise_selectors,
        )
        if page["blocks"]:
            results.append(page)

    type_counts: Dict[str, int] = {}
    for p in results:
        type_counts[p["page_type"]] = type_counts.get(p["page_type"], 0) + 1

    logger.info(
        f"Extracted {len(results)}/{len(scraped_data)} pages. "
        f"Types: {type_counts}"
    )
    return results
