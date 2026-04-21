"""
document_parser.py
Parses .docx Word files into clean plain text, preserving headings,
paragraphs, and table content so the chunker can process it properly.
"""

import io
import logging

logger = logging.getLogger(__name__)


def parse_docx(file_bytes: bytes) -> str:
    """Parse a .docx file from raw bytes and return clean plain text.

    Preserves:
      - Headings (prefixed with # for emphasis)
      - Paragraphs (separated by blank lines)
      - Table cells (tab-separated, rows separated by newlines)

    Returns the extracted text as a single string.
    """
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx is required: pip install python-docx>=1.1.0")

    doc = Document(io.BytesIO(file_bytes))
    parts = []

    # Track all block-level elements in document order
    # docx stores body elements as paragraphs and tables
    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            # Paragraph or heading
            from docx.oxml.ns import qn
            style_name = ""
            style_el = element.find(f".//{qn('w:pStyle')}")
            if style_el is not None:
                style_name = style_el.get(qn("w:val"), "")

            # Collect all text runs
            text = "".join(
                node.text or ""
                for node in element.iter()
                if node.tag == qn("w:t")
            ).strip()

            if not text:
                continue

            if "Heading" in style_name or "heading" in style_name.lower() or "Title" in style_name:
                parts.append(f"\n{text}\n")
            else:
                parts.append(text)

        elif tag == "tbl":
            # Table — extract each row as tab-separated cells
            from docx.oxml.ns import qn
            rows = element.findall(f".//{qn('w:tr')}")
            for row in rows:
                cells = row.findall(f".//{qn('w:tc')}")
                cell_texts = []
                for cell in cells:
                    cell_text = "".join(
                        node.text or ""
                        for node in cell.iter()
                        if node.tag == qn("w:t")
                    ).strip()
                    cell_texts.append(cell_text)
                row_text = "\t".join(cell_texts)
                if row_text.strip():
                    parts.append(row_text)

    return "\n\n".join(parts).strip()
