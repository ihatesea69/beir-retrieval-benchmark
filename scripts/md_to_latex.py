"""
Markdown to LaTeX Converter

Converts Markdown files to LaTeX format, preserving content structure
(headings, lists, tables, code blocks, math formulas) in valid LaTeX syntax.

Usage:
    python scripts/md_to_latex.py <input.md> <output.tex>
    python scripts/md_to_latex.py --batch <input_dir> <output_dir>
"""

import re
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal Representation
# ---------------------------------------------------------------------------

@dataclass
class MarkdownElement:
    """Internal representation of a parsed Markdown element."""
    type: str  # 'heading', 'paragraph', 'code_block', 'list', 'table', 'math_block'
    content: str = ''
    level: int = 0        # For headings (1-4)
    language: str = ''    # For code blocks
    children: List['MarkdownElement'] = field(default_factory=list)
    ordered: bool = False  # For lists


# ---------------------------------------------------------------------------
# LaTeX special-character escaping
# ---------------------------------------------------------------------------

_LATEX_SPECIAL = re.compile(r'([&%$#_{}~^\\])')

_LATEX_ESCAPE_MAP = {
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
    '\\': r'\textbackslash{}',
}


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    return _LATEX_SPECIAL.sub(lambda m: _LATEX_ESCAPE_MAP[m.group(1)], text)


# ---------------------------------------------------------------------------
# Emoji stripping
# ---------------------------------------------------------------------------

# Covers most common emoji ranges
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"              # zero width joiner
    "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    """Remove emoji characters from text."""
    return _EMOJI_RE.sub('', text)


# ---------------------------------------------------------------------------
# Inline formatting conversion
# ---------------------------------------------------------------------------

def convert_inline(text: str) -> str:
    """Convert inline Markdown formatting to LaTeX.

    Handles: bold, italic, inline code, links, images, math, badges.
    LaTeX special chars in *plain text* are escaped; content inside
    math delimiters or code spans is left untouched.
    """
    text = strip_emoji(text)

    # Remove badge images [![...](...)
    text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)
    # Remove standalone images ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    parts: list[str] = []
    pos = 0

    while pos < len(text):
        # --- inline math $...$ (not $$) ---
        m = re.match(r'\$([^$]+)\$', text[pos:])
        if m and (pos == 0 or text[pos - 1] != '$'):
            parts.append(f'${m.group(1)}$')
            pos += m.end()
            continue

        # --- inline code `...` ---
        m = re.match(r'`([^`]+)`', text[pos:])
        if m:
            parts.append(r'\texttt{' + escape_latex(m.group(1)) + '}')
            pos += m.end()
            continue

        # --- bold **...** ---
        m = re.match(r'\*\*(.+?)\*\*', text[pos:])
        if m:
            parts.append(r'\textbf{' + convert_inline(m.group(1)) + '}')
            pos += m.end()
            continue

        # --- italic *...* ---
        m = re.match(r'\*(.+?)\*', text[pos:])
        if m:
            parts.append(r'\textit{' + convert_inline(m.group(1)) + '}')
            pos += m.end()
            continue

        # --- link [text](url) ---
        m = re.match(r'\[([^\]]+)\]\(([^)]+)\)', text[pos:])
        if m:
            link_text = convert_inline(m.group(1))
            url = m.group(2)
            parts.append(r'\href{' + url + '}{' + link_text + '}')
            pos += m.end()
            continue

        # --- plain character (escape if special) ---
        ch = text[pos]
        if ch in _LATEX_ESCAPE_MAP:
            parts.append(_LATEX_ESCAPE_MAP[ch])
        else:
            parts.append(ch)
        pos += 1

    return ''.join(parts)


# ---------------------------------------------------------------------------
# Markdown Parser  →  List[MarkdownElement]
# ---------------------------------------------------------------------------

def parse_markdown(text: str) -> List[MarkdownElement]:
    """Parse a Markdown string into a list of MarkdownElement objects."""
    elements: List[MarkdownElement] = []
    lines = text.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # --- fenced code block / mermaid ---
        m = re.match(r'^```(\w*)', line)
        if m:
            lang = m.group(1)
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            block_type = 'mermaid' if lang.lower() == 'mermaid' else 'code_block'
            elements.append(MarkdownElement(
                type=block_type,
                content='\n'.join(code_lines),
                language=lang,
            ))
            continue

        # --- math block $$...$$ ---
        if line.strip().startswith('$$'):
            math_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('$$'):
                math_lines.append(lines[i])
                i += 1
            i += 1  # skip closing $$
            elements.append(MarkdownElement(type='math_block', content='\n'.join(math_lines)))
            continue

        # --- heading ---
        m = re.match(r'^(#{1,4})\s+(.*)', line)
        if m:
            level = len(m.group(1))
            elements.append(MarkdownElement(type='heading', content=m.group(2).strip(), level=level))
            i += 1
            continue

        # --- horizontal rule ---
        if re.match(r'^-{3,}$', line.strip()) or re.match(r'^\*{3,}$', line.strip()):
            i += 1
            continue

        # --- table ---
        if '|' in line and i + 1 < len(lines) and re.match(r'^[\s|:-]+$', lines[i + 1]):
            table_lines: list[str] = []
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            elements.append(_parse_table(table_lines))
            continue

        # --- unordered list ---
        if re.match(r'^(\s*)[-*+]\s', line):
            list_lines: list[str] = []
            while i < len(lines) and re.match(r'^(\s*)[-*+]\s', lines[i]):
                list_lines.append(lines[i])
                i += 1
            elements.append(_parse_list(list_lines, ordered=False))
            continue

        # --- ordered list ---
        if re.match(r'^(\s*)\d+\.\s', line):
            list_lines = []
            while i < len(lines) and re.match(r'^(\s*)\d+\.\s', lines[i]):
                list_lines.append(lines[i])
                i += 1
            elements.append(_parse_list(list_lines, ordered=True))
            continue

        # --- blank line → skip ---
        if line.strip() == '':
            i += 1
            continue

        # --- paragraph (collect consecutive non-blank lines) ---
        para_lines: list[str] = []
        while i < len(lines) and lines[i].strip() != '' and not _is_block_start(lines, i):
            para_lines.append(lines[i])
            i += 1
        if para_lines:
            elements.append(MarkdownElement(type='paragraph', content='\n'.join(para_lines)))

    return elements


def _is_block_start(lines: list[str], i: int) -> bool:
    """Return True if lines[i] starts a new block element."""
    line = lines[i]
    if re.match(r'^```', line):
        return True
    if re.match(r'^#{1,4}\s', line):
        return True
    if re.match(r'^(\s*)[-*+]\s', line):
        return True
    if re.match(r'^(\s*)\d+\.\s', line):
        return True
    if re.match(r'^-{3,}$', line.strip()):
        return True
    if '|' in line and i + 1 < len(lines) and re.match(r'^[\s|:-]+$', lines[i + 1]):
        return True
    if line.strip().startswith('$$'):
        return True
    return False


def _parse_table(lines: list[str]) -> MarkdownElement:
    """Parse Markdown table lines into a MarkdownElement."""
    rows: list[str] = []
    for idx, line in enumerate(lines):
        # skip separator row
        if idx == 1 and re.match(r'^[\s|:-]+$', line):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        rows.append('|'.join(cells))
    return MarkdownElement(type='table', content='\n'.join(rows))


def _parse_list(lines: list[str], ordered: bool) -> MarkdownElement:
    """Parse list lines into a MarkdownElement with children."""
    children: list[MarkdownElement] = []
    for line in lines:
        if ordered:
            m = re.match(r'^\s*\d+\.\s+(.*)', line)
        else:
            m = re.match(r'^\s*[-*+]\s+(.*)', line)
        if m:
            children.append(MarkdownElement(type='list_item', content=m.group(1)))
    return MarkdownElement(type='list', children=children, ordered=ordered)


# ---------------------------------------------------------------------------
# LaTeX Generation  (MarkdownElement → LaTeX string)
# ---------------------------------------------------------------------------

_HEADING_MAP = {
    1: 'section',
    2: 'subsection',
    3: 'subsubsection',
    4: 'paragraph',
}


def element_to_latex(el: MarkdownElement) -> str:
    """Convert a single MarkdownElement to a LaTeX string."""
    if el.type == 'heading':
        cmd = _HEADING_MAP.get(el.level, 'paragraph')
        return f'\\{cmd}{{{convert_inline(el.content)}}}'

    if el.type == 'paragraph':
        return convert_inline(el.content)

    if el.type == 'code_block':
        lang_opt = f'[language={el.language}]' if el.language else ''
        return (
            f'\\begin{{lstlisting}}{lang_opt}\n'
            f'{el.content}\n'
            f'\\end{{lstlisting}}'
        )

    if el.type == 'mermaid':
        commented = '\n'.join(f'% {l}' for l in el.content.split('\n'))
        return f'% Mermaid diagram (not renderable in LaTeX)\n{commented}'

    if el.type == 'math_block':
        return f'\\[\n{el.content}\n\\]'

    if el.type == 'list':
        env = 'enumerate' if el.ordered else 'itemize'
        items = '\n'.join(f'  \\item {convert_inline(c.content)}' for c in el.children)
        return f'\\begin{{{env}}}\n{items}\n\\end{{{env}}}'

    if el.type == 'table':
        return _table_to_latex(el)

    # fallback
    return convert_inline(el.content)


def _table_to_latex(el: MarkdownElement) -> str:
    """Convert a table MarkdownElement to LaTeX tabular."""
    rows = el.content.split('\n')
    if not rows:
        return ''
    first_row_cells = rows[0].split('|')
    ncols = len(first_row_cells)
    col_spec = '|' + '|'.join(['l'] * ncols) + '|'

    latex_rows: list[str] = []
    for idx, row in enumerate(rows):
        cells = row.split('|')
        converted = ' & '.join(convert_inline(c.strip()) for c in cells)
        latex_rows.append(f'  {converted} \\\\')
        if idx == 0:
            latex_rows.append('  \\hline')

    return (
        f'\\begin{{tabular}}{{{col_spec}}}\n'
        f'  \\hline\n'
        + '\n'.join(latex_rows) + '\n'
        f'  \\hline\n'
        f'\\end{{tabular}}'
    )


# ---------------------------------------------------------------------------
# Full document conversion
# ---------------------------------------------------------------------------

def elements_to_latex(elements: List[MarkdownElement]) -> str:
    """Convert a list of MarkdownElements to a full LaTeX string."""
    parts = [element_to_latex(el) for el in elements]
    return '\n\n'.join(parts)


def convert_markdown_to_latex(md_text: str) -> str:
    """High-level: Markdown text → LaTeX text."""
    elements = parse_markdown(md_text)
    return elements_to_latex(elements)


# ---------------------------------------------------------------------------
# LaTeX Pretty-Printer  (MarkdownElement → LaTeX text, for round-trip)
# ---------------------------------------------------------------------------

def pretty_print_latex(elements: List[MarkdownElement]) -> str:
    """Serialize internal representation back to LaTeX text.

    This is the pretty-printer required for round-trip validation.
    It produces the same output as elements_to_latex.
    """
    return elements_to_latex(elements)


# ---------------------------------------------------------------------------
# LaTeX Parser  (LaTeX text → List[MarkdownElement], for round-trip)
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r'^\\(section|subsection|subsubsection|paragraph)\{(.*)\}$'
)
_LEVEL_MAP = {
    'section': 1,
    'subsection': 2,
    'subsubsection': 3,
    'paragraph': 4,
}


def parse_latex(latex_text: str) -> List[MarkdownElement]:
    """Parse LaTeX text (produced by this converter) back into MarkdownElements.

    This enables round-trip testing: elements → LaTeX → elements.
    """
    elements: List[MarkdownElement] = []
    lines = latex_text.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # --- lstlisting code block ---
        m = re.match(r'^\\begin\{lstlisting\}(?:\[language=(\w+)\])?$', line)
        if m:
            lang = m.group(1) or ''
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i] != '\\end{lstlisting}':
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip \end{lstlisting}
            elements.append(MarkdownElement(type='code_block', content='\n'.join(code_lines), language=lang))
            continue

        # --- mermaid comment block ---
        if line.startswith('% Mermaid diagram'):
            mermaid_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].startswith('% '):
                mermaid_lines.append(lines[i][2:])  # strip '% '
                i += 1
            elements.append(MarkdownElement(type='mermaid', content='\n'.join(mermaid_lines)))
            continue

        # --- math block \[...\] ---
        if line.strip() == '\\[':
            math_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != '\\]':
                math_lines.append(lines[i])
                i += 1
            i += 1  # skip \]
            elements.append(MarkdownElement(type='math_block', content='\n'.join(math_lines)))
            continue

        # --- heading ---
        m = _SECTION_RE.match(line)
        if m:
            cmd = m.group(1)
            content = m.group(2)
            elements.append(MarkdownElement(type='heading', content=content, level=_LEVEL_MAP[cmd]))
            i += 1
            continue

        # --- itemize / enumerate ---
        m_env = re.match(r'^\\begin\{(itemize|enumerate)\}$', line)
        if m_env:
            env = m_env.group(1)
            ordered = env == 'enumerate'
            children: list[MarkdownElement] = []
            i += 1
            while i < len(lines) and not lines[i].startswith(f'\\end{{{env}}}'):
                item_m = re.match(r'^\s*\\item\s+(.*)', lines[i])
                if item_m:
                    children.append(MarkdownElement(type='list_item', content=item_m.group(1)))
                i += 1
            i += 1  # skip \end
            elements.append(MarkdownElement(type='list', children=children, ordered=ordered))
            continue

        # --- table ---
        if line.startswith('\\begin{tabular}'):
            table_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith('\\end{tabular}'):
                stripped = lines[i].strip()
                if stripped == '\\hline':
                    i += 1
                    continue
                if stripped.endswith('\\\\'):
                    stripped = stripped[:-2].strip()
                cells = [c.strip() for c in stripped.split('&')]
                table_lines.append('|'.join(cells))
                i += 1
            i += 1  # skip \end{tabular}
            elements.append(MarkdownElement(type='table', content='\n'.join(table_lines)))
            continue

        # --- blank line ---
        if line.strip() == '':
            i += 1
            continue

        # --- paragraph (collect consecutive non-blank, non-command lines) ---
        para_lines: list[str] = []
        while i < len(lines) and lines[i].strip() != '' and not _is_latex_block_start(lines[i]):
            para_lines.append(lines[i])
            i += 1
        if para_lines:
            elements.append(MarkdownElement(type='paragraph', content='\n'.join(para_lines)))

    return elements


def _is_latex_block_start(line: str) -> bool:
    """Check if a line starts a LaTeX block element."""
    if _SECTION_RE.match(line):
        return True
    if re.match(r'^\\begin\{', line):
        return True
    if line.startswith('% Mermaid diagram'):
        return True
    if line.strip() == '\\[':
        return True
    return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def convert_file(input_path: str, output_path: str) -> None:
    """Convert a single Markdown file to LaTeX."""
    with open(input_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
    latex_text = convert_markdown_to_latex(md_text)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_text)
    print(f"Converted: {input_path} -> {output_path}")


def batch_convert(input_dir: str, output_dir: str) -> None:
    """Convert all .md files in input_dir to .tex files in output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for md_file in sorted(input_path.glob('*.md')):
        tex_file = output_path / (md_file.stem + '.tex')
        convert_file(str(md_file), str(tex_file))


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python md_to_latex.py <input.md> <output.tex>")
        print("  python md_to_latex.py --batch <input_dir> <output_dir>")
        sys.exit(1)

    if sys.argv[1] == '--batch':
        if len(sys.argv) < 4:
            print("Usage: python md_to_latex.py --batch <input_dir> <output_dir>")
            sys.exit(1)
        batch_convert(sys.argv[2], sys.argv[3])
    else:
        convert_file(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
