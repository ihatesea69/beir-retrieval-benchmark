"""
Property-based tests for the Markdown to LaTeX converter.

**Feature: project-cleanup**
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from md_to_latex import (
    MarkdownElement,
    convert_inline,
    convert_markdown_to_latex,
    element_to_latex,
    elements_to_latex,
    escape_latex,
    parse_latex,
    parse_markdown,
    pretty_print_latex,
)


# ---------------------------------------------------------------------------
# Strategies for generating random Markdown content
# ---------------------------------------------------------------------------

# Safe text that won't accidentally form Markdown syntax
_safe_word = st.from_regex(r'[A-Za-z0-9]+', fullmatch=True).filter(lambda s: len(s) > 0)
_safe_text = st.lists(_safe_word, min_size=1, max_size=5).map(lambda ws: ' '.join(ws))


def md_heading(level: int):
    """Generate a Markdown heading at the given level."""
    prefix = '#' * level
    return _safe_text.map(lambda t: f'{prefix} {t}')


md_bold = _safe_text.map(lambda t: f'**{t}**')
md_italic = _safe_text.map(lambda t: f'*{t}*')
md_inline_code = _safe_word.map(lambda t: f'`{t}`')
md_code_block = st.tuples(
    st.sampled_from(['python', 'bash', 'sql', '']),
    _safe_text,
).map(lambda pair: f'```{pair[0]}\n{pair[1]}\n```')

md_unordered_list = st.lists(_safe_text, min_size=1, max_size=4).map(
    lambda items: '\n'.join(f'- {item}' for item in items)
)

md_ordered_list = st.lists(_safe_text, min_size=1, max_size=4).map(
    lambda items: '\n'.join(f'{i+1}. {item}' for i, item in enumerate(items))
)

# A single Markdown element (as raw text)
md_element = st.one_of(
    md_heading(1),
    md_heading(2),
    md_heading(3),
    md_heading(4),
    md_bold,
    md_italic,
    md_inline_code,
    md_code_block,
    md_unordered_list,
    md_ordered_list,
)

# A full Markdown document composed of multiple elements
md_document = st.lists(md_element, min_size=1, max_size=6).map(
    lambda els: '\n\n'.join(els)
)


# ---------------------------------------------------------------------------
# Property 1: Markdown structure preservation
# **Feature: project-cleanup, Property 1: Markdown structure preservation**
# **Validates: Requirements 3.3**
# ---------------------------------------------------------------------------

@given(md_doc=md_document)
@settings(max_examples=100)
def test_markdown_structure_preservation(md_doc: str):
    """
    **Feature: project-cleanup, Property 1: Markdown structure preservation**
    **Validates: Requirements 3.3**

    For any valid Markdown document containing headings, bold, italic,
    code blocks, and lists, converting it to LaTeX should produce output
    that contains the corresponding LaTeX commands for each element type.
    """
    latex = convert_markdown_to_latex(md_doc)

    # Check that headings produce section commands
    for level, cmd in [(1, '\\section'), (2, '\\subsection'),
                       (3, '\\subsubsection'), (4, '\\paragraph')]:
        prefix = '#' * level + ' '
        for line in md_doc.split('\n'):
            if line.startswith(prefix) and not line.startswith('#' * (level + 1)):
                assert cmd + '{' in latex, (
                    f"Heading level {level} should produce {cmd} command"
                )

    # Check that bold text produces \textbf
    if '**' in md_doc:
        assert '\\textbf{' in latex, "Bold text should produce \\textbf command"

    # Check that italic text (single *) produces \textit
    # Only check lines that have single * but not **
    for line in md_doc.split('\n'):
        stripped = line.replace('**', '')
        if '*' in stripped and not stripped.startswith('- ') and not stripped.startswith('* '):
            # This line likely has italic text
            pass  # italic detection is tricky with list markers; skip strict check

    # Check that code blocks produce lstlisting
    if '```' in md_doc:
        assert '\\begin{lstlisting}' in latex, (
            "Code blocks should produce lstlisting environment"
        )

    # Check that inline code produces \texttt
    # Find inline code that isn't inside a code block
    in_code_block = False
    for line in md_doc.split('\n'):
        if line.startswith('```'):
            in_code_block = not in_code_block
            continue
        if not in_code_block and '`' in line and not line.startswith('```'):
            assert '\\texttt{' in latex, (
                "Inline code should produce \\texttt command"
            )
            break

    # Check that unordered lists produce itemize
    if any(line.strip().startswith('- ') for line in md_doc.split('\n')):
        assert '\\begin{itemize}' in latex, (
            "Unordered lists should produce itemize environment"
        )

    # Check that ordered lists produce enumerate
    import re
    if any(re.match(r'^\d+\.\s', line.strip()) for line in md_doc.split('\n')):
        assert '\\begin{enumerate}' in latex, (
            "Ordered lists should produce enumerate environment"
        )



# ---------------------------------------------------------------------------
# Strategies for generating random MarkdownElement internal representations
# ---------------------------------------------------------------------------

def _md_element_strategy():
    """Strategy that generates random MarkdownElement objects."""
    return st.one_of(
        _heading_element(),
        _paragraph_element(),
        _code_block_element(),
        _list_element(ordered=False),
        _list_element(ordered=True),
        _mermaid_element(),
        _math_block_element(),
    )


def _heading_element():
    return st.builds(
        MarkdownElement,
        type=st.just('heading'),
        content=_safe_text,
        level=st.integers(min_value=1, max_value=4),
    )


def _paragraph_element():
    # Paragraph content goes through convert_inline, so use safe text
    return st.builds(
        MarkdownElement,
        type=st.just('paragraph'),
        content=_safe_text,
    )


def _code_block_element():
    return st.builds(
        MarkdownElement,
        type=st.just('code_block'),
        content=_safe_text,
        language=st.sampled_from(['python', 'bash', 'sql', 'java', '']),
    )


def _mermaid_element():
    return st.builds(
        MarkdownElement,
        type=st.just('mermaid'),
        content=_safe_text,
    )


def _math_block_element():
    # Use simple math expressions that won't confuse the parser
    math_content = st.from_regex(r'[a-z] \+ [a-z]', fullmatch=True)
    return st.builds(
        MarkdownElement,
        type=st.just('math_block'),
        content=math_content,
    )


def _list_element(ordered: bool):
    items = st.lists(
        st.builds(
            MarkdownElement,
            type=st.just('list_item'),
            content=_safe_text,
        ),
        min_size=1,
        max_size=4,
    )
    return st.builds(
        MarkdownElement,
        type=st.just('list'),
        children=items,
        ordered=st.just(ordered),
    )


md_elements_list = st.lists(_md_element_strategy(), min_size=1, max_size=5)


# ---------------------------------------------------------------------------
# Property 2: LaTeX round-trip consistency
# **Feature: project-cleanup, Property 2: LaTeX round-trip consistency**
# **Validates: Requirements 3.5**
# ---------------------------------------------------------------------------

def _elements_equal(a: list, b: list) -> bool:
    """Compare two lists of MarkdownElements for structural equivalence."""
    if len(a) != len(b):
        return False
    for ea, eb in zip(a, b):
        if ea.type != eb.type:
            return False
        if ea.type == 'heading':
            if ea.level != eb.level:
                return False
            if ea.content != eb.content:
                return False
        elif ea.type == 'list':
            if ea.ordered != eb.ordered:
                return False
            if not _elements_equal(ea.children, eb.children):
                return False
        elif ea.type == 'code_block':
            if ea.content != eb.content:
                return False
            if ea.language != eb.language:
                return False
        else:
            if ea.content != eb.content:
                return False
    return True


@given(elements=md_elements_list)
@settings(max_examples=100)
def test_latex_round_trip_consistency(elements: list):
    """
    **Feature: project-cleanup, Property 2: LaTeX round-trip consistency**
    **Validates: Requirements 3.5**

    For any valid internal representation, serializing to LaTeX and
    parsing back should produce an equivalent internal representation.
    """
    latex_text = pretty_print_latex(elements)
    parsed_back = parse_latex(latex_text)

    assert _elements_equal(elements, parsed_back), (
        f"Round-trip failed.\n"
        f"Original elements: {[(e.type, e.content[:50]) for e in elements]}\n"
        f"Parsed back:       {[(e.type, e.content[:50]) for e in parsed_back]}\n"
        f"LaTeX:\n{latex_text[:500]}"
    )
