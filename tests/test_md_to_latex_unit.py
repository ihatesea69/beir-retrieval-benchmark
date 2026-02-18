"""
Unit tests for Markdown to LaTeX converter edge cases.

_Requirements: 3.3_
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from md_to_latex import (
    convert_markdown_to_latex,
    escape_latex,
    parse_markdown,
    strip_emoji,
)


class TestEmptyInput:
    def test_empty_string(self):
        result = convert_markdown_to_latex('')
        assert result == ''

    def test_whitespace_only(self):
        result = convert_markdown_to_latex('   \n\n   ')
        assert result == ''


class TestCodeBlockOnly:
    def test_single_code_block(self):
        md = '```python\nx = 1\n```'
        result = convert_markdown_to_latex(md)
        assert '\\begin{lstlisting}[language=python]' in result
        assert 'x = 1' in result
        assert '\\end{lstlisting}' in result

    def test_code_block_no_language(self):
        md = '```\nhello\n```'
        result = convert_markdown_to_latex(md)
        assert '\\begin{lstlisting}\n' in result


class TestNestedLists:
    def test_unordered_list(self):
        md = '- item 1\n- item 2\n- item 3'
        result = convert_markdown_to_latex(md)
        assert '\\begin{itemize}' in result
        assert '\\item' in result
        assert '\\end{itemize}' in result

    def test_ordered_list(self):
        md = '1. first\n2. second'
        result = convert_markdown_to_latex(md)
        assert '\\begin{enumerate}' in result
        assert '\\end{enumerate}' in result


class TestLatexSpecialCharEscaping:
    def test_ampersand(self):
        assert escape_latex('A & B') == r'A \& B'

    def test_percent(self):
        assert escape_latex('100%') == r'100\%'

    def test_dollar(self):
        assert escape_latex('$5') == r'\$5'

    def test_hash(self):
        assert escape_latex('#tag') == r'\#tag'

    def test_underscore(self):
        assert escape_latex('a_b') == r'a\_b'

    def test_braces(self):
        assert escape_latex('{x}') == r'\{x\}'

    def test_tilde(self):
        assert escape_latex('~') == r'\textasciitilde{}'

    def test_caret(self):
        assert escape_latex('^') == r'\textasciicircum{}'

    def test_backslash(self):
        assert escape_latex('\\') == r'\textbackslash{}'

    def test_multiple_specials(self):
        result = escape_latex('a & b % c')
        assert result == r'a \& b \% c'


class TestMermaidDiagram:
    def test_mermaid_becomes_comment(self):
        md = '```mermaid\ngraph TD\nA --> B\n```'
        result = convert_markdown_to_latex(md)
        assert '% Mermaid diagram' in result
        assert '% graph TD' in result
        assert '% A --> B' in result
        assert '\\begin{lstlisting}' not in result


class TestEmojiStripping:
    def test_strip_emoji_from_heading(self):
        md = '# 🔬 Hello World'
        result = convert_markdown_to_latex(md)
        assert '\\section{' in result
        assert '🔬' not in result

    def test_strip_emoji_preserves_text(self):
        result = strip_emoji('Hello 🌟 World')
        assert 'Hello' in result
        assert 'World' in result
        assert '🌟' not in result
