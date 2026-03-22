#!/usr/bin/env python3
"""
tex.py - LaTeX to Markdown Converter

Converts .tex (LaTeX) files to markdown (default) or plain text
using pure regex-based conversion — no external dependencies (no pandoc,
no pylatexenc, no latextomd).

Usage:
    python tex.py [options] <input.tex>
    python tex.py [options] <directory/>

Examples:
    python tex.py paper.tex
    python tex.py docs/ -o ~/notes/
    python tex.py thesis.tex -f txt
"""

import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from typing_extensions import Annotated

from any2md.common import build_frontmatter, setup_logging, OutputFormat, write_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Math preservation helpers
# ---------------------------------------------------------------------------

# Placeholder token to protect math from regex mangling
_MATH_PLACEHOLDER = "\x00MATH{}\x00"


def _protect_math(text: str) -> Tuple[str, List[str]]:
    """
    Replace math regions with numbered placeholders so other regexes won't
    accidentally transform them.

    Handles (in order of priority, longest-first):
      - \\[...\\]        display math
      - \\(...\\)        inline math
      - $$...$$          display math
      - $...$            inline math
      - \\begin{equation}...\\end{equation}
      - \\begin{align}...\\end{align}  (and align* / aligned)
      - \\begin{gather}...\\end{gather}
      - \\begin{multline}...\\end{multline}

    Args:
        text: Raw LaTeX string

    Returns:
        (protected_text, math_stash) where math_stash[i] is the original math
        region for placeholder i.
    """
    stash: List[str] = []

    def _stash(m: re.Match) -> str:
        idx = len(stash)
        stash.append(m.group(0))
        return _MATH_PLACEHOLDER.format(idx)

    # Display math environments (order: named envs first, then delimiters)
    math_envs = (
        r'equation\*?', r'align\*?', r'aligned', r'gather\*?',
        r'multline\*?', r'eqnarray\*?', r'flalign\*?', r'alignat\*?',
    )
    env_pat = '|'.join(math_envs)
    text = re.sub(
        rf'\\begin{{({env_pat})}}(.*?)\\end{{\1}}',
        _stash, text, flags=re.DOTALL
    )

    # \[...\]
    text = re.sub(r'\\\[.*?\\\]', _stash, text, flags=re.DOTALL)

    # \(...\)
    text = re.sub(r'\\\(.*?\\\)', _stash, text, flags=re.DOTALL)

    # $$...$$
    text = re.sub(r'\$\$.*?\$\$', _stash, text, flags=re.DOTALL)

    # $...$  (non-greedy, single-line preferred but allow escaped dollars)
    text = re.sub(r'\$[^$\n]+?\$', _stash, text)

    return text, stash


def _restore_math(text: str, stash: List[str]) -> str:
    """Restore math placeholders with their original LaTeX."""
    for idx, math in enumerate(stash):
        text = text.replace(_MATH_PLACEHOLDER.format(idx), math)
    return text


_CODE_PLACEHOLDER = "\x00CODE{}\x00"


def _protect_code(text: str) -> Tuple[str, List[str]]:
    """
    Replace already-converted markdown code regions with placeholders so that
    the subsequent LaTeX-stripping passes cannot mangle them.

    Protects:
    - Fenced code blocks (``` ... ```)
    - Inline code spans (`...`)
    """
    stash: List[str] = []

    def _stash(m: re.Match) -> str:
        idx = len(stash)
        stash.append(m.group(0))
        return _CODE_PLACEHOLDER.format(idx)

    # Fenced blocks first (multi-line)
    text = re.sub(r'```.*?```', _stash, text, flags=re.DOTALL)
    # Inline code spans
    text = re.sub(r'`[^`\n]+`', _stash, text)

    return text, stash


def _restore_code(text: str, stash: List[str]) -> str:
    """Restore code placeholders."""
    for idx, code in enumerate(stash):
        text = text.replace(_CODE_PLACEHOLDER.format(idx), code)
    return text


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_tex_metadata(tex_content: str, source_path: Path) -> Tuple[Dict, str]:
    """
    Extract metadata from LaTeX preamble commands and the abstract environment.

    Handles:
      \\title{...}, \\author{...}, \\date{...}
      \\begin{abstract}...\\end{abstract}

    Args:
        tex_content: Raw LaTeX file content
        source_path: Path to the source .tex file

    Returns:
        (metadata_dict, abstract_text) where metadata_dict is suitable for
        build_frontmatter() and abstract_text is the raw abstract body (or '').
    """
    metadata: Dict = {
        'source': str(source_path.resolve()),
        'fetched_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    def _extract_braced(cmd: str, content: str) -> Optional[str]:
        """Return the first {}-enclosed argument of \\cmd in content."""
        m = re.search(
            rf'\\{re.escape(cmd)}\s*\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}',
            content, re.DOTALL
        )
        return _strip_tex_commands(m.group(1)).strip() if m else None

    for field in ('title', 'author', 'date'):
        val = _extract_braced(field, tex_content)
        if val:
            metadata[field] = val

    # Abstract environment
    abstract_match = re.search(
        r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
        tex_content, re.DOTALL
    )
    abstract_text = ''
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        metadata['abstract'] = _strip_tex_commands(abstract_text).strip()

    return {k: v for k, v in metadata.items() if v}, abstract_text


def _strip_tex_commands(text: str) -> str:
    """
    Lightly clean LaTeX markup from a short metadata string (title/author/date).

    Removes common formatting commands while keeping the text content.
    """
    # Remove simple commands that wrap text: \cmd{text} → text
    text = re.sub(r'\\(?:textbf|textit|emph|textrm|textsc|texttt|text)\{([^}]*)\}', r'\1', text)
    # Remove \\ line breaks
    text = re.sub(r'\\\\', ' ', text)
    # Remove other known formatting commands
    text = re.sub(r'\\(?:and|thanks)\b\s*', '', text)
    text = re.sub(r'\\[a-zA-Z]+\*?\s*', '', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Preamble / document wrapper stripping
# ---------------------------------------------------------------------------

_PREAMBLE_COMMANDS = re.compile(
    r'\\(?:documentclass|usepackage|RequirePackage|LoadClass|NeedsTeXFormat)'
    r'(?:\[[^\]]*\])?\{[^}]*\}\s*\n?',
    re.MULTILINE,
)

_DOCUMENT_WRAPPERS = re.compile(
    r'\\begin\{document\}|\\end\{document\}|\\maketitle\b',
)


def _strip_preamble_and_wrappers(text: str) -> str:
    """Remove document-structure commands that have no markdown equivalent."""
    text = _PREAMBLE_COMMANDS.sub('', text)
    text = _DOCUMENT_WRAPPERS.sub('', text)
    # Strip \title{}, \author{}, \date{}, \begin{abstract}...\end{abstract}
    # (already extracted into metadata)
    text = re.sub(
        r'\\(?:title|author|date)\s*\{(?:[^{}]|\{[^{}]*\})*\}\s*\n?',
        '', text, flags=re.DOTALL
    )
    text = re.sub(
        r'\\begin\{abstract\}.*?\\end\{abstract\}\s*\n?',
        '', text, flags=re.DOTALL
    )
    return text


# ---------------------------------------------------------------------------
# Core LaTeX → Markdown conversion
# ---------------------------------------------------------------------------

def _convert_sections(text: str) -> str:
    """
    Convert LaTeX section commands to ATX-style markdown headings.

      \\chapter{T}         →  # T
      \\section{T}         →  ## T
      \\subsection{T}      →  ### T
      \\subsubsection{T}   →  #### T
      \\paragraph{T}       →  ##### T
      \\subparagraph{T}    →  ###### T
    """
    mapping = [
        ('chapter', '#'),
        ('subsubsection', '####'),
        ('subsection', '###'),
        ('section', '##'),
        ('paragraph', '#####'),
        ('subparagraph', '######'),
    ]
    for cmd, prefix in mapping:
        text = re.sub(
            rf'\\{cmd}\*?\s*\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}',
            lambda m, p=prefix: f'\n{p} {m.group(1).strip()}\n',
            text
        )
    return text


def _convert_emphasis(text: str) -> str:
    """
    Convert LaTeX text-formatting commands to markdown equivalents.

      \\textbf{T}   →  **T**
      \\textit{T}   →  *T*
      \\emph{T}     →  *T*
      \\texttt{T}   →  `T`
      \\underline{T} → T   (markdown has no underline)
    """
    # Bold
    text = re.sub(
        r'\\textbf\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
        lambda m: f'**{m.group(1).strip()}**', text
    )
    # Italic (textit + emph)
    text = re.sub(
        r'\\(?:textit|emph)\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
        lambda m: f'*{m.group(1).strip()}*', text
    )
    # Monospace
    text = re.sub(
        r'\\texttt\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
        lambda m: f'`{m.group(1).strip()}`', text
    )
    # Underline — drop the decoration, keep text
    text = re.sub(
        r'\\underline\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
        r'\1', text
    )
    # textrm, textsc, textmd — drop formatting, keep text
    text = re.sub(
        r'\\(?:textrm|textsc|textmd|textnormal)\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
        r'\1', text
    )
    return text


def _convert_verbatim(text: str) -> str:
    """
    Convert verbatim environments and \\verb to markdown code.

      \\begin{verbatim}...\\end{verbatim}  →  ``` fenced block
      \\begin{lstlisting}...\\end{lstlisting}  →  ``` fenced block
      \\verb|text|  →  `text`  (any delimiter char)
    """
    # verbatim / lstlisting environments
    for env in ('verbatim', 'lstlisting', 'Verbatim', 'minted'):
        text = re.sub(
            rf'\\begin\{{{env}\}}(?:\[.*?\])?(.*?)\\end\{{{env}\}}',
            lambda m: '\n```\n' + m.group(1) + '\n```\n',
            text, flags=re.DOTALL
        )

    # \verb|text| — delimiter can be any non-alpha character
    text = re.sub(
        r'\\verb([^a-zA-Z\s])(.*?)\1',
        lambda m: f'`{m.group(2)}`',
        text
    )
    return text


def _convert_lists(text: str) -> str:
    """
    Convert itemize and enumerate environments to markdown lists.

    Strategy: handle enumerate first (numbered items), then itemize/description
    (bulleted items). Each pass matches the full environment block and replaces
    \\item markers before stripping the begin/end wrappers.
    """
    item_pat = re.compile(r'\\item(?:\[([^\]]*)\])?\s*')

    def _number_items(block: str) -> str:
        """Replace \\item markers in an enumerate block with 1., 2., ..."""
        counter = [0]

        def repl(im: re.Match) -> str:
            counter[0] += 1
            label = im.group(1)
            return f'{counter[0]}. ' if not label else f'{counter[0]}. **{label}** '

        return item_pat.sub(repl, block)

    def _bullet_items(block: str) -> str:
        """Replace \\item markers with - bullets."""
        def repl(im: re.Match) -> str:
            label = im.group(1)
            return '- ' if not label else f'- **{label}** '

        return item_pat.sub(repl, block)

    # Process enumerate blocks: number their items, then strip wrappers
    text = re.sub(
        r'\\begin\{enumerate\}(?:\[.*?\])?(.*?)\\end\{enumerate\}',
        lambda m: '\n' + _number_items(m.group(1)) + '\n',
        text, flags=re.DOTALL
    )

    # Process itemize blocks: bullet their items, then strip wrappers
    text = re.sub(
        r'\\begin\{itemize\}(?:\[.*?\])?(.*?)\\end\{itemize\}',
        lambda m: '\n' + _bullet_items(m.group(1)) + '\n',
        text, flags=re.DOTALL
    )

    # Process description blocks: bullet their items, then strip wrappers
    text = re.sub(
        r'\\begin\{description\}(?:\[.*?\])?(.*?)\\end\{description\}',
        lambda m: '\n' + _bullet_items(m.group(1)) + '\n',
        text, flags=re.DOTALL
    )

    # Any remaining stray \item markers (outside known environments) → bullet
    text = item_pat.sub(lambda im: '- ' if not im.group(1) else f'- **{im.group(1)}** ', text)

    return text


def _convert_figures(text: str) -> str:
    """
    Convert \\includegraphics to markdown image syntax.

      \\includegraphics[options]{path}  →  ![](path)

    Also handles figure/subfigure environments (strips wrappers).
    """
    # \\includegraphics[opts]{path}
    text = re.sub(
        r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',
        r'![](\1)',
        text
    )
    # Strip figure environment wrappers (keep contents)
    text = re.sub(r'\\begin\{(?:figure|subfigure|wrapfigure)\*?\}(?:\[.*?\])?', '\n', text)
    text = re.sub(r'\\end\{(?:figure|subfigure|wrapfigure)\*?\}', '\n', text)
    # \caption{text} → *text*
    text = re.sub(r'\\caption\{((?:[^{}]|\{[^{}]*\})*)\}', r'*\1*', text)
    return text


def _convert_citations_and_refs(text: str) -> str:
    """
    Convert citation and cross-reference commands.

      \\cite{key}        →  [key]
      \\cite[note]{key}  →  [key, note]
      \\ref{label}       →  [label]
      \\eqref{label}     →  [(label)]
      \\pageref{label}   →  [p. label]
      \\label{x}         →  (stripped)
    """
    # \cite[note]{keys}
    text = re.sub(
        r'\\cite(?:p|t|alt|alp|num|year|author)?\s*(?:\[([^\]]*)\])?\{([^}]+)\}',
        lambda m: f'[{m.group(2)}, {m.group(1)}]' if m.group(1) else f'[{m.group(2)}]',
        text
    )
    # \ref{label}
    text = re.sub(r'\\ref\{([^}]+)\}', r'[\1]', text)
    # \eqref{label}
    text = re.sub(r'\\eqref\{([^}]+)\}', r'[(\1)]', text)
    # \pageref{label}
    text = re.sub(r'\\pageref\{([^}]+)\}', r'[p. \1]', text)
    # \label{x} — strip completely
    text = re.sub(r'\\label\{[^}]+\}', '', text)
    return text


def _convert_links(text: str) -> str:
    """
    Convert hyperref link commands.

      \\href{url}{text}  →  [text](url)
      \\url{url}         →  <url>
    """
    text = re.sub(
        r'\\href\{([^}]+)\}\{([^}]+)\}',
        r'[\2](\1)',
        text
    )
    text = re.sub(r'\\url\{([^}]+)\}', r'<\1>', text)
    return text


def _convert_tables(text: str) -> str:
    """
    Strip table environments (tabular, table) — too complex for regex, output raw.

    The environment wrapper is removed and content is left as-is (or lightly
    cleaned) rather than attempting a full conversion that would be brittle.
    """
    text = re.sub(r'\\begin\{(?:table|tabular|tabularx|longtable)\*?\}(?:\[.*?\])?(?:\{[^}]*\})?', '\n', text)
    text = re.sub(r'\\end\{(?:table|tabular|tabularx|longtable)\*?\}', '\n', text)
    text = re.sub(r'\\hline\b', '', text)
    text = re.sub(r'\\cline\{[^}]+\}', '', text)
    text = re.sub(r'&', ' | ', text)
    text = re.sub(r'\\\\', '\n', text)
    return text


def _convert_misc_commands(text: str) -> str:
    """
    Handle miscellaneous LaTeX commands with simple markdown equivalents or stripping.
    """
    # Footnotes → inline parenthetical
    text = re.sub(r'\\footnote\{((?:[^{}]|\{[^{}]*\})*)\}', r' (\1)', text)

    # Horizontal rules
    text = re.sub(r'\\(?:rule|hrule|noindent|hline)\b[^{}\n]*', '', text)

    # Spacing commands → strip
    text = re.sub(r'\\(?:vspace|hspace|vskip|hskip)\*?\{[^}]*\}', '', text)
    text = re.sub(r'\\(?:bigskip|medskip|smallskip|noindent|indent)\b', '', text)

    # Quotes
    # LaTeX quote ligatures: `` → " and '' → " (only double-char sequences)
    # Do NOT replace single backtick — that is already-converted markdown code.
    text = text.replace('``', '\u201c').replace("''", '\u201d')

    # Ligatures and special chars
    text = text.replace('---', '\u2014').replace('--', '\u2013')
    text = text.replace('~', '\u00a0')  # non-breaking space → nbsp or space
    text = text.replace('\\~{}', '~')
    text = re.sub(r'\\(?:LaTeX|TeX)\b', lambda m: m.group(0)[1:].rstrip('{}'), text)

    # \newline and \\ line breaks
    text = re.sub(r'\\(?:newline|linebreak)\b', '\n', text)

    # Strip remaining known no-arg commands
    no_arg = (
        'clearpage', 'cleardoublepage', 'newpage', 'pagebreak',
        'tableofcontents', 'listoffigures', 'listoftables',
        'bibliographystyle', 'printbibliography', 'bibliography',
        'appendix', 'frontmatter', 'mainmatter', 'backmatter',
        'centering', 'raggedright', 'raggedleft',
        'small', 'footnotesize', 'large', 'Large', 'LARGE', 'huge', 'Huge',
        'normalsize', 'normalfont',
    )
    text = re.sub(r'\\(?:' + '|'.join(no_arg) + r')\b\*?(?:\{[^}]*\}|\[[^\]]*\])?', '', text)

    # Strip remaining {}-argument commands that have no markdown analog
    # This is a last-resort sweep: \someCmd{content} → content
    text = re.sub(r'\\[a-zA-Z]+\*?\{((?:[^{}]|\{[^{}]*\})*)\}', r'\1', text)

    # Strip bare \command markers (no args)
    text = re.sub(r'\\[a-zA-Z]+\*?\b', '', text)

    # Remove orphaned braces
    text = re.sub(r'[{}]', '', text)

    # Tilde used as non-breaking space in running text
    text = text.replace('\u00a0', ' ')

    return text


def _cleanup_whitespace(text: str) -> str:
    """Normalize whitespace: collapse blank lines, fix trailing spaces."""
    # Strip trailing whitespace on each line
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main conversion entry point
# ---------------------------------------------------------------------------

def tex_to_markdown_text(tex_content: str) -> str:
    """
    Convert a LaTeX string to markdown using pure regex transforms.

    Conversion order:
    1. Protect math regions from other transforms
    2. Convert verbatim/code environments (before emphasis strip)
    3. Strip preamble and document wrapper commands
    4. Convert sections, emphasis, lists, figures, citations
    5. Strip remaining LaTeX commands
    6. Restore math regions
    7. Normalize whitespace

    Args:
        tex_content: Raw LaTeX string

    Returns:
        Markdown-formatted string (without frontmatter)
    """
    text = tex_content

    # Step 1: protect math
    text, math_stash = _protect_math(text)

    # Step 2: verbatim / code blocks (before other transforms), then protect them
    text = _convert_verbatim(text)
    text, code_stash = _protect_code(text)

    # Step 3: strip preamble and document wrappers
    text = _strip_preamble_and_wrappers(text)

    # Step 4a: sections
    text = _convert_sections(text)

    # Step 4b: emphasis
    text = _convert_emphasis(text)

    # Step 4c: figures
    text = _convert_figures(text)

    # Step 4d: links
    text = _convert_links(text)

    # Step 4e: citations and cross-refs
    text = _convert_citations_and_refs(text)

    # Step 4f: lists
    text = _convert_lists(text)

    # Step 4g: tables (light cleanup)
    text = _convert_tables(text)

    # Step 5: miscellaneous command stripping
    text = _convert_misc_commands(text)

    # Step 6: restore code spans/blocks, then math
    text = _restore_code(text, code_stash)
    text = _restore_math(text, math_stash)

    # Step 7: normalize whitespace
    text = _cleanup_whitespace(text)

    return text


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def tex_to_full_markdown(
    md_content: str,
    metadata: Optional[Dict] = None,
    abstract: str = '',
) -> str:
    """
    Assemble final markdown output with optional YAML frontmatter and abstract.

    Args:
        md_content: Converted markdown from tex_to_markdown_text()
        metadata: Optional metadata dict for YAML frontmatter
        abstract: Optional abstract text to prepend as a blockquote

    Returns:
        Full markdown string ready to write to disk
    """
    lines = []

    if metadata:
        lines.append(build_frontmatter(metadata))
        lines.append('')

    if abstract:
        # Render abstract as a blockquoted paragraph
        lines.append('> **Abstract.** ' + abstract.replace('\n', ' '))
        lines.append('')

    lines.append(md_content)
    return '\n'.join(lines)


def tex_to_plain_text(md_content: str) -> str:
    """
    Strip markdown syntax from converted content to produce plain text.

    Args:
        md_content: Markdown string from tex_to_markdown_text()

    Returns:
        Plain text string
    """
    text = md_content
    # Remove ATX heading markers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove code fences
    text = re.sub(r'^```[^\n]*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    # Remove image syntax
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    # Remove link syntax
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Single-file processor
# ---------------------------------------------------------------------------

def process_tex_file(tex_path: Path, output_dir: Path, fmt: str) -> Path:
    """
    Convert one LaTeX file to the requested output format and write to disk.

    Args:
        tex_path: Path to the source .tex file
        output_dir: Directory in which to write the output file
        fmt: Output format string ('md' or 'txt')

    Returns:
        Path to the written output file
    """
    logger.info("Processing: %s", tex_path)

    tex_content = tex_path.read_text(encoding='utf-8', errors='replace')
    metadata, abstract = extract_tex_metadata(tex_content, tex_path)

    md_content = tex_to_markdown_text(tex_content)

    if fmt == 'md':
        output = tex_to_full_markdown(md_content, metadata=metadata, abstract=abstract)
    else:
        output = tex_to_plain_text(md_content)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (tex_path.stem + '.' + fmt)
    out_path.write_text(output, encoding='utf-8')
    logger.info("Written: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Convert LaTeX (.tex) files to markdown or plain text.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.command()
def main(
    input_path: Annotated[Path, typer.Argument(
        help="LaTeX file or directory containing .tex files.",
    )],
    output_dir: Annotated[Path, typer.Option(
        "--output-dir", "-o",
        help="Directory to save output files.",
    )] = Path("."),
    format: Annotated[OutputFormat, typer.Option(
        "--format", "-f",
        help="Output format: [bold]md[/bold] (markdown with frontmatter), [bold]txt[/bold] (plain text).",
    )] = OutputFormat.md,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose (DEBUG) logging.",
    )] = False,
) -> None:
    """
    Convert LaTeX files to markdown (default) or plain text.

    Accepts a single .tex file or a directory of .tex files.
    Produces YAML frontmatter from \\title, \\author, \\date, and \\begin{abstract}.
    Math is preserved as-is ($...$, $$...$$, environments).
    """
    setup_logging(verbose)

    if input_path.is_dir():
        tex_files: List[Path] = list(input_path.glob("*.tex"))
        if not tex_files:
            typer.echo(f"No .tex files found in {input_path}", err=True)
            raise typer.Exit(1)
        tex_files = sorted(tex_files)
    else:
        if not input_path.exists():
            typer.echo(f"File not found: {input_path}", err=True)
            raise typer.Exit(1)
        tex_files = [input_path]

    fmt = format.value
    for tex_file in tex_files:
        out = process_tex_file(tex_file, output_dir, fmt)
        typer.echo(f"Written: {out}", err=True)


if __name__ == "__main__":
    app()
