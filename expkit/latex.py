"""LaTeX rendering helpers: jinja with LaTeX-safe delimiters, number formatting, pdflatex.

Templates use ``<% %>`` blocks, ``<< >>`` variables and ``<# #>`` comments, so
LaTeX's own braces never collide with jinja's. Never put a raw Unicode
infinity (or other non-ASCII math) into .tex; the ``num`` and ``level``
filters spell it out.
"""
import math
import os
import re
import subprocess

from expkit.record import REPO

TEX_SPECIALS = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
                '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\^{}',
                '\\': r'\textbackslash{}'}


def escape(value):
    return ''.join(TEX_SPECIALS.get(c, c) for c in str(value))


def num(value, digits=4):
    """A number for a table cell: significant digits, with missing, NaN and inf spelled out."""
    if value is None:
        return '--'
    if isinstance(value, bool):
        return 'on' if value else 'off'
    try:
        value = float(value)
    except (TypeError, ValueError):
        return escape(value)
    if math.isnan(value):
        return '--'
    if math.isinf(value):
        return r'$\infty$' if value > 0 else r'$-\infty$'
    if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
        mantissa, exponent = f'{value:.{digits - 1}e}'.split('e')
        return f'${mantissa}\\times10^{{{int(exponent)}}}$'
    return f'{value:.{digits}g}'


def level(value):
    """A factor level for the page: numbers via ``num``, booleans as on/off, else escaped."""
    if isinstance(value, str):
        if value.strip().lower() in ('inf', '+inf', 'infinity'):
            return r'$\infty$'
        return escape(value)
    return num(value)


def pvalue(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return '--'
    return r'$<10^{-4}$' if p < 1e-4 else f'{p:.4f}'


def display_path(path):
    """Repository-relative when inside the repository, so it fits on the page."""
    if not path:
        return ''
    rel = os.path.relpath(os.path.abspath(path), REPO)
    return str(path) if rel.startswith('..') else rel


def environment(template_dir):
    """Jinja with LaTeX-safe delimiters: <% %> blocks, << >> variables, <# #> comments."""
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        block_start_string='<%', block_end_string='%>',
        variable_start_string='<<', variable_end_string='>>',
        comment_start_string='<#', comment_end_string='#>',
        trim_blocks=True, lstrip_blocks=True, undefined=jinja2.StrictUndefined)
    env.filters.update(escape=escape, num=num, level=level, pvalue=pvalue,
                       display_path=display_path)
    env.globals.update(zip=zip)
    return env


OVERFULL = re.compile(r'^Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\) (.*)$')


def overfull(log_path, tolerance=1.0):
    """``[(points, where)]`` for every box wider than the text by more than
    ``tolerance`` points: a table or line running into the margin."""
    try:
        with open(log_path, errors='replace') as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    found = [OVERFULL.match(line) for line in lines]
    return [(float(m.group(1)), m.group(2)) for m in found if m and float(m.group(1)) > tolerance]


def compile_pdf(tex_path, log=print):
    """pdflatex twice (for references). Returns the PDF path, or None if it failed."""
    directory, name = os.path.split(os.path.abspath(tex_path))
    for _ in range(2):
        try:
            result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', name],
                                    cwd=directory, capture_output=True, text=True)
        except FileNotFoundError:
            log('pdflatex is not installed; the .tex is written but not compiled')
            return None
        if result.returncode != 0:
            errors = [line for line in result.stdout.splitlines() if line.startswith('!')]
            log(f'pdflatex failed on {tex_path}: {"; ".join(errors[:3]) or "see its .log"}')
            return None
    for points, where in overfull(os.path.join(directory, name[:-4] + '.log')):
        log(f'warning: {tex_path}: overfull by {points:g}pt {where}')
    return os.path.join(directory, name[:-4] + '.pdf')
