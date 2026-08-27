# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add src directory to Python path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PowderLine'
copyright = '2026, NSLS-II / Brookhaven National Laboratory'
author = 'Dan Olds and Adam Corrao'
# Single-source the version from the package (pyproject reads the same
# attribute via [tool.hatch.version]).
from powderline import __version__ as release  # noqa: E402

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',     # Support Google/NumPy docstring styles
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx.ext.intersphinx',  # Link to other project docs (e.g., Python, NumPy)
    'myst_parser',             # Support Markdown files
]

# GSAS-II is a heavy conda dependency absent from the RTD / doc-build
# environment; mock it so autodoc can import powderline.kicker
# (kicker.py imports `from GSASII import ...` unguarded at module top).
autodoc_mock_imports = ["GSASII"]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping to link to external docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'pydantic': ('https://docs.pydantic.dev/latest/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'dev']  # dev/ = in-repo development dossiers, not user-facing docs

# Treat missing cross-reference targets as warnings (not silently ignored)
nitpicky = True

# RefinementParameter fields are `Annotated[tuple[...], PlainSerializer(...)]`.
# Sphinx's type-hint renderer expands the PlainSerializer repr into bogus
# sub-references (its keyword args and the fully-expanded Annotated/list/dict
# forms) that can never resolve to real objects. Silence just those synthetic
# targets; genuine unresolved references still warn normally.
nitpick_ignore_regex = [
    ('py:class', r'^func=.*$'),
    ('py:class', r'^return_type=.*$'),
    ('py:class', r'^when_used=.*$'),
    ('py:class', r'.*PlainSerializer.*'),
    ('py:obj', r'.*PlainSerializer.*'),
    ('py:class', r'^ConfigDict$'),
]

# pandas' public docs index `pandas.DataFrame`, but runtime type hints resolve
# to its internal module path; the pandas intersphinx inventory has no entry
# for the latter, so it can never resolve.
nitpick_ignore = [
    ('py:class', 'pandas.core.frame.DataFrame'),
]

# MyST parser settings for Markdown support
myst_enable_extensions = [
    "deflist",      # Definition lists
    "colon_fence",  # ::: fences
    "dollarmath",   # $$...$$ math blocks (e.g. Rwp formula in DEVELOPMENT.md)
]
# Auto-generate GitHub-style heading anchors (up to H4) so in-page TOC links
# like `[Root Causes Summary](#1-root-causes-summary)` in cross-platform-guide.md
# resolve instead of raising 'myst.xref_missing'.
myst_heading_anchors = 4
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # Read the Docs theme (install via pixi if not available)
html_static_path = ['_static']

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Type hints configuration
autodoc_typehints = 'description'
autodoc_type_aliases = {
    'RefinementParameter': 'powderline.schema.RefinementParameter',
}
