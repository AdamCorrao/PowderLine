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
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'dev']  # dev/ = in-repo development dossiers, not user-facing docs

# Treat missing cross-reference targets as errors, not warnings
nitpicky = True

# MyST parser settings for Markdown support
myst_enable_extensions = [
    "deflist",      # Definition lists
    "colon_fence",  # ::: fences
    "dollarmath",   # $$...$$ math blocks (e.g. Rwp formula in DEVELOPMENT.md)
]
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
