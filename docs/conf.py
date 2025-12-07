# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Project information -----------------------------------------------
project = "torchvision-customizer"
copyright = "2025, CodeWithDark"
author = "CodeWithDark"
release = "2.0.0"
version = "2.0"

# -- General configuration -----------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.todo",
    "sphinxcontrib.autodoc_pydantic",
]

# Configure autodoc to include private members and sort by source
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}

# Napoleon settings for Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_method = True
napoleon_include_private_with_doc = False
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Templates path
templates_path = ["_templates"]

# Source file suffixes
source_suffix = {
    ".rst": None,
    ".md": "markdown",
}

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Master document
master_doc = "index"

# Language
language = "en"

# -- Options for HTML output -----------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2c3e50",
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# HTML context
html_context = {
    "display_github": True,
    "github_user": "codewithdark-git",
    "github_repo": "torchvision-customizer",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Add custom CSS
html_css_files = ["custom.css"]

# -- Options for LaTeX output -----------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": r"""
\usepackage{amsmath}
\usepackage{amssymb}
""",
}

latex_documents = [
    (
        master_doc,
        "torchvision-customizer.tex",
        "torchvision-customizer Documentation",
        "CodeWithDark",
        "manual",
    ),
]

# -- Options for manual page output -----------------------------------------------
man_pages = [
    (
        master_doc,
        "torchvision-customizer",
        "torchvision-customizer Documentation",
        [author],
        1,
    ),
]

# -- Options for Texinfo output -----------------------------------------------
texinfo_documents = [
    (
        master_doc,
        "torchvision-customizer",
        "torchvision-customizer Documentation",
        author,
        "torchvision-customizer",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- Sphinx Todo configuration -----------------------------------------------
todo_include_todos = True

# -- Suppress warnings -----------------------------------------------
suppress_warnings = ["app.add_config_value"]

# Autodoc settings for better documentation
add_module_names = False
