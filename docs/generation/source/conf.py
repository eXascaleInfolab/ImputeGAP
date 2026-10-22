# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

project = 'imputegap'
copyright = '2026, Quentin Nater'
author = 'Quentin Nater'
html_title = 'ImputeGAP 1.1.3'
html_short_title = 'ImputeGAP 1.1.3 Documentation'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    'sphinx_tabs.tabs'
]

autosummary_generate = True  # Automatically generate method summaries
html_sidebars = {
    "**": [
        "sidebar/brand.html",  # Logo and project name
        "sidebar/search.html",  # Search bar
        "sidebar/github.html",
       # "sidebar/tutorial.html",  # NEW: KDD tutorial
        "sidebar/navigation.html",  # Collapsible navigation
        "sidebar/scroll-start.html",
        "sidebar/scroll-end.html",
    ]
}


html_logo = "https://raw.githubusercontent.com/eXascaleInfolab/ImputeGAP/main/assets/logo_imputegab.png"
html_favicon = "https://raw.githubusercontent.com/eXascaleInfolab/ImputeGAP/main/assets/favicon.png"

html_static_path = ['static']
html_css_files = ['custom.css']

# Set the version and release info
version = '1.1.3'
release = '1.1.3'



templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ['_static']

napoleon_google_docstring = True
napoleon_numpy_docstring = False



sys.path.insert(0, os.path.abspath('../../../'))  # Adjust path to the project root


def setup(app):
    app.add_css_file("custom.css")