# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

project = 'imputegap'
copyright = '2025, Quentin Nater'
author = 'Quentin Nater'
html_title = 'ImputeGAP 1.1.2'
html_short_title = 'ImputeGAP 1.1.2 Documentation'


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
        "sidebar/navigation.html",  # Collapsible navigation
        "sidebar/scroll-start.html",
        "sidebar/scroll-end.html",
    ]
}




html_logo = "https://www.naterscreations.com/imputegap/logo_imputegab.png"
html_favicon = "https://www.naterscreations.com/imputegap/favicon.png"

html_static_path = ['static']
html_css_files = ['custom.css']

# Set the version and release info
version = '1.1.2'
release = '1.1.2'


html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-sidebar-background": "#1e1e1e",  # Dark sidebar background
        "color-sidebar-text": "#ffffff",  # White text
        "sidebar-width": "300px",  # Adjust sidebar width
    },
}




templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ['_static']

napoleon_google_docstring = True
napoleon_numpy_docstring = False


html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
}

sys.path.insert(0, os.path.abspath('../../../'))  # Adjust path to the project root
