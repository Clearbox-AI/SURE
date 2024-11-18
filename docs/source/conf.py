import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adjust the path as needed

project = 'sure'
author = 'Dario Brunelli'
release = '0.1.9.9'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "canonical_url": "https://synthetics.docs.gretel.ai/en/stable/",
    "logo_only": True,
    "display_version": True,
    "style_nav_header_background": "#483a8f",
}
html_logo = "img/cb_purple_logo_compact.png"
html_static_path = ['_static']

master_doc = 'index'  # Ensure this points to your main document


# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False