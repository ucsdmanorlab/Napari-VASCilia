# Configuration file for the Sphinx documentation builder.
# Build by Yasmin Kassim
# -- Project information -----------------------------------------------------
project = 'VASCilia'
copyright = '2024, Yasmin Kassim'
author = 'Yasmin Kassim'
release = '1.3.0'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",      # Auto-generate documentation from docstrings
    "sphinx.ext.viewcode",     # Add links to source code
    "sphinx.ext.githubpages"   # Ensure compatibility with GitHub Pages
]

templates_path = ['_templates']  # Path for custom templates
exclude_patterns = []           # Patterns to exclude from the build

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']  # Path for static assets (CSS, images, JS)
html_logo = "_static/VASCilia_logo1.png"  # Logo file

# GitHub Pages base URL
html_baseurl = "https://ucsdmanorlab.github.io/Napari-VASCilia/"

# Include custom CSS files
html_css_files = [
    "custom.css"  # Your custom styles
]

# Theme options for sphinx_book_theme
html_theme_options = {
    "repository_url": "https://github.com/ucsdmanorlab/Napari-VASCilia",
    "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "show_navbar_depth": 2,
}

# -- Custom settings for GitHub Actions -------------------------------------
# Ensure the path to images inside the `features/_static` folder works
html_extra_path = ['features/_static']
