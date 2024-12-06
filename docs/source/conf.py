# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'VASCilia'
copyright = '2024, Yasmin Kassim'
author = 'Yasmin Kassim'
release = '1.3.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_logo = "_static/VASCilia_logo1.png"
html_baseurl = "https://ucsdmanorlab.github.io/Napari-VASCilia/"
html_css_files = [
    "_static/styles/sphinx-book-theme.css",
    "_static/styles/theme.css",
    "_static/pygments.css",
    "custom.css",
]



html_theme_options = {
    "repository_url": "https://github.com/ucsdmanorlab/Napari-VASCilia",
    "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "show_navbar_depth": 2,
    "logo": {
        "image_light": "_static/VASCilia_logo1.png",
        "image_dark": "_static/VASCilia_logo1.png",
    },
}
