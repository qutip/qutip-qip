# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import pathlib
import sys
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'qutip_qip'
copyright = '2021, QuTiP Community'
author = 'QuTiP Community'


def qutip_qip_version():
    """ Retrieve the qutip-qip version from ``../../VERSION``.
    """
    src_folder_root = pathlib.Path(__file__).absolute().parent.parent.parent
    version = src_folder_root.joinpath(
        "VERSION"
    ).read_text().strip()
    return version


# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
# The full version, including alpha/beta/rc tags.
release = qutip_qip_version()
# The short X.Y version.
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.mathjax',
              'matplotlib.sphinxext.plot_directive',
              'sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.autosummary',
              'numpydoc',
              'sphinx.ext.extlinks',
              'sphinx.ext.viewcode',
              'sphinx.ext.ifconfig',
              'sphinx.ext.napoleon',
              'sphinxcontrib.bibtex',
              'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# -- Options for LaTeX output ---------------------------------------------

# AJGP 2017-01-8: Switching to manual, ditching the preamble

# latex_header = open('latex_output_files/latex_preamble.tex', 'r+')
# PREAMBLE = latex_header.read();

latex_elements = {
                  'papersize':'a4paper',
                  'pointsize':'10pt',
                  'classoptions': '',
                  'babel': '\\usepackage[english]{babel}',
                  'fncychap' : '',
                  'figure_align': 'H',
#                  'preamble': PREAMBLE
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  ('index', 'qutip.tex', project, author, 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = 'figures/logo.png'

# Sometimes make might suggest setting this to False.
# It screws a few things up if you do - don't be tempted.
latex_keep_old_macro_names=True

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = True

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True

# -- Doctest Setup ---------------------------------------

os_nt = False
if os.name == "nt":
    os_nt = True

doctest_global_setup = '''
from pylab import *
from qutip import *
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.set_printoptions(precision=5)
os_nt = {}
'''.format(os_nt)

# -- Options for plot directive ---------------------------------------

plot_working_directory = "./"
plot_pre_code = """
from pylab import *
from scipy import *
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
"""
plot_include_source = True
plot_html_show_source_link = False

# -- Options for numpydoc ---------------------------------------

numpydoc_show_class_members = False
napoleon_numpy_docstring = True
napoleon_use_admonition_for_notes = True

# -- Options for api doc ---------------------------------------
# autosummary_generate can be turned on to automatically generate files
# in the apidoc folder. This is particularly useful for modules with 
# lots of functions/classes like qutip_qip.operations. However, pay
# attention that some api docs files are adjusted manually for better illustration
# and should not be overwritten.
autosummary_generate = False
autosummary_imported_members = True

# -- Options for biblatex ---------------------------------------

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'

# -- Options for intersphinx ---------------------------------------

intersphinx_mapping = {
    'qutip': ('https://qutip.readthedocs.io/en/stable/', None),
}