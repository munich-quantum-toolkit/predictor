# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sphinx configuration file."""

from __future__ import annotations

import warnings
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import pybtex.plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.template import field, href

ROOT = Path(__file__).parent.parent.resolve()


try:
    version = metadata.version("mqt.predictor")
except ModuleNotFoundError:
    msg = (
        "Package should be installed to produce documentation! "
        "Assuming a modern git archive was used for version discovery."
    )
    warnings.warn(msg, stacklevel=1)

    from setuptools_scm import get_version

    version = get_version(root=str(ROOT), fallback_root=ROOT)

# Filter git details from version
release = version.split("+")[0]
if TYPE_CHECKING:
    from pybtex.database import Entry
    from pybtex.richtext import HRef

project = "MQT Predictor"
author = "Chair for Design Automation, Technical University of Munich"
language = "en"
project_copyright = "2023, Chair for Design Automation, Technical University of Munich"
# -- General configuration ---------------------------------------------------

master_doc = "index"

templates_path = ["_templates"]
html_css_files = ["custom.css"]

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinxext.opengraph",
]

pygments_style = "colorful"

add_module_names = False

modindex_common_prefix = ["mqt.predictor."]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest/", None),
    "qiskit": ("https://qiskit.org/documentation/", None),
    "mqt": ("https://mqt.readthedocs.io/en/latest/", None),
}

nbsphinx_execute = "never"
highlight_language = "python3"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=200",
]
nbsphinx_kernel_name = "python3"

autosectionlabel_prefix_document = True

exclude_patterns = ["_build", "build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store", ".env"]


class CDAStyle(UnsrtStyle):
    """Custom style for including PDF links."""

    def format_url(self, _e: Entry) -> HRef:
        """Format URL field as a link to the PDF."""
        url = field("url", raw=True)
        return href()[url, "[PDF]"]


pybtex.plugin.register_plugin("pybtex.style.formatting", "cda_style", CDAStyle)

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "cda_style"

copybutton_prompt_text = r"(?:\(venv\) )?(?:\[.*\] )?\$ "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"

autosummary_generate = True


typehints_use_rtype = False
napoleon_use_rtype = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "mqt_dark.png",
    "dark_logo": "mqt_light.png",
    "source_repository": "https://github.com/munich-quantum-toolkit/predictor/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}
