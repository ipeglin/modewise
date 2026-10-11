"""Sphinx configuration for the modewise documentation (docs/README.md explains the build)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

project = "modewise"
author = "Ian Philip Eglin"
copyright = "Ian Philip Eglin. BSD-3-Clause"

extensions = ["myst_parser", "autoapi.extension", "sphinx_copybutton", "sphinx.ext.intersphinx", "sphinx_gallery.gen_gallery"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "_extra", "README.md", "requirements.txt", "doctests", "auto_examples/*.ipynb", "auto_examples/*.py", "auto_examples/*.zip", "auto_examples/*.json"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
# numpy documents these as data and attributes, not classes, so the class role cannot resolve them.
nitpick_ignore = [
    ("py:class", name)
    for name in ("numpy.typing.NDArray", "numpy.typing.ArrayLike", "numpy.float64", "numpy.bool_", "numpy.uint64")
]

# The Python API comes from the type stubs, the single source of its documentation; nothing is
# imported, so the docs build without compiling the Rust extension.
autoapi_dirs = [str(ROOT / "drivers" / "python" / "python" / "modewise")]
autoapi_file_patterns = ["*.pyi", "*.py"]
autoapi_root = "reference/python"
autoapi_add_toctree_entry = False
autoapi_options = ["members", "undoc-members", "show-module-summary", "imported-members"]
autoapi_python_class_content = "both"

html_theme = "pydata_sphinx_theme"
html_title = "modewise"
html_static_path = ["assets"]
html_extra_path = ["_extra"]
html_logo = "assets/modewise-monochrome-light-transparent.svg"
html_theme_options = {
    "logo": {
        "image_light": "assets/modewise-monochrome-light-transparent.svg",
        "image_dark": "assets/modewise-monochrome-dark-transparent.svg",
        "alt_text": "modewise",
    },
    "navigation_with_keys": False,
    "show_toc_level": 2,
}
html_favicon = "assets/modewise-symbol-monochrome-light-transparent.svg"

# The gallery runs every example on each build, so code and figures always match modewise.
sphinx_gallery_conf = {
    "examples_dirs": str(ROOT / "examples" / "gallery"),
    "gallery_dirs": "auto_examples",
    "filename_pattern": "/plot_",
    "download_all_examples": False,
    "remove_config_comments": True,
    "write_computation_times": False,
}
