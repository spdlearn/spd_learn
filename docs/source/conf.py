import spd_learn


project = "spd_learn"
copyright = "2025, Bruno Aristimunha"
author = "Bruno Aristimunha"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.linkcode",
    "sphinx.ext.graphviz",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_sitemap",  # SEO: Generate sitemap.xml
]

# -- BibTeX configuration -----------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# Plot directive configuration
plot_include_source = True
plot_html_show_source_link = True
plot_formats = [("png", 150)]
plot_html_show_formats = False
plot_rcparams = {
    "figure.figsize": (10, 5),
    "figure.dpi": 100,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Autosummary configuration ------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False

# -- Autodoc configuration ----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,  # Don't document inherited members from nn.Module
    "exclude-members": "__weakref__",
}

# Suppress warnings for inherited members that don't have stub files
# and undefined labels from PyTorch autograd docstrings
suppress_warnings = [
    "autosummary",
    "ref.ref",
    "bibtex.duplicate_citation",
    "bibtex.duplicate_label",
]


# -- Numpydoc configuration ---------------------------------------------------
numpydoc_show_class_members = False  # Don't show members in class docstring
numpydoc_class_members_toctree = False

# -- Intersphinx configuration ------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# GitHub repository info for edit buttons and Colab integration
github_user = "spdlearn"
github_repo = "spd_learn"
github_version = "main"


# -- MathJax configuration (notation macros) ---------------------------------
# Define LaTeX-style macros for consistent notation across all documentation.
# Usage: :math:`\spd` renders as S^n_{++}, :math:`\dairm{A}{B}` renders distance.
# Reference: docs/source/notation.rst

mathjax3_config = {
    "tex": {
        "macros": {
            # ===== Spaces and Manifolds =====
            "spd": r"\mathcal{S}^n_{++}",  # SPD manifold
            "sym": r"\text{Sym}(n)",  # Symmetric matrices
            "syms": r"\mathcal{S}^n",  # Symmetric matrices (alternative)
            "reals": r"\mathbb{R}",  # Real numbers
            "choleskyspace": r"\mathcal{L}_+",  # Cholesky space
            "manifold": r"\mathcal{M}",  # Generic manifold
            # ===== Groups =====
            "gl": r"\text{GL}(n)",  # General linear group
            "stiefel": [r"\text{St}(#1, #2)", 2],  # Stiefel manifold St(n,k)
            # ===== Tangent Spaces =====
            "tangent": [r"T_{#1} \mathcal{M}", 1],  # Tangent space at P
            "tangentspd": [r"T_{#1} \mathcal{S}^n_{++}", 1],  # Tangent space on SPD
            # ===== Maps =====
            "Exp": [r"\text{Exp}_{#1}", 1],  # Riemannian exponential
            "Log": [r"\text{Log}_{#1}", 1],  # Riemannian logarithm
            "logchol": r"\log_{\text{chol}}",  # Log-Cholesky logarithm
            "expchol": r"\exp_{\text{chol}}",  # Log-Cholesky exponential
            "tril": [r"\text{tril}(#1)", 1],  # Lower triangular
            # ===== Distance Functions =====
            "dairm": [r"d_{\text{AIRM}}(#1, #2)", 2],  # AIRM distance
            "dlem": [r"d_{\text{LEM}}(#1, #2)", 2],  # Log-Euclidean distance
            "dbw": [r"d_{\text{BW}}(#1, #2)", 2],  # Bures-Wasserstein distance
            "dlcm": [r"d_{\text{LCM}}(#1, #2)", 2],  # Log-Cholesky distance
            # ===== Inner Products =====
            "gairm": [r"g^{\text{AIRM}}_{#1}", 1],  # AIRM inner product at P
            "glem": [r"g^{\text{LEM}}_{#1}", 1],  # LEM inner product at P
            "gbw": [r"g^{\text{BW}}_{#1}", 1],  # BW inner product at P
            "glcm": [r"g^{\text{LCM}}_{#1}", 1],  # LCM inner product at P
            # ===== Common Operations =====
            "tr": r"\text{tr}",  # Trace
            "diag": r"\text{diag}",  # Diagonal
            "frob": [r"\| #1 \|_F", 1],  # Frobenius norm
            "frobinner": [r"\langle #1, #2 \rangle_F", 2],  # Frobenius inner product
            "lyap": [r"\mathcal{L}_{#1}", 1],  # Lyapunov operator
            # ===== Layer Operations =====
            "reeig": r"\text{ReEig}",  # ReEig layer
            "logeig": r"\text{LogEig}",  # LogEig layer
            "expeig": r"\text{ExpEig}",  # ExpEig layer
            "frechet": r"\mathcal{G}",  # Fréchet mean
            "geomean": r"G",  # Geometric mean
            # ===== Special Symbols =====
            "In": r"I_n",  # Identity matrix (n x n)
            "I": r"I",  # Identity matrix
            "transpose": r"^\top",  # Transpose
            # ===== Log-Euclidean Operations =====
            "lemult": r"\odot",  # Log-Euclidean multiplication
            "lescalar": r"\circledast",  # Log-Euclidean scalar multiplication
        }
    }
}

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static/"]
html_css_files = ["custom.css"]

html_sidebars = {
    "api": [],
    "installation": [],
    "user_guide": [],
}

html_logo = "_static/spd_learn.png"
html_favicon = "_static/spd_learn.png"

# -- Project information -----------------------------------------------------
from datetime import datetime, timezone


project = "SPD Learn"
td = datetime.now(tz=timezone.utc)

# We need to triage which date type we use so that incremental builds work
# (Sphinx looks at variable changes and rewrites all files if some change)
copyright = f"2024-{td.year}, {project} Developers"  # noqa: E501

author = f"{project} developers"


release = spd_learn.__version__
# The full version, including alpha/beta/rc tags.
version = ".".join(release.split(".")[:2])

exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
switcher_version_match = "dev" if release.endswith("dev0") else version


html_theme_options = {
    # SEO: Add proper meta description
    "analytics": {"google_analytics_id": ""},  # Add GA ID if available
    "icon_links_label": "External Links",  # for screen reader
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "collapse_navigation": False,
    "header_links_before_dropdown": 6,
    "navigation_depth": 6,
    "show_toc_level": 1,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/spdlearn/spd_learn",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/spd_learn/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "logo": {
        "text": "<strong>SPD</strong> Learn",
        "image_light": "_static/spd_learn.png",
        "image_dark": "_static/spd_learn.png",
        "alt_text": "SPD Learn Logo",
    },
    # Secondary sidebar items - includes Colab launcher for examples
    # Note: We only specify auto_examples pattern; other pages use default sidebar
    "secondary_sidebar_items": {
        "user_guide": [],  # Remove secondary sidebar from user guide
        "generated/auto_examples/**": [
            "page-toc",
            "sg_download_links",
            "sg_launcher_links",
        ],
    },
    "footer_start": ["copyright"],
}

# HTML context for templates (including Colab integration)
html_context = {
    "github_user": github_user,
    "github_repo": github_repo,
    "github_version": github_version,
    "doc_path": "docs/source",
    # Colab launcher for Sphinx-Gallery examples (see _templates/sg_launcher_links.html)
    "colab_repo": f"{github_user}/{github_repo}",
    "colab_branch": "gh-pages",
    # Notebooks are deployed at root of gh-pages
    "colab_docs_path": "",
}

from sphinx_gallery.sorting import ExplicitOrder


sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["generated/auto_examples"],
    "parallel": True,  # Enable parallel execution for faster builds
    "nested_sections": True,  # Enable nested galleries for visualizations subdirectory
    # Point 3: Image optimization - compress images and reduce thumbnail size
    "compress_images": ("images", "thumbnails"),
    "thumbnail_size": (400, 280),  # Smaller thumbnails for faster loading
    # Order: tutorials first, then visualizations, then applied examples
    "subsection_order": ExplicitOrder(
        [
            "../../examples/tutorials",
            "../../examples/visualizations",
            "../../examples/applied_examples",
        ]
    ),
    "backreferences_dir": "gen_modules/backreferences",
    "inspect_global_variables": True,
    "doc_module": ("spd_learn", "numpy", "scipy", "matplotlib"),
    "reference_url": {"spd_learn": None},
    "matplotlib_animations": (
        True,
        "jshtml",
    ),  # Embed animations as JavaScript (no ffmpeg needed)
    # Include both plot_* files and tutorial_* files
    "filename_pattern": r"/(plot_|tutorial_)",
    "ignore_pattern": r"(__init__|spd_visualization_utils)\.py",
    # Show signature link template (includes Colab launcher)
    "show_signature": False,
    # First cell in generated notebooks (for Colab compatibility)
    "first_notebook_cell": (
        "# SPD Learn Example\n"
        "# ==================\n"
        "#\n"
        "# First, install the required packages:\n"
        "\n"
        "!uv pip install -q spd_learn\n"
    ),
    # Last cell in generated notebooks
    "last_notebook_cell": (
        "# Cleanup\nimport matplotlib.pyplot as plt\nplt.close('all')\n"
    ),
}

# -- SEO and Build Size Optimization ------------------------------------------

# Point 2: Disable source file duplication (~87 MB saved)
html_show_sourcelink = False
html_copy_source = False

# Point 6: SEO - Add canonical URLs
html_baseurl = "https://spdlearn.org/"

# Point 7: Prevent duplicate images - don't copy extra files
html_extra_path = []

# -- Linkcheck configuration -------------------------------------------------

# -- LaTeX configuration -----------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "a4paper",
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    "preamble": r"""
\usepackage{amsmath,amsfonts,amssymb,amsthm}

% ===== SPD Learn Notation Macros =====
% These match the MathJax macros defined in mathjax3_config for consistency.
% Reference: docs/source/notation.rst

% Spaces and Manifolds
\newcommand{\spd}{\mathcal{S}^n_{++}}
\newcommand{\sym}{\text{Sym}(n)}
\newcommand{\syms}{\mathcal{S}^n}
\newcommand{\reals}{\mathbb{R}}
\newcommand{\choleskyspace}{\mathcal{L}_+}
\newcommand{\manifold}{\mathcal{M}}

% Groups
\newcommand{\gl}{\text{GL}(n)}
\newcommand{\stiefel}[2]{\text{St}(#1, #2)}

% Tangent Spaces
\newcommand{\tangent}[1]{T_{#1} \mathcal{M}}
\newcommand{\tangentspd}[1]{T_{#1} \mathcal{S}^n_{++}}

% Maps
\newcommand{\Exp}[1]{\text{Exp}_{#1}}
\newcommand{\Log}[1]{\text{Log}_{#1}}
\newcommand{\logchol}{\log_{\text{chol}}}
\newcommand{\expchol}{\exp_{\text{chol}}}
\newcommand{\tmark}[1]{\text{tril}(#1)}

% Distance Functions
\newcommand{\dairm}[2]{d_{\text{AIRM}}(#1, #2)}
\newcommand{\dlem}[2]{d_{\text{LEM}}(#1, #2)}
\newcommand{\dbw}[2]{d_{\text{BW}}(#1, #2)}
\newcommand{\dlcm}[2]{d_{\text{LCM}}(#1, #2)}

% Inner Products
\newcommand{\gairm}[1]{g^{\text{AIRM}}_{#1}}
\newcommand{\glem}[1]{g^{\text{LEM}}_{#1}}
\newcommand{\gbw}[1]{g^{\text{BW}}_{#1}}
\newcommand{\glcm}[1]{g^{\text{LCM}}_{#1}}

% Common Operations
\newcommand{\trop}{\text{tr}}
\newcommand{\diagop}{\text{diag}}
\newcommand{\frob}[1]{\| #1 \|_F}
\newcommand{\frobinner}[2]{\langle #1, #2 \rangle_F}
\newcommand{\lyap}[1]{\mathcal{L}_{#1}}

% Layer Operations
\newcommand{\reeig}{\text{ReEig}}
\newcommand{\logeig}{\text{LogEig}}
\newcommand{\expeig}{\text{ExpEig}}
\newcommand{\frechet}{\mathcal{G}}
\newcommand{\geomean}{G}

% Special Symbols
\newcommand{\In}{I_n}
\newcommand{\transpose}{^\top}

% Log-Euclidean Operations
\newcommand{\lemult}{\odot}
\newcommand{\lescalar}{\circledast}
% Handle common math Unicode characters
\usepackage{newunicodechar}
\newunicodechar{∈}{\ensuremath{\in}}
\newunicodechar{∑}{\ensuremath{\sum}}
\newunicodechar{∏}{\ensuremath{\prod}}
\newunicodechar{⊗}{\ensuremath{\otimes}}
\newunicodechar{⊕}{\ensuremath{\oplus}}
\newunicodechar{ℝ}{\ensuremath{\mathbb{R}}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≠}{\ensuremath{\neq}}
\newunicodechar{λ}{\ensuremath{\lambda}}
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{γ}{\ensuremath{\gamma}}
\newunicodechar{θ}{\ensuremath{\theta}}
\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{μ}{\ensuremath{\mu}}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{↔}{\ensuremath{\leftrightarrow}}
\newunicodechar{≡}{\ensuremath{\equiv}}
\newunicodechar{▼}{\ensuremath{\blacktriangledown}}
\newunicodechar{┌}{\ensuremath{\ulcorner}}
\newunicodechar{┐}{\ensuremath{\urcorner}}
\newunicodechar{└}{\ensuremath{\llcorner}}
\newunicodechar{┘}{\ensuremath{\lrcorner}}
\newunicodechar{─}{\textminus}
\newunicodechar{│}{|}
\newunicodechar{┴}{\ensuremath{\perp}}
\newunicodechar{┬}{\ensuremath{\top}}
\newunicodechar{├}{|}
\newunicodechar{┤}{|}
\newunicodechar{┼}{+}
\newunicodechar{►}{\ensuremath{\blacktriangleright}}
\newunicodechar{◄}{\ensuremath{\blacktriangleleft}}
\newunicodechar{▲}{\ensuremath{\blacktriangle}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{←}{\ensuremath{\leftarrow}}
\newunicodechar{●}{\ensuremath{\bullet}}
\newunicodechar{○}{\ensuremath{\circ}}
\newunicodechar{■}{\ensuremath{\blacksquare}}
\newunicodechar{□}{\ensuremath{\square}}
\newunicodechar{τ}{\ensuremath{\tau}}
\newunicodechar{φ}{\ensuremath{\phi}}
\newunicodechar{ψ}{\ensuremath{\psi}}
\newunicodechar{ω}{\ensuremath{\omega}}
\newunicodechar{Φ}{\ensuremath{\Phi}}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{Σ}{\ensuremath{\Sigma}}
\newunicodechar{Ω}{\ensuremath{\Omega}}
\newunicodechar{Π}{\ensuremath{\Pi}}
\newunicodechar{Θ}{\ensuremath{\Theta}}
\newunicodechar{Ψ}{\ensuremath{\Psi}}
\newunicodechar{Ξ}{\ensuremath{\Xi}}
\newunicodechar{Γ}{\ensuremath{\Gamma}}
\newunicodechar{Λ}{\ensuremath{\Lambda}}
\newunicodechar{Υ}{\ensuremath{\Upsilon}}
\newunicodechar{ζ}{\ensuremath{\zeta}}
\newunicodechar{η}{\ensuremath{\eta}}
\newunicodechar{κ}{\ensuremath{\kappa}}
\newunicodechar{ν}{\ensuremath{\nu}}
\newunicodechar{ξ}{\ensuremath{\xi}}
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{υ}{\ensuremath{\upsilon}}
\newunicodechar{χ}{\ensuremath{\chi}}
""",
    # Latex figure (float) alignment
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        "index",
        "spd_learn.tex",
        "SPD Learn Documentation",
        "SPD Learn developers",
        "manual",
    ),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = "_static/spd_learn.png"

# Patterns for URLs to ignore during linkcheck
linkcheck_ignore = [
    # GitHub repo not created yet
    r"https://github\.com/spdlearn/spd_learn.*",
    # SIAM blocks automated requests
    r"https://doi\.org/10\.1137/.*",
    r"https://epubs\.siam\.org/.*",
]

# Timeout for linkcheck requests (seconds)
linkcheck_timeout = 30

# Number of retries for linkcheck
linkcheck_retries = 2


# -- Linkcode configuration --------------------------------------------------
def linkcode_resolve(domain, info):
    """Resolve GitHub links for source code.

    This function is required by sphinx.ext.linkcode.
    """
    if domain != "py":
        return None
    if not info["module"]:
        return None

    filename = info["module"].replace(".", "/")
    return f"https://github.com/{github_user}/{github_repo}/blob/{github_version}/src/{filename}.py"
