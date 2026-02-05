# Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -j auto
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Determine the number of CPUs for parallel builds
NJOBS         ?= auto

# Put it first so that "make" without argument is like "make help".
help:
	@echo "SPD Learn Documentation Build Targets"
	@echo "======================================"
	@echo ""
	@echo "Primary targets:"
	@echo "  html           Build HTML documentation (runs all examples)"
	@echo "  html-noplot    Build HTML without running gallery examples (fast)"
	@echo "  html-fast      Build HTML in parallel, no plots, minimal output"
	@echo "  html-strict    Build HTML with warnings as errors (for CI)"
	@echo ""
	@echo "Development targets:"
	@echo "  livehtml       Start live-reload server (requires sphinx-autobuild)"
	@echo "  view           Open the built documentation in browser"
	@echo ""
	@echo "Cleaning targets:"
	@echo "  clean          Remove all build files"
	@echo "  clean-html     Clean and rebuild HTML"
	@echo "  clean-gallery  Remove only generated gallery files"
	@echo ""
	@echo "Quality targets:"
	@echo "  linkcheck      Check all external links for integrity"
	@echo "  doctest        Run doctests in documentation"
	@echo "  coverage       Check documentation coverage"
	@echo ""
	@echo "Other targets:"
	@echo "  latexpdf       Build PDF documentation via LaTeX"
	@echo "  text           Build plain text documentation"
	@echo "  man            Build manual pages"
	@echo ""
	@echo "Options:"
	@echo "  NJOBS=N        Number of parallel jobs (default: auto)"
	@echo "  SPHINXOPTS=... Additional Sphinx options"
	@echo ""
	@echo "Examples:"
	@echo "  make html-noplot              # Quick build without examples"
	@echo "  make html NJOBS=4             # Build with 4 parallel jobs"
	@echo "  make html-strict              # CI-style build with strict warnings"

.PHONY: help Makefile clean clean-html clean-gallery html html-noplot html-fast \
        html-strict livehtml view linkcheck doctest coverage latexpdf text man

# Standard HTML build (runs all examples)
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build HTML without running gallery examples (much faster)
html-noplot:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) \
		-D plot_gallery=0

# Fast build: no plots, parallel, quiet output
html-fast:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -j $(NJOBS) -q \
		-D plot_gallery=0 $(SPHINXOPTS) $(O)

# Strict build for CI: warnings as errors
html-strict:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going \
		$(SPHINXOPTS) $(O)

# Strict build without plots for faster CI checks
html-strict-noplot:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going \
		-D plot_gallery=0 $(SPHINXOPTS) $(O)

# Live reload server for development (requires: pip install sphinx-autobuild)
livehtml:
	@echo "Starting live-reload server..."
	@echo "Install with: pip install sphinx-autobuild"
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--ignore "*/generated/*" \
		--ignore "*/.ipynb_checkpoints/*" \
		-D plot_gallery=0 \
		$(SPHINXOPTS) $(O)

# Open built documentation in default browser
view:
	@if [ -f "$(BUILDDIR)/html/index.html" ]; then \
		echo "Opening documentation in browser..."; \
		open "$(BUILDDIR)/html/index.html" 2>/dev/null || \
		xdg-open "$(BUILDDIR)/html/index.html" 2>/dev/null || \
		start "$(BUILDDIR)/html/index.html" 2>/dev/null || \
		echo "Could not open browser. Documentation is at: $(BUILDDIR)/html/index.html"; \
	else \
		echo "Documentation not built yet. Run 'make html' first."; \
	fi

# Clean all build files
clean:
	@rm -rf $(BUILDDIR)/*
	@rm -rf $(SOURCEDIR)/generated
	@rm -rf $(SOURCEDIR)/gen_modules
	@echo "Build and generated directories cleaned."

# Clean and rebuild HTML
clean-html: clean html

# Clean only generated gallery examples (preserves other build artifacts)
clean-gallery:
	@rm -rf $(SOURCEDIR)/generated/auto_examples
	@rm -rf $(SOURCEDIR)/gen_modules/backreferences
	@echo "Gallery files cleaned."

# Check external links
linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Run doctests embedded in documentation
doctest:
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Check documentation coverage
coverage:
	@$(SPHINXBUILD) -M coverage "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Coverage report written to $(BUILDDIR)/coverage/"

# Build PDF via LaTeX
latexpdf:
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build plain text
text:
	@$(SPHINXBUILD) -M text "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build manual pages
man:
	@$(SPHINXBUILD) -M man "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option. $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
