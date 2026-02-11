# Sphinx Makefile (run from docs/: make html)
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

html:
	$(SPHINXBUILD) -b html -c . $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo "Build finished. Open $(BUILDDIR)/html/index.html"

clean:
	rm -rf $(BUILDDIR)
