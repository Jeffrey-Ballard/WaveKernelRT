PYTHON ?= python3

.PHONY: docs docs-build docs-serve docs-pdf offline-bundle clean

docs: docs-build

docs-build:
	$(PYTHON) -m mkdocs build --strict

docs-serve:
	$(PYTHON) -m mkdocs serve

docs-pdf:
	pandoc docs/concepts/conventions.md docs/concepts/data-shapes.md -o WaveKernelRT_Coordinate_and_Unit_Conventions.pdf \
	--pdf-engine=xelatex -V geometry:margin=1in -V fontsize=11pt
	
	pandoc README.md -o WaveKernelRT_README.pdf --pdf-engine=xelatex -V geometry:margin=1in -V fontsize=11pt

offline-bundle:
	$(PYTHON) tools/build_offline_bundle.py

clean:
	rm -f *pdf
