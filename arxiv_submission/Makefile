all: main.pdf

main.pdf: main.tex appendix.tex references.bib
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf *.fls *.fdb_latexmk *.synctex.gz

.PHONY: all clean
