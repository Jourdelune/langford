# Le PDF est versionne : un lecteur ne devrait pas avoir a installer TeX Live
# pour lire un papier. La source l'est aussi, pour qu'il puisse le refaire.
#   make        -> langford31.pdf
#   make clean  -> supprime les artefacts, garde le PDF
langford31.pdf: langford31.tex
	pdflatex -interaction=nonstopmode -halt-on-error langford31.tex
	pdflatex -interaction=nonstopmode langford31.tex   # references croisees
clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz
.PHONY: clean
