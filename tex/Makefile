OO=out
all: main.pdf

main.pdf: $(OO)/main.pdf
	cp $< $@

$(OO)/main.pdf: $(if $(wildcard $(OO)/main.aux),main_filtered.bib,)

main_filtered.bib: $(OO)/main.aux $(HOME)/wrk/group-bib/shared.bib
	nohup aux2bib $< 2>/dev/null | tail -n +4 > $@

media/%:
	$(MAKE) -C media/ $*

include pdflatex.mk
