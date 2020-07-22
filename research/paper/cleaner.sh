#infile='thesis.tex'
#outfile='draft.tex'

sed -e "s/{\\textbackslash}'\\\\{i\\\\}/\\'{i}/g" draft.bib | grep -E -v '^\W*(doi|url|urldate|abstract|file|keywords|annote|note)' > draft.bib.tmp
mv draft.bib.tmp draft.bib

#grep -E -v '^(%|[[:blank:]]*%|\\COMMENT|\includegraphics*)' "$infile" | fold -w80 -s > "$outfile"
