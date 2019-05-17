infile='from_thesis.tex'
outfile='draft.tex'

sed -e "s/{\\textbackslash}'\\\\{i\\\\}/\\'{i}/g" from_thesis.bib | grep -E -v '^\W*(doi|url|urldate|abstract|file|keywords|annote|note)' > from_thesis.bib.tmp
mv from_thesis.bib.tmp thesis.bib

#grep -E -v '^(%|[[:blank:]]*%|\\COMMENT|\includegraphics*)' "$infile" | fold -w80 -s > "$outfile"
