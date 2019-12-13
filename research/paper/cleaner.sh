#infile='thesis.tex'
#outfile='draft.tex'

sed -e "s/{\\textbackslash}'\\\\{i\\\\}/\\'{i}/g" thesis.bib | grep -E -v '^\W*(doi|url|urldate|abstract|file|keywords|annote|note)' > thesis.bib.tmp
mv thesis.bib.tmp thesis.bib

#grep -E -v '^(%|[[:blank:]]*%|\\COMMENT|\includegraphics*)' "$infile" | fold -w80 -s > "$outfile"
