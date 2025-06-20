create-env:
	conda create -n parkinson python=3.9 ipython --file requirements.txt

module:
	pip install -e .

report:
	pandoc report/report.md -o report/report.pdf -N --citeproc --bibliography report/refs.bib 