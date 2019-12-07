result.csv: mmit.py targets.csv
	python mmit.py

result_figure_1.pdf result_figure_2.pdf：generate_figure.R result.csv
	R CMD BATCH generate_figure.R
