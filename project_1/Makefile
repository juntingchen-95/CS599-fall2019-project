test_set_result.csv: main.py
	python main.py dataset/test_set.csv

training_set_result.csv: main.py
	python main.py dataset/training_set.csv

result_figure.pdf：generate_figure.R
	R CMD BATCH generate_figure.R
