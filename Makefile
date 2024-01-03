run:
	python3 a4_program.py

help:
	python3 a4_program.py --help

install:
	pip3 install tensorflow tdqm argparse statistics 

linear:
	python3 a4_program.py --models linear 
	python3 a4_program.py --models linear --lrate 0.05
	python3 a4_program.py --models linear --lrate 0.005
