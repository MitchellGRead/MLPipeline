SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv           	 : creates a virtual environment."
	@echo "dev            	 : creates a virtual environment and sets up dev dependencies."
	@echo "docs           	 : creates a virtual environment and sets up doc dependencies."
	@echo "style          	 : executes style formatting."
	@echo "clean          	 : cleans all unnecessary files."
	@echo "servemlflow    	 : serves the mlflow dashboard."
	@echo "precommitall   	 : runs pre-commit hooks on all files."
	@echo "gentestcoverage   : generate unit test coverage report."
	@echo "runallcodetest    : run all code tests"
	@echo "rundatatest       : run data tests with data location pointing to ./data/labeled_projects"
	@echo "runmodeltest      : run model tests a provided run id. ex: make run=<id> runmodeltest"

.ONESHELL:
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install pip setuptools wheel && \
	python3 -m pip install -e .

.ONESHELL:
dev:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install pip setuptools wheel && \
	python3 -m pip install -e ".[dev]"
	pre-commit install
	pre-commit autoupdate

.ONESHELL:
docs:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install pip setuptools wheel && \
	python3 -m pip install -e ".[docs]"

.PHONY: style
style:
	black .
	-flake8
	isort .

.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage

.PHONY: servemlflow
servemlflow:
	mlflow server -h 0.0.0.0 -p 8080 --backend-store-uri ./stores/model

.PHONY: precommitall
precommitall:
	pre-commit run --all-files

.PHONY: gentestcoverage
gentestcoverage:
	pytest tests/ --cov tagifai --cov-report html --disable-warnings && \
	coverage report -m

.PHONY: runallcodetest
runallcodetest:
	pytest tests/code

.PHONY: rundatatest
rundatatest:
	pytest --dataset-loc="./data/labeled_projects.csv" tests/data --verbose --disable-warnings

.PHONY: runmodeltest
runmodeltest:
	pytest --run-id=$(run) tests/model --verbose --disable-warnings
