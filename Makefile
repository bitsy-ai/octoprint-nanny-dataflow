
.PHONY: local-dev dataflow-prod

PYTHON=.venv/bin/python
PIP=.venv/bin/pip


clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-dist:
	rm -rf dist/

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean: clean-dist clean-pyc clean-build

local-dev:
	$(PYTHON) print_nanny_dataflow/sliding_windowed_health.py \
	--runner DirectRunner \
	--loglevel INFO \
	--api-url="http://localhost:8000/api" \
	--api-token=$$PRINT_NANNY_API_TOKEN

dataflow-prod:
	$(PYTHON) windowed_tfrecords.py \
	--runner DataflowRunner \
	--topic projects/print-nanny/topics/bounding-boxes-prod \
	--window 300 \
	--sink gs://print-nanny-prod/dataflow/bounding-box-events/windowed \
	--loglevel INFO


lint:
	$(PYTHON) -m black setup.py print_nanny_dataflow conftest.py tests

install-git-hooks:
	cp -a hooks/. .git/hooks/