
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

docker-image:
	gcloud builds submit --tag gcr.io/print-nanny-sandbox/print-nanny-dataflow:$(shell git rev-parse HEAD)

alerts-local-dev:
	$(PYTHON) print_nanny_dataflow/pipelines/video_render.py \
	--runner DirectRunner \
	--loglevel INFO \
	--api-url="http://localhost:8000/api" \
	--api-token=$$PRINT_NANNY_API_TOKEN \
	--runtime_type_check

health-local-dev:
	$(PYTHON) print_nanny_dataflow/pipelines/sliding_window_health.py \
	--runner DirectRunner \
	--loglevel INFO \
	--api-url="http://localhost:8000/api" \
	--api-token=$$PRINT_NANNY_API_TOKEN \
	--direct_num_workers=12 \
	--runtime_type_check

health-local-portable:
	$(PYTHON) -m print_nanny_dataflow.pipelines.sliding_window_health \
	--runner PortableRunner \
	--loglevel INFO \
	--api-url="http://localhost:8000/api" \
	--api-token=$$PRINT_NANNY_API_TOKEN \
	--job_endpoint=embed \
	--environment_type=DOCKER \
	--environment_config=$(ARGS) \
	--sdk_location=container \
	​--setup_file=setup.py \
	--requirements_file=requirements.txt

dataflow-prod:
	$(PYTHON) print_nanny_dataflow/pipelines/sliding_window_health.py \
	--runner DataflowRunner \
	--topic projects/print-nanny/topics/bounding-boxes-prod \
	--api-url="http://localhost:8000/api" \
	--api-token=$$PRINT_NANNY_API_TOKEN \
	--window 300 \
	--sink gs://print-nanny-prod/dataflow/bounding-box-events/windowed \
	--loglevel INFO 


lint:
	$(PYTHON) -m black setup.py print_nanny_dataflow conftest.py tests

install-git-hooks:
	cp -a hooks/. .git/hooks/