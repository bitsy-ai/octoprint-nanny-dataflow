
.PHONY: local-dev dataflow-prod

PYTHON=.venv/bin/python
PIP=.venv/bin/pip

PROJECT ?= "print-nanny-sandbox"
PRINT_NANNY_API_URL ?= "http://localhost:8000/api"
PIPELINE ?= "print_nanny_dataflow.pipelines.sliding_window_health"
IMAGE ?= "gcr.io/${PROJECT}/print-nanny-dataflow:$(shell git rev-parse HEAD)"
BUCKET ?= "print-nanny-sandbox"

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
	gcloud builds submit --tag $(IMAGE)

direct:
	$(PYTHON) -m $(PIPELINE) \
	--runner DirectRunner \
	--loglevel INFO \
	--api-url=$(PRINT_NANNY_API_URL) \
	--api-token=$$PRINT_NANNY_API_TOKEN \
	--direct_num_workers=12 \
	--runtime_type_check

portable:
	$(PYTHON) -m $(PIPELINE) \
	--runner PortableRunner \
	--loglevel INFO \
	--api-url=$(PRINT_NANNY_API_URL) \
	--api-token=$$PRINT_NANNY_API_TOKEN \
	--job_endpoint=embed \
	--environment_type=DOCKER \
	--environment_config=$(IMAGE) \
	--sdk_location=container \
	​--setup_file=setup.py \
	--requirements_file=requirements.txt

dataflow:
	$(PYTHON) -m $(PIPELINE) \
	--runner DataflowRunner \
	--api-url=$(PRINT_NANNY_API_URL) \
	--api-token=$$PRINT_NANNY_API_TOKEN \
	--project=$(PROJECT) \
	--experiment=use_runner_v2 \
	--worker_harness_container_image=$(IMAGE) \
	--temp_location=gs://$(BUCKET)/dataflow/tmp \
	--staging_location=gs://$(BUCKET)/dataflow/staging

lint:
	$(PYTHON) -m black setup.py print_nanny_dataflow conftest.py tests

install-git-hooks:
	cp -a hooks/. .git/hooks/