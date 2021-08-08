
.PHONY: local-dev dataflow-prod docker-image

PYTHON=.venv/bin/python
PIP=.venv/bin/pip

PROJECT ?= "print-nanny-sandbox"
PRINT_NANNY_API_URL ?= "http://localhost:8000/api"
JOB_NAME ?= "sliding-window-health"
JOB_ID ?= $(shell gcloud dataflow jobs list --filter="name=$(JOB_NAME)" --status=active --format=json --region=$(GCP_REGION) | jq '.[].id')

PIPELINE ?= "print_nanny_dataflow.pipelines.sliding_window_health"
GIT_SHA ?= $(shell git rev-parse HEAD)
IMAGE ?= "gcr.io/${PROJECT}/print-nanny-dataflow:${GIT_SHA}"
BUCKET ?= "print-nanny-sandbox"
MAX_NUM_WORKERS ?= 1
MACHINE_TYPE ?= n1-standard-1
GCP_REGION ?= "us-central1"

mypy:
	mypy print_nanny_dataflow/

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
	rm -rf .mypy_cache

clean: clean-dist clean-pyc clean-build

docker-image:
	gcloud builds submit --tag $(IMAGE) --project $(PROJECT)

pytest:
	python -m pytest --disable-pytest-warnings

pytest-coverage:
	python -m pytest --cov=./ --cov-report=xml

direct:
	$(PYTHON) -m $(PIPELINE) \
	--runner DirectRunner \
	--loglevel INFO \
	--runtime_type_check \
	--bucket=$(BUCKET) \
	--job_name=$(JOB_NAME)

portable: docker-image
	$(PYTHON) -m $(PIPELINE) \
	--runner PortableRunner \
	--loglevel INFO \
	--job_endpoint=embed \
	--environment_type=DOCKER \
	--environment_config=$(IMAGE) \
	--sdk_location=container \
	​--setup_file=setup.py \
	--requirements_file=requirements.txt \
	--bucket=$(BUCKET)

sdist:
	python setup.py sdist

dataflow: clean docker-image sdist
	$(PYTHON) -m $(PIPELINE) \
	--runner DataflowRunner \
	--project=$(PROJECT) \
	--experiment=use_runner_v2 \
  	--worker_harness_container_image=$(IMAGE) \
  	--temp_location=gs://$(BUCKET)/dataflow/tmp \
	--job_name=$(JOB_NAME) \
	​--setup_file=$(PWD)/setup.py \
	--staging_location=gs://$(BUCKET)/dataflow/staging \
	--streaming \
	--max_num_workers=$(MAX_NUM_WORKERS) \
	--bucket=$(BUCKET) \
	--extra_package=dist/print-nanny-dataflow-0.1.0.tar.gz \
	--region=$(GCP_REGION) \
	--save_main_session \
	--machine_type=$(MACHINE_TYPE)


dataflow-cancel:
	gcloud dataflow jobs cancel $(JOB_ID) --region=$(GCP_REGION)

dataflow-clean: dataflow-cancel dataflow

dataflow-update: clean docker-image sdist
	$(PYTHON) -m $(PIPELINE) \
	--runner DataflowRunner \
	--project=$(PROJECT) \
	--experiment=use_runner_v2 \
	--worker_harness_container_imag=$(IMAGE) \
	--temp_location=gs://$(BUCKET)/dataflow/tmp \
	--job_name=$(JOB_NAME) \
	​--setup_file=$(PWD)/setup.py \
	--staging_location=gs://$(BUCKET)/dataflow/staging \
	--streaming \
	--max_num_workers=$(MAX_NUM_WORKERS) \
	--bucket=$(BUCKET) \
	--extra_package=dist/print-nanny-dataflow-0.1.0.tar.gz \
	--region=$(GCP_REGION) \
	--update \
	--save_main_session

lint:
	$(PYTHON) -m black setup.py print_nanny_dataflow conftest.py tests

install-git-hooks:
	cp -a hooks/. .git/hooks/