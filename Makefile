
.PHONY: local-dev dataflow-prod docker-image

PYTHON=.venv/bin/python
PIP=.venv/bin/pip

PROJECT ?= "print-nanny-sandbox"
PRINT_NANNY_API_URL ?= "http://localhost:8000/api"
JOB_NAME ?= "sliding-window-health"
PIPELINE ?= "print_nanny_dataflow.pipelines.sliding_window_health"
GIT_SHA ?= $(shell git rev-parse HEAD)
IMAGE ?= "gcr.io/${PROJECT}/print-nanny-dataflow:${GIT_SHA}"
BUCKET ?= "print-nanny-sandbox"
MAX_NUM_WORKERS ?= 2

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

dist/$(GIT_SHA).image:dist/%.image:
	gcloud builds submit --tag $(IMAGE)
	touch dist/$(GIT_SHA).image

docker-image: dist/$(GIT_SHA).image

direct:
	$(PYTHON) -m $(PIPELINE) \
	--runner DirectRunner \
	--loglevel INFO \
	--api-url=$(PRINT_NANNY_API_URL) \
	--api-token=$$PRINT_NANNY_API_TOKEN \
	--direct_num_workers=12 \
	--runtime_type_check \
	--bucket=$(BUCKET)

portable: docker-image
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
	--requirements_file=requirements.txt \
	--bucket=$(BUCKET)

sdist:
	python setup.py sdist

dataflow: clean docker-image sdist
	$(PYTHON) -m $(PIPELINE) \
	--runner DataflowRunner \
	--api-url=$(PRINT_NANNY_API_URL) \
	--api-token=$$PRINT_NANNY_API_TOKEN \
	--project=$(PROJECT) \
	--experiment=use_runner_v2 \
	--sdk_container_image=$(IMAGE) \
	--temp_location=gs://$(BUCKET)/dataflow/tmp \
	--job_name=$(JOB_NAME) \
	​--setup_file=$(PWD)/setup.py \
	--staging_location=gs://$(BUCKET)/dataflow/staging \
	--streaming \
	--update \
	--max_num_workers=$(MAX_NUM_WORKERS) \
	--bucket=$(BUCKET) \
	--extra_package=dist/print-nanny-dataflow-0.1.0.tar.gz \

lint:
	$(PYTHON) -m black setup.py print_nanny_dataflow conftest.py tests

install-git-hooks:
	cp -a hooks/. .git/hooks/