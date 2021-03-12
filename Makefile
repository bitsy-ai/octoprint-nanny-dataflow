
.PHONY: local-dev dataflow-prod

PYTHON=.venv/bin/python
PIP=.venv/bin/pip

local-dev:
	$(PYTHON) windowed_health.py \
	--runner DirectRunner \
	--loglevel INFO

dataflow-prod:
	$(PYTHON) windowed_tfrecords.py \
	--runner DataflowRunner \
	--topic projects/print-nanny/topics/bounding-boxes-prod \
	--window 300 \
	--sink gs://print-nanny-prod/dataflow/bounding-box-events/windowed \
	--loglevel INFO