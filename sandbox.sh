#!/bin/bash
export PROJECT=print-nanny-sandbox
export REPO=print-nanny-dataflow
export TAG=$(git rev-parse HEAD)
export REGISTRY_HOST=gcr.io
export IMAGE_URI="$REGISTRY_HOST/$PROJECT/$REPO:$TAG"
gcloud builds submit --tag $IMAGE_URI