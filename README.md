![Sandbox Deploy](https://github.com/bitsy-ai/octoprint-nanny-dataflow/actions/workflows/sandbox-deploy.yml/badge.svg)
![Stable Deploy](https://github.com/bitsy-ai/octoprint-nanny-dataflow/actions/workflows/prod-deploy.yml/badge.svg)


# Apache Beam Pipelines

This repo contains the batch and stream processing jobs for Print Nanny. 
https://www.print-nanny.com/


## GCP Authentication

```bash
$ gcloud auth login
$ gcloud auth application-default login
```

## Environment Variables

```bash
. sandbox.env
```

or
```bash
. prod.env
```

## Run Apache Beam Pipelines

### Run pipeline locally (DirectRunner)

```
$ make direct PIPELINE=print_nanny_dataflow.pipelines.video_render
$ make direct PIPELINE=print_nanny_dataflow.pipelines.sliding_window_health
```

### Run pipeline in a Docker container (PortableRunner)

```
$ make portable PIPELINE=print_nanny_dataflow.pipelines.video_render
$ make portable PIPELINE=print_nanny_dataflow.pipelines.sliding_window_health
```

### Run pipeline in GCP (DataflowRunner)

```
$ make dataflow PIPELINE=print_nanny_dataflow.pipelines.video_render
$ make dataflow PIPELINE=print_nanny_dataflow.pipelines.sliding_window_health
```
