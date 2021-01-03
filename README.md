# Apache Beam Pipelines

This repo contains the batch and stream processing jobs for Print Nanny. 
https://www.print-nanny.com/


### Authentication

```bash
$ gcloud auth login
$ gcloud auth application-default login
```

### Running the pipelines

Start a local `DirectRunner` with debug-level output:
```
$ make local-dev
```

Run the pipeline in GCP Dataflow
```
$ make dataflow-prod
```