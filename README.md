
I want to create an web app to turn my bank statement into spend analysis

also trying to write the whole process in the strategy design pattern


# tech stacks (plan):
### -- raw data storage: storage bucket -> Bigquery
### --model tracking: MLFlow --> host on CloudSQL (too costly)
### --researching Vertex AI
### --model storage: storage bucket
### --model deployment: FAST-API/Flask -> research cloud run vs app engine vs vertex ai for cost
### --front end: streamlit, on app engine if using it for deployment

--
<div>
    <img alt="Version" src="https://img.shields.io/badge/Project Number-2-orange.svg?cacheSeconds=2592000" />
</div>

streamlit App for spend analysis & interaction [App]()

App url : <>

## Data Pipeline
* Train:
    * CSV file -> train model -> evaluation, tracked by MLflow
* Deployment:
    * Model upload to GCP Bucket -> Load bucket from Cloud run -> Make prediction

* Tech stack & Cloud services:
* Serverless Container: [Cloud Run](https://cloud.google.com/run)
* Data warehouse: [Bigquery](https://cloud.google.com/bigquery)
* Visualization: [Streamlit](https://docs.streamlit.io/)
* CI/CD: GCP Cloud Build
* local testing/deployment:
    * Container: docker
    * testing: pytest