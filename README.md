
I want to create an web app to turn my bank statement into spend analysis

--
<div>
    <img alt="Version" src="https://img.shields.io/badge/Project Number-2-orange.svg?cacheSeconds=2592000" />
</div>

streamlit App for spend analysis & interaction [App](https://kollerbud-monthly-budget-appapp-pcogau.streamlit.app/)

App url : <https://kollerbud-monthly-budget-appapp-pcogau.streamlit.app/>

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
