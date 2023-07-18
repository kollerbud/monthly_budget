
<font size='5'>I want to create an web app to turn my bank statement into spend analysis \
also there is a button to predict spend categories. </font>

<div>
    <img alt="Version" src="https://img.shields.io/badge/Project Number-2-orange.svg?cacheSeconds=2592000" />
</div>

streamlit App for spend analysis & interaction [App](https://kollerbud-monthly-budget-appapp-pcogau.streamlit.app/)

App url : <https://kollerbud-monthly-budget-appapp-pcogau.streamlit.app/>

## Data Pipeline
* Train:
    * Google sheets -> BigQuery -> Preprocessing & Encoding -> train model -> evaluation -> Deploy to Bucket, tracked by Vertex AI(Model Registry)
* Deployment:
    * Model upload to GCP Bucket -> Cloud run using the latest model -> Make prediction

## pipeline architect
![pipe_image](https://github.com/kollerbud/monthly_budget/blob/master/img/budget-2023-07-15-0123.png)

Tech stack & Cloud services:
* User input(me): Google_sheets
* Serverless Container: [Cloud Run](https://cloud.google.com/run)
* Data warehouse: [BigQuery](https://cloud.google.com/bigquery)
* Model registry [VertexAI](https://cloud.google.com/vertexai) (Vertex AI--Model Registry)
* Visualization: [Streamlit](https://docs.streamlit.io/)
* CI/CD: GCP--Cloud Build & GCP Vertex AI--Model Registry
* local testing/deployment:
    * Container: docker
    * testing: pytest


## note for my future self
* CI/CD
    * Tried MLFlow, but deploy it on GCP was expensive (VM + SQL required)
    * deleting versions on Model Registry doesn't automatically delete the model in Cloud Bucket, \
      so still need to manually deleting old versions of models in the Bucket
    * Still confused on Vertex AI artifacts
    * CloudRun Services Vs CloudRun Jobs vs Cloud Function:
        - Function is just cloud run without the docker part 
        - Services do need an invoke point, so i do need to use Flask/FastAPI \
        but then I get built in CI/CD option, definitely for more "requrests" type 
        - Jobs don't need invoke point, but no built in CI/CD option, so i need to \
        write manual github action yaml, also the way to deploy is literally held \
        together by this person's pull request, https://github.com/google-github-actions/deploy-cloudrun/pull/422
        - maybe dataflow could be good here? but I need to understand the cost and CI/CD workflow

* Pipeline
    * using copy bank statements onto Google sheet,  as I don't want to connect to a bank API (wasn't sure if that's possible) \
      or deal with authentication/security issue
    * BigQuery is cheaper to use than CloudSQL, and probably a better choice anyways since this is analytics focused
     - something weird with google sheets to BigQuery, need to adjust 

* ML
    * over sampling was needed, I was able to bring the prediction accuracy from 90% -> 99%*<font size='0.5'>(in training)</font>
    * factory/strategy pattern was painful to write at first, but quite good when I have \
    decided to refactor some components
    

* Python
    * leave specific versions out, just do pip install -U
    * should look into setup.py to all necessary paths situated instead of init path files
    

* GCP
    * BigQuery optimization: minium query size is 10mb, I need to find ways to reduce query requests calls
    * Write directly to BigQuery (streaming) has a cost of $0.01/200mb
    
