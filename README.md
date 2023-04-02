
I want to create an web app to turn my bank statement into spend analysis

also trying to write the whole process in the strategy design pattern


# tech stacks (plan):
### -- raw data storage: storage bucket -> Bigquery
### --model tracking: MLFlow --> host on CloudSQL
### --model storage: storage bucket
### --model deployment: FAST-API/Flask -> research cloud run vs app engine vs vertex ai for cost
### --front end: streamlit, on app engine if using it for deployment


--