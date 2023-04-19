import requests
import pandas as pd

new_act = pd.read_csv('raw_data/activity_mar.csv', index_col=None)

new_act_json = new_act.to_json()

url = 'http://127.0.0.1:9696/predict'

response = requests.get(url, json=new_act_json)


print(response.json())