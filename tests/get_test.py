import requests
import pandas as pd


new_act = pd.read_csv('raw_data/activity_feb.csv', index_col=None)

new_act_json = new_act.to_json()

url = 'http://127.0.0.1:8080/predict'

response = requests.post(url, json=new_act_json)

print(response.status_code)
print(response.json())