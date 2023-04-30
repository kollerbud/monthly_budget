import requests
import pandas as pd
import json

new_act = pd.read_csv('raw_data/activity_feb.csv', index_col=None)

new_act_json = new_act.to_json()

ml_repo_path = 'mlruns/1/684f7cf6eef84b3b8e46dbef8e8b39a8'

data_package = {'data': new_act_json, 'ml_path': ml_repo_path}

data_package_json = json.dumps(data_package)

url = 'http://127.0.0.1:8080/predict'
new_url = 'https://monthly-budget-owyxadjwpa-uc.a.run.app:8080/predict'

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
response = requests.get(new_url, json=data_package_json, headers=headers)

print(response.status_code)
print(response.json())
