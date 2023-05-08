import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()


def test_local(url):

    'local url = http://127.0.0.1:8080/predict'
    new_act = pd.read_csv('raw_data/activity_feb.csv', index_col=None)
    new_act.dropna(subset=['Category'], inplace=True)
    new_act_json = new_act.to_json()

    response = requests.post(url, json=new_act_json)
    print(response.status_code)


def test_deployed(url):

    'test deployed cloud run API'
    new_act = pd.read_csv('raw_data/activity_feb.csv', index_col=None)
    new_act.dropna(subset=['Category'], inplace=True)
    new_act_json = new_act.to_json()

    response = requests.post(url=url, json=new_act_json, timeout=60)

    print(response.status_code)
    print(response.json())


test_local('http://127.0.0.1:8080/predict')
#clour_run_url = os.getenv('deploy_site')
#test_deployed(clour_run_url)