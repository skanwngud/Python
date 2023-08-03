import requests
import json
import os

api_url = 'http://192.168.0.126:8000/run'

ROOT_PATH = '/mnt/germany/data/pnuts/person_detection/'
first = os.path.join(ROOT_PATH, '2023-04-01/2023-04-01_00_00_05_')
second = os.path.join(ROOT_PATH, '2023-04-02/2023-04-02_00_00_09_')
third = os.path.join(ROOT_PATH, '2023-04-03/2023-04-03_00_00_08_')
fourth = os.path.join(ROOT_PATH, '2023-04-04/2023-04-04_00_00_08_')

resp = requests.post(
    url=api_url,
    headers={'USER_ID': '', 'X-API-KEY': ''},
    data=json.dumps({'root_path': str(second)}),
    verify=False
)