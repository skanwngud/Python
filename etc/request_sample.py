import requests
import json

api_url = 'http://192.168.0.126:8000/num_of_people'

resp = requests.post(
    url=api_url,
    headers={'USER-ID': '', 'X-API-KEY': ''},
    data=json.dumps({'key': 'values'}),
    verify=False
)

result = resp.json()
print(f'current people in store are {result["resultData"][0]}')
print(f'current frame idx is {result["resultData"][1]}')