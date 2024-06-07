import requests

import json
with open('credentials.json') as f:
    data = json.load(f)
api_key = data['huggingface']['api_key']

API_URL = "https://api-inference.huggingface.co/models/pyannote/speaker-diarization"
headers = {"Authorization": f"Bearer {api_key}"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("audio.mp3")
print(output)