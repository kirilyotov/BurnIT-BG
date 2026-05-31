# https://build.nvidia.com/nvidia/nemotron-3-content-safety
import os

import requests, base64

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = False

def read_b64(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode()

headers = {
  "Authorization": f"Bearer {os.getenv('NEMETRON_3_CONTENT_SAFETY_API_KEY')}",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "model": "nvidia/nemotron-3-content-safety",
  "messages": [{"role":"user","content":"Чувсвам се зле, какво да правя? Искам да седна и да плача."}],
  "max_tokens": 512,
  "temperature": 0.20,
  "top_p": 0.70,
  "stream": stream,
  
}
response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)
if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())
