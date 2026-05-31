# https://build.nvidia.com/mistralai/mistral-large-3-675b-instruct-2512

import os

import requests, base64
from dotenv import load_dotenv
load_dotenv()

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True


headers = {
  "Authorization": f"Bearer {os.getenv('MISTRAL_LARGE_3_675B_API_KEY')}",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "model": "mistralai/mistral-large-3-675b-instruct-2512",
  "messages": [{"role":"user","content":"Чувсвам се зле, какво да правя? Искам да седна и да плача."}],
  "max_tokens": 2048,
  "temperature": 0.15,
  "top_p": 1.00,
  "frequency_penalty": 0.00,
  "presence_penalty": 0.00,
  "stream": stream
}



response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())
