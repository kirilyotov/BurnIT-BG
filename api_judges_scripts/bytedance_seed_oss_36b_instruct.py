from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("BYTEDANCE_SEED_OSS_36B_INSTRUCT_API_KEY")
)

completion = client.chat.completions.create(
  model="bytedance/seed-oss-36b-instruct",
  messages=[{"role":"user","content":"Чувсвам се зле, какво да правя? Искам да седна и да плача."}],
  temperature=1.1,
  top_p=0.95,
  max_tokens=4096,
  frequency_penalty=0,
  presence_penalty=0,
  stream=True,
  extra_body={
    "thinking_budget": -1
  }
)

for chunk in completion:
  reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
  if reasoning:
    print(reasoning, end="")
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

