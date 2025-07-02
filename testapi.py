import os
from openai import OpenAI

openai_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_key)

try:
    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(resp.choices[0].message.content)
    print("API key is accepted.")
except Exception as e:
    print("There was an error:", e)
