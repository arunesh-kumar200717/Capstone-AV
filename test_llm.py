
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(".env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

try:
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[
            {"role": "user", "content": "Reply with: Connection successful"}
        ],
    )
    print("LLM connection successful!")
    print(response.choices[0].message.content)

except Exception as e:
    print("LLM connection failed.")
    print("Error type:", type(e).__name__)
    print("Error:", str(e)[:500])
