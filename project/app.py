from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os 

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print("API key loaded:", bool(api_key))

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=api_key
)

response = model.invoke(
    "What is RAG? Explain in simple words."
)

print("\nANSWER:\n")

if isinstance(response.content, list):
    for item in response.content:
        if item.get("type") == "text":
            print(item.get("text"))
else:
    print(response.content)
