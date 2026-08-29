import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# Read data
with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

print("Total Chunk:", len(chunks))

# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# Create FAISS vector store
vector_store = FAISS.from_texts(
    chunks,
    embedding=embeddings
)

print("\nVector Store Successfully!")

# Search question
question = "What is RAG?"

results = vector_store.similarity_search(
    question,
    k=2
)

print("\nQuestion:")
print(question)

print("\nRetrieved Information:")

for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(result.page_content)