import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the API key for Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define index name
index_name = "langchaindemo"  # Use the exact index name as in your image

# Check if the index exists
if index_name not in pc.list_indexes():
    print(f"Index '{index_name}' does not exist. Creating...")
    # Create the index if it doesn't exist
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Matching region and cloud
    )
else:
    print(f"Index '{index_name}' already exists.")
