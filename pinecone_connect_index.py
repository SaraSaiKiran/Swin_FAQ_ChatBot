import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def connect_to_pinecone():
    """Connect to the Pinecone index and return the index object."""
    # Set the API key for Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Pinecone API Key not found. Please set it in the environment variables.")
        return

    pc = Pinecone(api_key=api_key)

    # Define index name
    index_name = "langchaindemo"  # Use the same index name

    # Connect to the existing index
    index = pc.Index(name=index_name)

    print(f"Connected to the index '{index_name}'")
    return index  # Return the index object

# Add this block to call the function when the script is executed
if __name__ == "__main__":
    connect_to_pinecone()