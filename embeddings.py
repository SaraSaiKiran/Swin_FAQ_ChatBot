import os
from langchain.embeddings.openai import OpenAIEmbeddings  # Corrected import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangDocument  # Updated import for documents
from dotenv import load_dotenv
from load_documents import load_documents  # Ensure this file has the load_documents function
from pinecone_connect_index import connect_to_pinecone  # Ensure this file has the connect_to_pinecone function

# Load environment variables globally
load_dotenv()

def create_embeddings(documents):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    document_chunks = []
    for document in documents:
        lang_doc = LangDocument(page_content=document)
        chunks = text_splitter.split_documents([lang_doc])
        document_chunks.extend(chunks)
    print(f"Split into {len(document_chunks)} chunks.")
    return document_chunks

def embed_and_upsert_chunks(chunks, pinecone_index):
    print("Starting embedding and upserting...")

    # Initialize OpenAI Embeddings with API key from environment
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i}...")
        embedding = embeddings_model.embed_documents([chunk.page_content])[0]  # Get the embedding for each chunk
        upsert_data = {
            'id': str(i),
            'values': embedding,
            'metadata': {'content': chunk.page_content}
        }
        pinecone_index.upsert(vectors=[upsert_data])
        print(f"Upserted chunk {i}.")
    
    print(f"Upserted {len(chunks)} chunks into Pinecone.")

# Example usage
if __name__ == "__main__":
    directory = 'data'  # Specify your data directory here
    documents = load_documents(directory)  # Ensure this function is defined elsewhere
    print(f"Loaded {len(documents)} documents.")
    
    chunks = create_embeddings(documents)  # Create embeddings
    pinecone_index = connect_to_pinecone()  # Connect to Pinecone
    embed_and_upsert_chunks(chunks, pinecone_index)  # Upsert chunks to Pinecone
