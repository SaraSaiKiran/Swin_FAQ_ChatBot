import os
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import openai
from pinecone import Pinecone

# Load environment variables (like API keys)
load_dotenv()

# Function to convert text to embeddings
def text_to_embedding(text):
    """Converts text to an embedding vector using OpenAI embeddings."""
    print("Converting text to embedding...")
    api_key = os.getenv('OPENAI_API_KEY')
    embeddings_model = OpenAIEmbeddings(api_key=api_key)
    embedding = embeddings_model.embed_query(text)
    print(f"Generated embedding: {embedding}")
    return embedding

# Function to retrieve similar documents from Pinecone
def get_similar_docs(query, index, k=5):
    """
    Query Pinecone to find similar documents based on the input query.
    
    Args:
        query (str): The text query for similarity search.
        index: The Pinecone index to query.
        k (int): The number of results to return.

    Returns:
        list: The list of similar documents and their metadata.
    """
    print(f"Getting similar documents for query: {query}")
    # Convert query to embedding
    embedding = text_to_embedding(query)

    # Query Pinecone index
    response = index.query(vector=embedding, top_k=k, include_metadata=True)
    print(f"Pinecone response: {response}")

    # Extract and return the relevant content from the response
    return [
        {'content': match['metadata'].get('content', '')}
        for match in response['matches']
    ]

# Function to answer query using context from similar documents
def answer_query_with_context(query, index):
    """
    Answer the user's query using the most similar documents as context.

    Args:
        query (str): The user's query.
        index: The Pinecone index to retrieve similar documents.

    Returns:
        str: The generated answer from OpenAI based on the context.
    """
    # Retrieve similar documents from Pinecone
    similar_docs = get_similar_docs(query, index)

    # If no relevant context is found, provide a fallback message
    if not similar_docs or all(doc['content'] == '' for doc in similar_docs):
        return "The current information is not available to me as it is not mentioned in the data provided. Please ask a question related to the available university data."

    # Combine the content of similar documents to create the context
    context = " ".join([doc['content'] for doc in similar_docs if doc['content']])
    print(f"Context for OpenAI: {context}")

    # Initialize the OpenAI client with your API key
    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.Client(api_key=api_key)

    # Prepare the messages for the chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the query based ONLY on the provided context. Do not use any external information."},
        {"role": "user", "content": f"Context: {context}"},
        {"role": "user", "content": query}
    ]

    try:
        # Query OpenAI to get a completion
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            max_tokens=300,
            temperature=0.3
        )
        # Extract and return the answer from OpenAI's response
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return "Error generating response"

# Main execution block
if __name__ == "__main__":
    # Connect to Pinecone (ensure this function is defined elsewhere)
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "langchaindemo"  # Use the same index name
    index = pc.Index(name=index_name)

    print(f"Connected to the index '{index_name}'")

    # Specify your query
    query = "How do I apply for an official academic transcript?"
    
    # Call the function to answer the query
    answer = answer_query_with_context(query, index)
    
    print("Answer:", answer)
