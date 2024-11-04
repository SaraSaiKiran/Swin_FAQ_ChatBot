import os
import json
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# File paths for chat history
CHAT_HISTORY_FILE = "chat_history.json"
chat_history = []  # Initialize chat history

# Load chat history from file
def load_chat_history():
    global chat_history
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            chat_history = json.load(f)
    else:
        chat_history = []

# Save chat history to file
def save_chat_history():
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=4)

# Load the chat history at startup
load_chat_history()

app = Flask(__name__)

# Function to update chat history
def update_memory(question, answer):
    global chat_history
    chat_history.append({"query": question, "answer": answer})
    save_chat_history()

# Connect to Pinecone
def connect_to_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "langchaindemo"
    index = pc.Index(name=index_name)
    print(f"Connected to the index '{index_name}'")
    return index

# Function to convert text to embeddings
def text_to_embedding(text):
    embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    return embeddings_model.embed_query(text)

# Retrieve similar documents from Pinecone
def get_similar_docs(query, index, k=5):
    try:
        embedding = text_to_embedding(query)
        response = index.query(vector=embedding, top_k=k, include_metadata=True)
        results = [{'content': match['metadata'].get('content', '')} for match in response['matches']]
        return results
    except Exception as e:
        print(f"Error in get_similar_docs: {e}")
        return []

# Answer query using context from similar documents
def answer_query_with_context(query, documents):
    try:
        context = " ".join([doc['content'] for doc in documents])
        client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the query based ONLY on the provided context."},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": query}
        ]

        response = client.invoke(messages)
        answer = response.content
        print(f"Answer generated: {answer}")
        return answer
    except Exception as e:
        print(f"Error in answering query: {e}")
        return "Error generating response"

# Generate smart recommendations using LangChain
PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template="""
    Based on the conversation history below, suggest follow-up questions that directly relate to the input and previous context:

    History:
    {history}

    Input:
    {input}

    Suggestions for follow-up questions:
    """
)

def generate_recommendations(question):
    try:
        global chat_history
        history = "\n".join([f"User: {item['query']}\nAI: {item['answer']}" for item in chat_history])
        print(f"Formatted Chat History: {history}")

        recommendation_chain = LLMChain(
            llm=ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
            prompt=PROMPT
        )
        result = recommendation_chain.run(history=history, input=question)

        recommendations = [rec.strip() for rec in result.split("\n") if rec.strip()]
        print(f"Recommendations generated: {recommendations}")
        return recommendations[:5]
    except Exception as e:
        print(f"Error during recommendation generation: {e}")
        return []

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        index = connect_to_pinecone()
        similar_docs = get_similar_docs(user_query, index)
        answer = answer_query_with_context(user_query, similar_docs)

        update_memory(user_query, answer)
        follow_up_questions = generate_recommendations(user_query)

        return jsonify({"answer": answer, "recommendations": follow_up_questions})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
