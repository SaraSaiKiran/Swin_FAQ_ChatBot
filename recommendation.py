from langchain_community.llms import OpenAI  # Updated import
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

# Define a PromptTemplate for recommendations
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

# Function to generate recommendations based on the chat history
def generate_recommendations(question, chat_history):
    try:
        # Format chat history
        history = "\n".join([f"Human: {q}\nAI: {a}" for q, a in chat_history])
        print(f"Formatted Chat History: {history}")

        # Initialize the OpenAI model
        llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

        # Create an LLMChain with the prompt and the model
        recommendation_chain = LLMChain(llm=llm, prompt=PROMPT)

        # Run the chain and get the recommendations
        result = recommendation_chain.run({"history": history, "input": question})

        # Process the result to extract individual recommendations
        recommendations = [rec.strip() for rec in result.split("\n") if rec.strip()]
        print(f"Recommendations generated: {recommendations}")
        return recommendations
    except Exception as e:
        print(f"Error during recommendation generation: {e}")
        return []
