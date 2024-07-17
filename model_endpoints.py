from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from flask import Flask, request

app = Flask(__name__)


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key is None:
    raise ValueError("Groq API key not found. Please set the GROQ_API_KEY environment variable in your .env file.")

cached_llm = ChatGroq(model_name="llama3-8b-8192")

@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    response = cached_llm.invoke([HumanMessage(content=query)])
    response_answer = "Answer: " + response.content
    return response_answer

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
