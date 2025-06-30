from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable, RunnableLambda  # ✅ Added RunnableLambda
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()

# Load API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load embeddings and Pinecone vector store
embeddings = download_hugging_face_embeddings()
index_name = "medical-bot"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up Groq LLM
chatModel = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192"
)

# Define prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create QA chain
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)

# ✅ Added input formatter to pass both input & context to the QA chain
def format_input(user_input):
    docs = retriever.invoke(user_input)
    return {"context": docs, "input": user_input}

rag_chain: Runnable = RunnableLambda(format_input) | question_answer_chain  # ✅ Fix here

# Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)

    # ✅ Pass raw user input to rag_chain
    response = rag_chain.invoke(msg)

    print("Response:", response)
    return str(response)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)