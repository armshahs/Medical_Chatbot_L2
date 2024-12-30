from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI

import os
from dotenv import load_dotenv, find_dotenv
from src.prompt import *

# initializing Flask
app = Flask(__name__)

# loading keys
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]

# defining embedding model
embeddings = download_hugging_face_embeddings()

# defining pinecone index
index_name = "medicalbot"

# if already present, then just add to exiting, else create new
# # Fetch or create
# existing_indexes = [index_info["name"] for index_info in pc.list_indexes()
# print(existing_indexes)

# if index_name not in existing_indexes:
#     pc.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )


# Create and add embeddings to pinecone
# Embed each chunk and upsert the embeddings into your Pinecone index.
# Assumes that the data is already in pinecone and we only need to retrieve it. For data uploading, create a new function if needed.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# specifying llm parameters
llm = OpenAI(temperature=0.4, max_tokens=500)

# propmt setup
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# chain setup
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
