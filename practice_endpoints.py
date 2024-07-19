from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("Groq API key not found. Please set the GROQ_API_KEY environment variable in your .env file.")

# Initialize components
folder_path = "db"
chat_history = []
cached_llm = ChatGroq(model_name="llama3-8b-8192")
embedding = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

raw_prompt = PromptTemplate(
    input_variables=["input", "context"],
    template="""
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

@app.route("/chat", methods=["POST"])
def aiPost():
    try:
        json_content = request.json
        query = json_content.get("query")
        print(f"query: {query}")
        response = cached_llm.invoke([HumanMessage(content=query)])
        response_answer = "Answer: " + response.content
        return jsonify({"answer": response_answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask_from_doc", methods=["POST"])
def askDocPost():
    try:
        json_content = request.json
        query = json_content.get("query")
        print(f"query: {query}")

        print("Loading vector store")
        vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

        print("Creating chain")
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 20, "score_threshold": 0.1},
        )

        retriever_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                ("human", "Given the above conversation, generate a search query to lookup in order to get information relevant to the conversation"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=cached_llm, retriever=retriever, prompt=retriever_prompt
        )

        document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
        retrieval_chain = create_retrieval_chain(
            history_aware_retriever, document_chain,
        )

        result = retrieval_chain.invoke({"input": query})
        print(result["answer"])

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))

        sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]

        response_answer = {"answer": result["answer"], "sources": sources}
        return jsonify(response_answer)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/add_document", methods=["POST"])
def pdfPost():
    try:
        file = request.files["file"]
        file_name = file.filename  
        save_file = "document_storage/" + file_name
        file.save(save_file)
        print(f"filename: {file_name}")

        loader = PDFPlumberLoader(save_file)
        docs = loader.load_and_split()
        print(f"doc len= {len(docs)}")

        chunks = text_splitter.split_documents(docs)
        print(f"chunks len= {len(chunks)}")

        vector_store = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=folder_path)
        vector_store.persist()

        response = {"Status": "Successfully uploaded", "filename": file_name, "doc_len": len(docs), "chunks": len(chunks)}
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
