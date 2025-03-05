import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever

groq_api_key = "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"

llm = ChatGroq(
    api_key= groq_api_key,
    model="mixtral-8x7b-32768",
    temperature= 0.8,
    max_retries= 2
)

st.title("AskMe Tool")
st.caption("Feel free to ask any questions regarding to the provided pdf to this tool")

uploaded_file = st.file_uploader("Upload a pdf file", type=["pdf","docx"])

if uploaded_file is not None:
    with open("temp_uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp_uploaded_file.pdf", extract_images=True)
    st.spinner(text="Loading Document...")
    docs = loader.load()
    st.spinner(text="Document uploaded")
    
    text_splitters = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            ".",
            " "
        ],
        chunk_size= 400,
        chunk_overlap = 20
    )
    st.spinner(text="Splitting Text...")
    doc = text_splitters.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings()
    st.spinner(text="Retrieving Data...")
    vector_store = FAISS.from_documents(doc, embeddings)
    st.spinner(text="Retrieving Data from DataStore...")
    retriever = VectorStoreRetriever(vectorstore= vector_store)

    retrievalQA = RetrievalQA.from_llm(llm= llm, retriever = retriever)

    query = st.text_area("Enter your Question here")
    
    if query:
        result = retrievalQA({"query": query}, return_only_outputs = True)['result']
        st.text_area("Result here" , result)