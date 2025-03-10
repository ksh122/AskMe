import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
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

uploaded_file = st.file_uploader("Upload a pdf file", type=["pdf"])

if uploaded_file is not None:
    with open("temp_uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Loading Document..."):
        loader = PyMuPDFLoader("temp_uploaded_file.pdf", extract_images=False)
        docs = loader.load()
    st.spinner(text="Document uploaded")
    
    with st.spinner("Splitting Text..."):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=500,
            chunk_overlap=20
        )
        doc = text_splitter.split_documents(docs)
    
    with st.spinner("Embedding Data..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(doc, embeddings)

    with st.spinner("Creating Retriever..."):
        retriever = vector_store.as_retriever()

    retrievalQA = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    query = st.text_area("Enter your Question here")
    
    if query:
        with st.spinner("Retrieving Answer..."):
            result = retrievalQA({"query": query})['result']
        st.text_area("Answer:", result)