import streamlit as st
import requests

API_URL = "https://ai-document-content-rags.vercel.app"

st.set_page_config(page_title="AI Document RAG")
st.title("📄 AI Document Intelligence RAG")

st.header("1. Document Upload")
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file and st.button("Upload Document"):
    with st.spinner("Uploading..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        res = requests.post(f"{API_URL}/upload", files=files)
        st.write(res.json() if res.status_code == 200 else res.text)

st.header("2. Ask Question")
question = st.text_input("Enter your question:")
if question and st.button("Submit Question"):
    with st.spinner("Processing..."):
        res = requests.post(f"{API_URL}/chat", data={"question": question})
        st.write(res.json() if res.status_code == 200 else res.text)
