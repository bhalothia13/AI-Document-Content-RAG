import streamlit as st
import requests

API_URL = "https://ai-document-content-rags.vercel.app"

st.set_page_config(page_title="AI Document Intelligence RAG", page_icon="📄")
st.title("📄 AI Document Intelligence RAG")

# Session State Initialization
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
if "filename" not in st.session_state:
    st.session_state.filename = ""

# 1. Document Upload Section
st.header("1. Document Upload")
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    if st.button("Upload Document"):
        # Status Spinner & Message
        with st.spinner("Processing & Indexing Document... Please wait."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    st.session_state.uploaded = True
                    st.session_state.filename = uploaded_file.name
                    st.success("Document uploaded and processed successfully!")
                else:
                    st.error(f"Failed to process: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

# Display Active File Status
if st.session_state.uploaded:
    st.info(f"Active Document: **{st.session_state.filename}**")

# 2. Ask Question Section
st.header("2. Ask Question")
question = st.text_input("Enter your question:")

if st.button("Submit Question"):
    if not st.session_state.uploaded:
        st.warning("Pehle koi document upload karein!")
    elif not question.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Analyzing context & fetching answer..."):
            try:
                res = requests.post(f"{API_URL}/chat", data={"question": question})
                if res.status_code == 200:
                    data = res.json()
                    st.subheader("Answer:")
                    st.write(data.get("answer", data))
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
