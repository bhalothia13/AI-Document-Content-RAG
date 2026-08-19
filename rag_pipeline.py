import os

from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

def get_hf_client():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not token:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN is not configured"
        )

    return InferenceClient(
        api_key=token
    )


# ============================================================
# CREATE EMBEDDINGS USING HUGGING FACE API
# ============================================================

class HuggingFaceAPIEmbeddings:

    def __init__(self, model):
        self.model = model
        self.client = get_hf_client()

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:
            result = self.client.feature_extraction(
                text,
                model=self.model
            )

            # Convert numpy array/list to normal Python list
            if hasattr(result, "tolist"):
                result = result.tolist()

            # Some models can return nested arrays
            if result and isinstance(result[0], list):
                result = result[0]

            embeddings.append(result)

        return embeddings

    def embed_query(self, text):
        result = self.client.feature_extraction(
            text,
            model=self.model
        )

        if hasattr(result, "tolist"):
            result = result.tolist()

        if result and isinstance(result[0], list):
            result = result[0]

        return result


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings():

    return HuggingFaceAPIEmbeddings(
        model=EMBEDDING_MODEL
    )


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vectorstore(chunks):

    embeddings = create_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# ============================================================
# LLM
# ============================================================

def load_llm():

    client = get_hf_client()

    return client


# ============================================================
# PROMPT
# ============================================================

def create_prompt():

    return """You are an AI document assistant.

Answer the user's question using ONLY the context provided below.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present in the context, say:
   "I could not find the answer in the document."
3. Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:
"""


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(vectorstore, llm, question):

    # Find relevant documents
    docs = vectorstore.similarity_search(
        question,
        k=4
    )

    # Build context
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # Create prompt
    prompt = create_prompt()

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    # Call Hugging Face API
    response = llm.text_generation(
        final_prompt,
        model=LLM_MODEL,
        max_new_tokens=256,
        temperature=0.1,
        return_full_text=False
    )

    return response, docs
