import os
import requests

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# MODELS
# --------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Small model for API-based inference
LLM_MODEL = "Qwen/Qwen2.5-1.5B"


# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


# --------------------------------------------------
# VECTOR STORE
# --------------------------------------------------

def create_vectorstore(chunks):

    embeddings = create_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# --------------------------------------------------
# HUGGING FACE LLM
# --------------------------------------------------

def call_huggingface(prompt):

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:

        raise RuntimeError(
            "HF_TOKEN is not configured in Vercel Environment Variables."
        )

    url = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an AI document assistant. "
                    "Answer only using the provided context. "
                    "If the answer is not in the context, "
                    "say that you could not find the answer "
                    "in the document."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 400,
        "stream": False
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"Hugging Face API error "
            f"{response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()

    return data["choices"][0]["message"]["content"]


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

def ask_question(vectorstore, question):

    docs = vectorstore.similarity_search(
        question,
        k=4
    )

    if not docs:

        return (
            "I could not find relevant information "
            "in the document.",
            []
        )

    context_parts = []

    sources = []

    for doc in docs:

        context_parts.append(
            doc.page_content
        )

        page = doc.metadata.get(
            "page",
            None
        )

        if isinstance(page, int):

            page_number = page + 1

        else:

            page_number = page

        sources.append({
            "page": page_number
        })

    context = "\n\n---\n\n".join(
        context_parts
    )

    prompt = f"""
Use ONLY the following document context.

Do not use outside knowledge.

If the answer is not present in the context,
say:

"I could not find the answer in the document."

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    answer = call_huggingface(prompt)

    return answer, sources
