import os
import math
from huggingface_hub import InferenceClient

# =========================
# Hugging Face Models
# =========================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# =========================
# Hugging Face Client
# =========================

def get_hf_client():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not token:
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN environment variable is missing."
        )

    # Note: provider="auto" removed here to prevent routing errors
    return InferenceClient(
        api_key=token
    )


# =========================
# Generate Embedding
# =========================

def create_embedding(text):
    client = get_hf_client()

    embedding = client.feature_extraction(
        text,
        model=EMBEDDING_MODEL
    )

    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    if embedding and isinstance(embedding[0], list):
        embedding = embedding[0]

    return embedding


# =========================
# Create Vector Store
# =========================

def create_vectorstore(chunks):
    documents = []
    for chunk in chunks:
        text = chunk.page_content
        vector = create_embedding(text)
        documents.append({
            "text": text,
            "embedding": vector,
            "metadata": getattr(chunk, "metadata", {})
        })
    return documents


# =========================
# Cosine Similarity
# =========================

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot / (norm_a * norm_b)


# =========================
# Retrieve Relevant Chunks
# =========================

def retrieve_documents(vectorstore, question, k=4):
    question_embedding = create_embedding(question)
    scored_documents = []

    for document in vectorstore:
        score = cosine_similarity(
            question_embedding,
            document["embedding"]
        )
        scored_documents.append((score, document))

    scored_documents.sort(key=lambda x: x[0], reverse=True)

    return [document for score, document in scored_documents[:k]]


# =========================
# Load LLM
# =========================

def load_llm():
    return get_hf_client()


# =========================
# Ask Question
# =========================

def ask_question(vectorstore, llm, question):
    docs = retrieve_documents(
        vectorstore,
        question,
        k=4
    )

    context = "\n\n".join(
        document["text"]
        for document in docs
    )

    prompt = f"""
You are an AI document assistant.

You must answer ONLY using the information present
in the provided document context.

If the answer is not present in the context, say:

"I could not find the answer in the uploaded document."

Do not use outside knowledge.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using only "
                    "the provided document context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=256
    )

    answer = response.choices[0].message.content

    return answer, docs
