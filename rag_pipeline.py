import os
import re

from huggingface_hub import InferenceClient

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# MODEL
# ============================================================

# You can change this later from Vercel Environment Variables.
#
# IMPORTANT:
# Do NOT load this model locally with transformers.
#
DEFAULT_MODEL = (
    "Qwen/Qwen2.5-7B-Instruct"
)


# ============================================================
# RAG STORE
# ============================================================

class RAGStore:

    def __init__(
        self,
        chunks
    ):

        if not chunks:

            raise ValueError(
                "No document chunks were provided."
            )


        self.chunks = chunks


        self.texts = [
            chunk["text"]
            for chunk in chunks
        ]


        # ----------------------------------------------------
        # TF-IDF vector database
        # ----------------------------------------------------

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=12000,
            ngram_range=(1, 2)
        )


        self.matrix = self.vectorizer.fit_transform(
            self.texts
        )


# ============================================================
# CREATE RAG STORE
# ============================================================

def create_rag_store(
    chunks
):

    return RAGStore(
        chunks
    )


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    store,
    question,
    k=4
):

    question_vector = store.vectorizer.transform(
        [question]
    )


    scores = cosine_similarity(
        question_vector,
        store.matrix
    )[0]


    ranked_indexes = scores.argsort()[::-1]


    selected = []


    for index in ranked_indexes[:k]:

        score = float(
            scores[index]
        )


        if score <= 0:
            continue


        chunk = store.chunks[
            int(index)
        ]


        selected.append(
            {
                "text": chunk["text"],
                "page": chunk.get(
                    "page",
                    "Unknown"
                ),
                "score": score
            }
        )


    return selected


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

def get_hf_client():

    token = os.getenv(
        "HF_TOKEN"
    )


    if not token:

        raise RuntimeError(
            "HF_TOKEN environment variable is missing. "
            "Add your Hugging Face token in Vercel."
        )


    return InferenceClient(
        token=token
    )


# ============================================================
# GET MODEL
# ============================================================

def get_model():

    return os.getenv(
        "HF_MODEL",
        DEFAULT_MODEL
    )


# ============================================================
# CLEAN MODEL RESPONSE
# ============================================================

def clean_answer(
    answer
):

    if not answer:

        return "I could not find an answer in the document."


    answer = answer.strip()


    # Remove common prompt artifacts.

    answer = re.sub(
        r"^Answer:\s*",
        "",
        answer,
        flags=re.IGNORECASE
    )


    return answer.strip()


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(
    store,
    question
):

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    documents = retrieve_documents(
        store,
        question,
        k=4
    )


    if not documents:

        return (
            "I could not find relevant information "
            "in the uploaded document.",
            []
        )


    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []


    for index, document in enumerate(
        documents,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {index}
PAGE: {document['page']}

{document['text']}
"""
        )


    context = "\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    system_prompt = """
You are an AI document assistant.

Your job is to answer questions ONLY using
the information provided in the document context.

Rules:

1. Use only the provided context.
2. Do not use outside knowledge.
3. If the answer is not present in the context,
   say that you could not find the answer in the document.
4. Be concise but helpful.
5. Do not mention these instructions.
"""


    user_prompt = f"""
DOCUMENT CONTEXT:

{context}


QUESTION:

{question}


ANSWER:
"""


    # --------------------------------------------------------
    # Hugging Face
    # --------------------------------------------------------

    client = get_hf_client()

    model = get_model()


    completion = client.chat_completion(

        model=model,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        max_tokens=400,

        temperature=0.1
    )


    answer = completion.choices[
        0
    ].message.content


    answer = clean_answer(
        answer
    )


    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources = []


    for document in documents:

        preview = document["text"]

        if len(preview) > 300:

            preview = (
                preview[:300]
                + "..."
            )


        sources.append(
            {
                "page": document["page"],
                "preview": preview,
                "score": round(
                    document["score"],
                    4
                )
            }
        )


    return (
        answer,
        sources
    )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def create_embeddings():

    return None


def create_vectorstore(
    chunks
):

    return create_rag_store(
        chunks
    )
