import os

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def create_embeddings():
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not api_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not configured")

    return HuggingFaceInferenceAPIEmbeddings(
        api_key=api_token,
        model_name=EMBEDDING_MODEL
    )


def create_vectorstore(chunks):
    embeddings = create_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def load_llm():
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not api_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not configured")

    return HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=api_token,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=256
    )


def create_prompt():
    template = """You are an AI document assistant.

Use ONLY the context below to answer the question.

If the answer is not available in the context, say that you could not find the answer.

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def ask_question(vectorstore, llm, question):
    docs = vectorstore.similarity_search(question, k=4)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = create_prompt()

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)

    return response, docs
