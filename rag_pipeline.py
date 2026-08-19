import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from document_processor import process_pdf

# Hugging Face Cloud API Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def create_embeddings():
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=api_token,
        model_name=EMBEDDING_MODEL
    )


def create_vectorstore(chunks):
    embeddings = create_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def load_llm():
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
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
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = create_prompt()
    final_prompt = prompt.format(context=context, question=question)

    response = llm.invoke(final_prompt)
    return response, docs


if __name__ == "__main__":
    from pathlib import Path

    print("Loading PDF...")
    pdf_path = Path(
        r"C:\Users\Sumit\OneDrive\Desktop\AI-Document-RAG\dl-curriculum.pdf"
    )

    chunks = process_pdf(pdf_path)
    print(f"Total chunks: {len(chunks)}")

    print("\nCreating embeddings via Hugging Face API...")
    vectorstore = create_vectorstore(chunks)
    print("Vector database created successfully!")

    print("\nConnecting to Hugging Face LLM API...")
    llm = load_llm()
    print("LLM connected successfully!")

    question = input("\nAsk a question about your PDF: ")

    response, docs = ask_question(vectorstore, llm, question)

    print("\n==============================")
    print("ANSWER")
    print("==============================")
    print(response)

    print("\n==============================")
    print("SOURCES")
    print("==============================")
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        print(f"Page: {page + 1 if isinstance(page, int) else page}")
