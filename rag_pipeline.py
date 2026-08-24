import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate

from document_processor import process_pdf


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


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


def create_vectorstore(chunks):

    embeddings = create_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        dtype=torch.float32
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.1
    )

    return HuggingFacePipeline(
        pipeline=generator
    )


def create_prompt():

    template = """
You are an AI document assistant.

Use ONLY the context below to answer the question.

If the answer is not available in the context,
say that you could not find the answer.

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    return PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question"
        ]
    )


def ask_question(vectorstore, llm, question):

    docs = vectorstore.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = create_prompt()

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(
        final_prompt
    )

    return response, docs

if __name__ == "__main__":

    from pathlib import Path

    print("Loading PDF...")

    pdf_path = Path(
        r"C:\Users\Sumit\OneDrive\Desktop\AI-Document-RAG\dl-curriculum.pdf"
    )

    # 1. PDF → chunks
    chunks = process_pdf(pdf_path)

    print(f"Total chunks: {len(chunks)}")

    # 2. Chunks → FAISS
    print("\nCreating embeddings and vector database...")

    vectorstore = create_vectorstore(chunks)

    print("Vector database created successfully!")

    # 3. Load Hugging Face model
    print("\nLoading Hugging Face model...")

    llm = load_llm()

    print("LLM loaded successfully!")

    # 4. Ask question
    question = input(
        "\nAsk a question about your PDF: "
    )

    # 5. RAG
    response, docs = ask_question(
        vectorstore,
        llm,
        question
    )

    print("\n==============================")
    print("ANSWER")
    print("==============================")

    print(response)

    print("\n==============================")
    print("SOURCES")
    print("==============================")

    for doc in docs:

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        print(
            f"Page: {page + 1 if isinstance(page, int) else page}"
        )
