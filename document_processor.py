from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(file_path):
    """
    Load PDF pages using PyPDFLoader.
    """

    loader = PyPDFLoader(str(file_path))

    documents = loader.load()

    return documents


def split_documents(documents):
    """
    Split documents into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def process_pdf(file_path):
    """
    Complete PDF processing pipeline.
    """

    documents = load_documents(file_path)

    chunks = split_documents(documents)

    return chunks


if __name__ == "__main__":

    pdf_path = Path(
        r"C:\Users\Sumit\OneDrive\Desktop\AI-Document-RAG\dl-curriculum.pdf"
    )

    print("PDF path:")
    print(pdf_path)

    print("\nPDF exists:")
    print(pdf_path.exists())

    if not pdf_path.exists():
        print("\nERROR: PDF file not found.")
        exit()

    chunks = process_pdf(pdf_path)

    print("\nDocument processed successfully!")

    print("Total chunks:", len(chunks))

    print("\nFirst chunk:")
    print(chunks[0].page_content)

    print("\nMetadata:")
    print(chunks[0].metadata)