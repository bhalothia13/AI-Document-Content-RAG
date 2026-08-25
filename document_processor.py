from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(file_path):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if path.suffix.lower() == ".pdf":

        loader = PyPDFLoader(str(path))

    elif path.suffix.lower() == ".txt":

        loader = TextLoader(
            str(path),
            encoding="utf-8"
        )

    else:

        raise ValueError(
            "Only PDF and TXT files are supported."
        )

    return loader.load()


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)


def process_document(file_path):

    documents = load_documents(file_path)

    chunks = split_documents(documents)

    return chunks


# Backward compatibility
def process_pdf(file_path):

    return process_document(file_path)
