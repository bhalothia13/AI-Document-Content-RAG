import os

from pathlib import Path

from pypdf import PdfReader


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " "
    )

    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    lines = [
        line
        for line in lines
        if line
    ]

    return "\n".join(lines)


# ============================================================
# PDF LOADER
# ============================================================

def load_pdf(file_path):

    reader = PdfReader(
        str(file_path)
    )

    documents = []

    for page_number, page in enumerate(
        reader.pages
    ):

        text = page.extract_text() or ""

        text = clean_text(
            text
        )

        if text:

            documents.append(
                {
                    "text": text,
                    "page": page_number + 1
                }
            )

    return documents


# ============================================================
# TXT LOADER
# ============================================================

def load_txt(file_path):

    path = Path(
        file_path
    )

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    text = clean_text(
        text
    )

    if not text:
        return []

    return [
        {
            "text": text,
            "page": 1
        }
    ]


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(
    text,
    chunk_size=1200,
    chunk_overlap=200
):

    text = text.strip()

    if not text:
        return []


    chunks = []

    start = 0

    text_length = len(text)


    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )


        chunk = text[
            start:end
        ].strip()


        if chunk:

            chunks.append(
                chunk
            )


        if end >= text_length:
            break


        start = max(
            end - chunk_overlap,
            start + 1
        )


    return chunks


# ============================================================
# COMPLETE DOCUMENT PROCESSOR
# ============================================================

def process_document(
    file_path,
    extension
):

    extension = extension.lower()


    if extension == ".pdf":

        documents = load_pdf(
            file_path
        )

    elif extension == ".txt":

        documents = load_txt(
            file_path
        )

    else:

        raise ValueError(
            "Unsupported file type."
        )


    final_chunks = []


    for document in documents:

        text_chunks = split_text(
            document["text"]
        )


        for chunk in text_chunks:

            final_chunks.append(
                {
                    "text": chunk,
                    "page": document["page"]
                }
            )


    return final_chunks


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def process_pdf(file_path):

    return process_document(
        file_path,
        ".pdf"
    )
