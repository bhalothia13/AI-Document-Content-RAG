import os
import tempfile
import requests

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from document_processor import process_document
from rag_pipeline import create_vectorstore, ask_question


app = FastAPI(
    title="AI Document Intelligence RAG API",
    version="1.0.0"
)


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# GLOBAL VECTORSTORE
# --------------------------------------------------

vectorstore = None
current_filename = None


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "success",
        "message": "AI Document Intelligence RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "document_loaded": vectorstore is not None,
        "filename": current_filename
    }


# --------------------------------------------------
# UPLOAD DOCUMENT
# --------------------------------------------------

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    global vectorstore
    global current_filename

    filename = file.filename or ""

    if not filename.lower().endswith((".pdf", ".txt")):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Only PDF and TXT files are supported."
            }
        )

    try:

        file_bytes = await file.read()

        if not file_bytes:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Uploaded file is empty."
                }
            )

        suffix = ".pdf" if filename.lower().endswith(".pdf") else ".txt"

        # Vercel serverless path fix: write files under /tmp directory
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            dir="/tmp"
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_path = temp_file.name

        try:

            # Extract chunks
            chunks = process_document(temp_path)

            if not chunks:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": "No text could be extracted from the document."
                    }
                )

            # Build vector store
            vectorstore = create_vectorstore(chunks)

            current_filename = filename

        finally:

            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

        return {
            "success": True,
            "message": "Document uploaded and indexed successfully.",
            "filename": filename,
            "chunks": len(chunks)
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# --------------------------------------------------
# CHAT
# --------------------------------------------------

@app.post("/chat")
async def chat(
    question: str = Form(...)
):

    global vectorstore

    if vectorstore is None:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Please upload a document first."
            }
        )

    if not question.strip():

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Question cannot be empty."
            }
        )

    try:

        answer, sources = ask_question(
            vectorstore,
            question
        )

        return {
            "success": True,
            "answer": answer,
            "sources": sources
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )
