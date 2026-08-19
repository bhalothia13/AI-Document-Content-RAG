import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from document_processor import process_pdf
from rag_pipeline import (
    create_vectorstore,
    load_llm,
    ask_question
)


app = FastAPI(
    title="AI Document Intelligence RAG API"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Session variables
VECTORSTORE = None
LLM = None


@app.get("/")
def read_root():
    return {
        "status": "AI Document Intelligence RAG API is running!"
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    global VECTORSTORE, LLM

    temp_pdf_path = None

    try:

        # Check file type
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed."
            )

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            contents = await file.read()

            temp_file.write(contents)

            temp_pdf_path = temp_file.name


        # Process PDF
        chunks = process_pdf(temp_pdf_path)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the PDF."
            )


        # Create FAISS vector store
        VECTORSTORE = create_vectorstore(chunks)


        # Load Hugging Face LLM
        if LLM is None:
            LLM = load_llm()


        return {
            "message": "Document processed successfully!",
            "chunks": len(chunks)
        }


    except HTTPException:
        raise

    except Exception as e:

        print("UPLOAD ERROR:", repr(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # Delete temporary PDF
        if temp_pdf_path and os.path.exists(temp_pdf_path):

            try:
                os.remove(temp_pdf_path)

            except Exception:
                pass


@app.post("/chat")
async def chat(question: str = Form(...)):

    global VECTORSTORE, LLM

    if VECTORSTORE is None:

        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first."
        )


    try:

        if LLM is None:
            LLM = load_llm()


        response, docs = ask_question(
            VECTORSTORE,
            LLM,
            question
        )


        return {
            "response": response
        }


    except Exception as e:

        print("CHAT ERROR:", repr(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
