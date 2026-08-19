import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from document_processor import process_pdf
from rag_pipeline import create_vectorstore, load_llm, ask_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for session handling
VECTORSTORE = None
LLM = None

@app.get("/")
def read_root():
    return {"status": "AI Document Intelligence RAG API is running!"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global VECTORSTORE, LLM
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_pdf_path = temp_file.name

        chunks = process_pdf(temp_pdf_path)
        VECTORSTORE = create_vectorstore(chunks)

        if LLM is None:
            LLM = load_llm()

        os.remove(temp_pdf_path)
        return {"message": "Document processed successfully!", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(question: str = Form(...)):
    global VECTORSTORE, LLM
    if VECTORSTORE is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    try:
        response, docs = ask_question(VECTORSTORE, LLM, question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
