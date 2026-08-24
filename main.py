from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Import your document processing & RAG functions
from document_processor import process_document
from rag_pipeline import answer_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "FastAPI Backend is Running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Process and index the document
        process_document(file_path)
        return {"status": "success", "message": "Document processed successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat")
async def chat(question: str = Form(...)):
    try:
        response = answer_question(question)
        return {"answer": response}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}
