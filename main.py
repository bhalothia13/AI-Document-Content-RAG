from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import traceback

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
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        process_document(tmp_path)
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return {"status": "success", "message": "Document processed successfully"}
    except Exception as e:
        print("UPLOAD ERROR:", traceback.format_exc())
        return {"status": "error", "message": str(e)}

@app.post("/chat")
async def chat(question: str = Form(...)):
    try:
        response = answer_question(question)
        return {"answer": response}
    except Exception as e:
        print("CHAT ERROR:", traceback.format_exc())
        return {"answer": f"Error: {str(e)}"}
