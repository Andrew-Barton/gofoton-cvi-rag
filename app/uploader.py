import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.document_loader import load_documents, create_vectorstore

UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    saved_files = []
    try:
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            saved_files.append(file_path)

        # Re-ingest after upload
        docs = load_documents(UPLOAD_DIR)
        create_vectorstore(docs)
        return JSONResponse({"detail": f"✅ Uploaded and ingested {len(saved_files)} file(s)."})

    except Exception as e:
        return JSONResponse({"detail": f"❌ Upload failed: {str(e)}"}, status_code=500)
