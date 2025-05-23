import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import pandas as pd
from fastapi import Request
import openpyxl
import joblib

vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('faq_classifier_model.pkl')


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qdrant setup
collection_name = "rag-infloat-collection"

client = QdrantClient(
    url="https://48b49ac1-8387-42bb-b0d7-10587d2aa625.eu-west-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.1ugiYzO7TerHdVXROwWBNgIMkv3zMymBGeMrKXVvm68",
)

# Create collection if not exists
if not client.collection_exists(collection_name):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

embed_model = SentenceTransformer('intfloat/multilingual-e5-small')

# Extract from excel
def extract_text_from_excel():
    wb = openpyxl.load_workbook()
    sheet = wb.active

    for sheet in wb:
        print (sheet.title)
    
    for row in sheet.itter_rows(values_only=True):
        print(row)


# Extract from pdf
def extract_text_from_pdf_bytes(file_bytes):
    doc = fitz.open(stream=file_bytes.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# Splitting text
def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

# Text embedding
def embed_texts(texts):
    formatted_texts = [f"passage: {text}" for text in texts]
    return embed_model.encode(formatted_texts)

# Uploading per chunks
def upload_chunks(chunks):
    vectors = embed_texts(chunks)
    payload = [{"text": chunk} for chunk in chunks]
    client.upsert(
        collection_name= collection_name,
        points=[{
            "id": idx,
            "vector": vector.tolist(),
            "payload": payload[idx]
        } for idx, vector in enumerate(vectors)]
    )



# 📌 POST API endpoint
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        text = extract_text_from_pdf_bytes(file.file)
        chunks = split_text(text)
        upload_chunks(chunks)
        return JSONResponse(status_code=200, content={"message": "Upload success", "total_chunks": len(chunks)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Upload failed", "error": str(e)})

@app.get("/get-data")
async def get_data():
    try:
        response = client.scroll(
            collection_name="rag-infloat-collection",
            limit=5
        )
        return JSONResponse(status_code=200, content={
            "message": "Success fetching data",
            "data": response[0]  # List of points
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "message": "Failed fetching data",
            "error": str(e) 
        })

@app.post("/predict")
async def predict_from_chat_messages():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:5000/api/chats")
            chat_data = response.json()  

        questions = [chat["user_message"] for chat in chat_data if "user_message" in chat]

        if not questions:
            return JSONResponse(status_code=400, content={"error": "No user messages found"})

        # Transform and predict
        vec_questions = vectorizer.transform(questions)
        predictions = model.predict(vec_questions)

        return JSONResponse(content={
            "questions": questions,
            "predicted_categories": predictions.tolist()
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
