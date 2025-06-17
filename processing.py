from collections import Counter, defaultdict
import umap
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, requests
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz
import os
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import pandas as pd
import joblib
import pickle
import openpyxl
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score


# Load your faq_data generated earlier
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# config
COLLECTION_NAME = "rag-academics-collection-small"
QDRANT_URL      = "https://48b49ac1-8387-42bb-b0d7-10587d2aa625.eu-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY  = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.1ugiYzO7TerHdVXROwWBNgIMkv3zMymBGeMrKXVvm68"
VECTOR_DIM      = 384
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

if not client.collection_exists(COLLECTION_NAME):
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

embed_model = SentenceTransformer('intfloat/multilingual-e5-small')
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
umap_model = joblib.load("umap_model.pkl")
hdbscan_model = joblib.load("hdbscan_model.pkl")

print("sentence_model:", type(sentence_model))  # Should show SentenceTransformer
print("umap_model:", type(umap_model))  # Should show UMAP class
print("hdbscan_model:", type(hdbscan_model))  # Should show HDBSCAN class

print(hasattr(hdbscan_model, 'prediction_data_'))
print(hdbscan_model.prediction_data_ is not None)

# unicode and formatting cleaner
def clean_text(text):
    chars_to_space = [
        '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
        '\u200e', '\u200f',
        '\u2066', '\u2067', '\u2068', '\u2069',
        '\u200b', '\ufeff',
    ]
    for char in chars_to_space:
        text = text.replace(char, ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

# Extract from excel
def extract_text_from_excel():
    wb = openpyxl.load_workbook()
    sheet = wb.active

    for sheet in wb:
        print (sheet.title)
    
    for row in sheet.iter_rows(values_only=True):
        print(row)

# extract creation date from PDF
def extract_pdf_creation_date(doc):
    meta = doc.metadata
    if meta and "creationDate" in meta and meta["creationDate"]:
        raw_date = meta["creationDate"]
        if raw_date.startswith("D:"):
            raw_date = raw_date[2:]
        try:
            return f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
        except Exception:
            return ""
    return ""

# extract text from PDF bytes
def extract_text_from_pdf_bytes(file_bytes):
    doc = fitz.open(stream=file_bytes.read(), filetype="pdf")
    texts = []
    for page in doc:
        page_text = page.get_text("text")
        page_text = clean_text(page_text)
        texts.append(page_text)
    full_text = "\n".join(texts)
    return full_text.strip(), doc

# Splitting text
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

# Text embedding
def embed_texts(texts):
    formatted_texts = [f"passage: {text}" for text in texts]
    return embed_model.encode(formatted_texts)

# upload per chunk
def upload_chunks(chunks, pdf_filename, creation_date, doc, start_id):
    vectors = embed_texts(chunks)
    points = []
    for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
        global_id = start_id + idx
        first_line = next((line.strip() for line in chunk.split("\n") if line.strip()), "")
        # page number
        page_number = 1
        for page_idx in range(len(doc)):
            if first_line in clean_text(doc[page_idx].get_text("text")):
                page_number = page_idx + 1
                break
        chunk_title = first_line

        points.append({
            "id": global_id,
            "vector": vec.tolist(),
            "payload": {
                "text": chunk,
                "filename": pdf_filename,
                "page": page_number,
                "title": chunk_title,
                "date": creation_date,
            }
        })
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    return len(points)



@app.post("/upload-bk")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # timestamp-based starting ID
        start_id = int(datetime.now().timestamp() * 1000)
        pdf_filename = file.filename
        text, doc = extract_text_from_pdf_bytes(file.file)
        creation_date = extract_pdf_creation_date(doc)
        chunks = split_text(text)
        num_uploaded = upload_chunks(chunks, pdf_filename, creation_date, doc, start_id)
        return JSONResponse(status_code=200, content={
            "message": "Upload success",
            "filename": pdf_filename,
            "total_chunks": num_uploaded,
            "creation_date": creation_date
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "message": "Upload failed",
            "error": str(e)
        })

@app.get("/get-data")
async def get_data():
    try:
        response = client.scroll(
            collection_name = COLLECTION_NAME,
            limit=5
        )
        points = [r.dict() if hasattr(r, 'dict') else r for r in response[0]]
        return JSONResponse(status_code=200, content={
            "message": "Success fetching data",
            "data": points  # List of points
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "message": "Failed fetching data",
            "error": str(e) 
        })
    
@app.get("/search-file")
async def search_by_filename(filename: str = Query(...)):
    try:
        # Fetch all or a lot of points
        response = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000
        )

        # Filter manually
        filtered = [
            p.dict() if hasattr(p, 'dict') else p
            for p in response[0]
            if filename.lower() in p.payload.get("filename", "").lower()
        ]

        return JSONResponse(status_code=200, content={
            "message": "Success filtering by filename",
            "data": filtered
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "message": "Failed filtering data",
            "error": str(e)
        })
    
@app.post("/predict")
async def predict_from_chat_messages():
    try:
        # 1. Fetch chat data
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.get("https://be-service-production.up.railway.app/api/chats")
            response.raise_for_status()

        chat_data = response.json()

        # Extract questions and answers
        questions = [chat["user_message"] for chat in chat_data if "user_message" in chat]
        answers = [chat["bot_response"] for chat in chat_data if "bot_response" in chat]

        if not questions or not answers or len(questions) != len(answers):
            raise HTTPException(status_code=400, detail="Invalid or empty question/answer data")

        print(f"Found {len(questions)} valid question-answer pairs.")

        # 2. Generate embeddings
        embeddings = sentence_model.encode(questions, convert_to_numpy=True)

        # 3. UMAP dimensionality reduction
        reducer = umap.UMAP(
            n_neighbors=5,
            n_components=15,
            metric='cosine',
            min_dist=0.0,
            random_state=42
        )
        reduced_embeddings = reducer.fit_transform(embeddings)

        # 4. HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
        cluster_labels = clusterer.fit_predict(reduced_embeddings)

        # 5. Build FAQ-style grouped output
        cluster_data = defaultdict(list)
        for q, a, label in zip(questions, answers, cluster_labels):
            if label != -1:  # Ignore noise
                cluster_data[label].append((q, a))

        faq_data = []
        for cluster_id, qa_list in cluster_data.items():
            main_q, main_a = qa_list[0]
            related_qs = [q for q, _ in qa_list[1:]]
            faq_data.append({
                "Cluster": int(cluster_id),
                "Frequency": int(len(qa_list)),
                "Question": main_q,
                "Answer": main_a,
                "Related Questions": related_qs
            })

        # 6. Preview output in desired format (printed to terminal/log)
        for faq in faq_data:
            print(f"\nCluster {faq['Cluster']} (Frequency: {faq['Frequency']})")
            print(f"Q: {faq['Question']}")
            print(f"A: {faq['Answer']}")
            if faq["Related Questions"]:
                print("Related Questions:")
                for rq in faq["Related Questions"]:
                    print(f" - {rq}")
            print("-" * 80)

        return {
            "total_questions": len(questions),
            "clusters_found": len(faq_data),
            "results": faq_data
        }

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Chat service unavailable: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")