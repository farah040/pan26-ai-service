from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from pipeline.extractor import extract_text_from_bytes
from pipeline.chunker import chunk_document
from pipeline.encoder import encode_chunks
from qdrant_store import upsert_chunks, is_document_indexed
from retriever import retrieve

app = FastAPI()


@app.post("/analyze")
async def analyze(
    document: UploadFile = File(...),
    sources: List[UploadFile] = File(...),
    source_ids: List[int] = Form(...),
    workspace_id: str = Form(...),
):
    # Step 1: extract text from the suspicious document
    doc_bytes = await document.read()
    doc_text = extract_text_from_bytes(doc_bytes, document.filename)

    # Step 2: for each source, skip if already indexed, otherwise encode and store
    for source_file, source_id in zip(sources, source_ids):
        if is_document_indexed(workspace_id, str(source_id)):
            continue  # already in Qdrant, skip

        source_bytes = await source_file.read()
        source_text = extract_text_from_bytes(source_bytes, source_file.filename)
        chunks = chunk_document(doc_id=str(source_id), text=source_text)
        encoded = encode_chunks(chunks, is_query=False)
        upsert_chunks(workspace_id, encoded)

    # Step 3: retrieve — returns [(doc_id, score), ...] sorted descending
    results = retrieve(workspace_id, doc_text, top_k=10)

    # Step 4: format response
    plagiarism_score = round(results[0][1] * 100, 1) if results else 0.0

    matched_sources = [
        {
            "source_id": int(doc_id),
            "match_percentage": round(score * 100, 1)
        }
        for doc_id, score in results
    ]

    return {
        "plagiarism_score": plagiarism_score,
        "matched_sources": matched_sources,
        "highlighted_text": "",
    }