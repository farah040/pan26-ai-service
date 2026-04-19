from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import requests

from pipeline.extractor import extract_text_from_bytes
from pipeline.chunker import chunk_document
from pipeline.encoder import encode_chunks
from qdrant_store import upsert_chunks, is_document_indexed
from retriever import retrieve

app = FastAPI()


class SourceItem(BaseModel):
    source_id: int
    source_name: str
    source_url: str


class AnalyzeRequest(BaseModel):
    submission_id: int
    document_id: int
    document_name: str
    document_url: str
    sources: list[SourceItem]
    result_callback_url: str


def run_analysis(req: AnalyzeRequest):
    workspace_id = f"submission_{req.submission_id}"

    # Step 1: download and extract suspicious document
    doc_bytes = requests.get(req.document_url).content
    doc_text = extract_text_from_bytes(doc_bytes, req.document_name)

    # Step 2: for each source, skip if already indexed, otherwise encode and store
    for src in req.sources:
        if is_document_indexed(workspace_id, str(src.source_id)):
            continue

        src_bytes = requests.get(src.source_url).content
        src_text = extract_text_from_bytes(src_bytes, src.source_name)
        chunks = chunk_document(doc_id=str(src.source_id), text=src_text)
        encoded = encode_chunks(chunks, is_query=False)
        upsert_chunks(workspace_id, encoded)

    # Step 3: retrieve — returns [(doc_id, score), ...] sorted descending
    results = retrieve(workspace_id, doc_text, top_k=None)

    # Step 4: format result
    top_score = round(results[0][1] * 100, 1) if results else 0.0

    matched_sources = [
        {
            "source_name": next(
                (s.source_name for s in req.sources if str(s.source_id) == doc_id),
                doc_id  # fallback if not found
            ),
            "match_percentage": round(score * 100, 1)
        }
        for doc_id, score in results
    ]

    # Step 5: POST result back to Django
    requests.post(req.result_callback_url, json={
        "submission_id": req.submission_id,
        "document_id": req.document_id,
        "plagiarism_score": top_score,
        "original_percentage": round(100 - top_score, 1),
        "matched_sources": matched_sources,
        "highlighted_paragraphs": [],  # future improvement
    })


@app.post("/analyze")
async def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis, req)
    return {"status": "processing"}