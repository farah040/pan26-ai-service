from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/e5-base-v2"

# Load once — not per request
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def encode_chunks(chunks, is_query=False) -> list[tuple[str, str, list[float]]]:
    """
    Takes (doc_id, chunk_text) tuples.
    Returns (doc_id, chunk_text, embedding) tuples.
    E5 requires 'query: ' prefix for queries, 'passage: ' for corpus.
    """
    model = get_model()
    prefix = "query: " if is_query else "passage: "
    
    doc_ids = [doc_id for doc_id, _ in chunks]
    texts = [prefix + chunk_text for _, chunk_text in chunks]
    
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64
    )
    
    return [(doc_ids[i], chunks[i][1], embeddings[i].tolist()) for i in range(len(chunks))]