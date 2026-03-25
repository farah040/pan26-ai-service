from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,       # enum: COSINE, DOT, EUCLID
    VectorParams,   # config for vector dimensions + distance metric
    PointStruct,    # represents one vector + its metadata
    Filter,         # for filtering points by metadata
    FieldCondition, # condition on a metadata field
    MatchValue      # exact match condition
)
import uuid  # to generate unique IDs for each chunk

QDRANT_URL = "http://localhost:6333"
VECTOR_DIM = 768  # e5-base-v2 produces 768-dimensional vectors

_client = None # no connection yet

def get_client():
    """
     first time it's called -> connection created 
     every call after that -> returns the same already-open connection
    """
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client

def ensure_collection(workspace_id: str):
    """
    Each workspace gets its own collection so workspaces are isolated
    This function creates the collection only if it doesn't exist yet
    """
    client = get_client() # connection started
    # Get names of all existing collections
    collections = [c.name for c in client.get_collections().collections]
    if workspace_id not in collections:
        client.create_collection(
            collection_name=workspace_id,
            vectors_config=VectorParams(
                size=VECTOR_DIM,      # must match the embedding model output
                distance=Distance.COSINE  # similarity metric
            )
        )

# update + insert -> upsert
def upsert_chunks(workspace_id, encoded_chunks):
    """
    Stores encoded chunks(doc_id, chunk_text, embedding)
    
    Each chunk becomes a 'point' in Qdrant with:
    - id: a unique UUID 
    - vector: the embedding (768)
    - payload: metadata dict — doc_id and chunk_text stored alongside the vector
    """
    client = get_client()
    ensure_collection(workspace_id)  # make sure collection exists first

    points = [
        PointStruct(
            id=str(uuid.uuid4()), # random unique id
            vector=embedding,        
            payload={
                "doc_id": doc_id,         # which document this chunk came from
                "chunk_text": chunk_text  # the original text (useful for reranking later)
            }
        )
        for doc_id, chunk_text, embedding in encoded_chunks
    ]

    client.upsert(collection_name=workspace_id, points=points) # sending points to qdrant


def search(workspace_id, query_embedding, top_k=1000) -> list[tuple[str, float]]:
    """
    Given a single query embedding, finds the top_k most similar chunks
    in the workspace and returns (doc_id, score) pairs
    
    Note: multiple chunks can belong to the same doc_id
    The aggregation (deciding the final score per document) happens 
    in retriever.py, not here
    """
    client = get_client()

    results = client.search(
        collection_name=workspace_id,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True # include the payload (doc_id, chunk_text) in results
    )

    return [(hit.payload["doc_id"], hit.score) for hit in results]


def is_document_indexed(workspace_id, doc_id):
    """
    Returns True if this document already has chunks stored in Qdrant
    Used to avoid re-encoding and re-upserting the same file twice
    """
    client = get_client()

    collections = [c.name for c in client.get_collections().collections]
    if workspace_id not in collections:
        return False

    results, _ = client.scroll(
        collection_name=workspace_id,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        ),
        limit=1
    )
    return len(results) > 0


def delete_document(workspace_id, doc_id):
    """
    Deletes ALL chunks belonging to a specific document.
    
    Since one document = many chunks, we can't delete by a single ID.
    Instead we filter by the doc_id stored in the payload metadata
    and delete all matching points
    """
    client = get_client()
    client.delete(
        collection_name=workspace_id,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="doc_id", # look at the payload field "doc_id"
                    match=MatchValue(value=doc_id) # where it equals our target doc_id
                )
            ]
        )
    )
