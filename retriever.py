from collections import defaultdict
from pipeline.chunker import chunk_document
from pipeline.encoder import encode_chunks
from qdrant_store import search

AGGREGATION_METHODS = ("max", "mean")

def aggregate_chunk_scores(
    chunk_results: list[tuple[str, float]],
    method = "max"
) -> dict[str, float]:
    """
    Collapses chunk-level (doc_id, score) pairs into one score per document.

    Args:
        chunk_results: list of (doc_id, score) from qdrant_store.search()
        method: aggregation strategy — "max" (default) or "mean"

    Returns:
        dict mapping doc_id -> aggregated score
    """
    if method not in AGGREGATION_METHODS:
        raise ValueError(f"Unknown aggregation method '{method}'. Choose from: {AGGREGATION_METHODS}")

    scores = defaultdict(list)
    for doc_id, score in chunk_results:
        scores[doc_id].append(score)

    if method == "max":
        return {doc_id: max(s) for doc_id, s in scores.items()}
    elif method == "mean":
        return {doc_id: sum(s) / len(s) for doc_id, s in scores.items()}

def retrieve(
    workspace_id,
    query_text,
    top_k: int | None = None,
    aggregation = "max",
    chunk_pool = 100
) -> list[tuple[str, float]]:
    """
    Full retrieval pipeline: chunk query → encode → search → aggregate → rank.

    Args:
        workspace_id:  Qdrant collection to search in
        query_text:    raw query text (will be chunked and encoded internally)
        top_k:         how many documents to return (None = return all)
        aggregation:   score aggregation method — "max" or "mean"
        chunk_pool:    how many chunks to fetch per query chunk before aggregating

    Returns:
        list of (doc_id, score) sorted by score descending
    """
    # Step 1: chunk the query the same way we chunk corpus documents
    query_chunks = chunk_document(doc_id="query", text=query_text)

    # Step 2: encode all query chunks in one batch (is_query=True adds "query: " prefix)
    encoded_query_chunks = encode_chunks(query_chunks, is_query=True)

    # Step 3: for each query chunk, search Qdrant and pool all results
    all_chunk_results = []
    for _, _, query_embedding in encoded_query_chunks:
        results = search(workspace_id, query_embedding, top_k=chunk_pool)
        all_chunk_results.extend(results)

    # Step 4: collapse to one score per document
    doc_scores = aggregate_chunk_scores(all_chunk_results, method=aggregation)

    # Step 5: sort documents by score descending
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # Step 6: optionally truncate
    if top_k is not None:
        ranked = ranked[:top_k]

    return ranked