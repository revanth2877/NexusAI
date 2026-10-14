
import hashlib
import threading

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# NEXUSAI SEMANTIC DOCUMENT RETRIEVAL
# =========================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_model_lock = threading.Lock()

# Reuse indexes when the exact same document collection is searched.
_index_cache = {}
_index_lock = threading.Lock()
MAX_CACHED_INDEXES = 5


def get_embedding_model():
    """Load the embedding model once and reuse it."""
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(MODEL_NAME)

    return _model


def split_text_into_chunks(text, chunk_size=180, overlap=35):
    """Split document text into overlapping word chunks."""
    if not isinstance(text, str) or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size.")

    words = text.split()

    if not words:
        return []

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])

        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(words):
            break

    return chunks


def _prepare_chunks(documents):
    """Convert documents into searchable text chunks."""
    chunks = []

    for document in documents:
        if not isinstance(document, dict):
            continue

        filename = str(
            document.get("filename") or "Unknown document"
        )
        content = document.get("content", "")

        if not isinstance(content, str) or not content.strip():
            continue

        for chunk_number, chunk in enumerate(split_text_into_chunks(content)):
            chunks.append({
                "filename": filename,
                "text": chunk,
                "chunk_number": chunk_number + 1,
            })

    return chunks


def _documents_fingerprint(chunks):
    """Create a stable cache key for the prepared chunks."""
    digest = hashlib.sha256()

    for item in chunks:
        digest.update(item["filename"].encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(item["text"].encode("utf-8", errors="replace"))
        digest.update(b"\0")

    return digest.hexdigest()


def _build_index(chunks):
    """Build a normalized-vector FAISS index."""
    model = get_embedding_model()
    passage_texts = [item["text"] for item in chunks]

    embeddings = model.encode(
        passage_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    embeddings = np.ascontiguousarray(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def _get_cached_index(chunks):
    """Reuse a cached index or create one for this document collection."""
    fingerprint = _documents_fingerprint(chunks)

    with _index_lock:
        cached = _index_cache.get(fingerprint)

        if cached is not None:
            # Move recently used entries to the end.
            _index_cache.pop(fingerprint)
            _index_cache[fingerprint] = cached
            return cached

    index = _build_index(chunks)
    cached_entry = (index, chunks)

    with _index_lock:
        # Another request may have built the same index meanwhile.
        if fingerprint in _index_cache:
            existing = _index_cache.pop(fingerprint)
            _index_cache[fingerprint] = existing
            return existing

        _index_cache[fingerprint] = cached_entry

        while len(_index_cache) > MAX_CACHED_INDEXES:
            oldest_key = next(iter(_index_cache))
            _index_cache.pop(oldest_key)

    return cached_entry


def retrieve_chunks(question, documents, top_k=5, min_score=None):
    """
    Find document passages semantically related to a question.

    Each document should contain:
        filename: document filename
        content: extracted document text

    Returns dictionaries containing filename, text, chunk_number,
    and cosine-similarity score.

    min_score is optional. If provided, results below that similarity
    score are excluded. Similarity thresholds should be tested against
    your own documents before being used in production.
    """
    if not isinstance(question, str) or not question.strip():
        return []

    if not documents:
        return []

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        return []

    chunks = _prepare_chunks(documents)

    if not chunks:
        return []

    index, indexed_chunks = _get_cached_index(chunks)

    model = get_embedding_model()

    question_embedding = model.encode(
        [question.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    question_embedding = np.ascontiguousarray(question_embedding)

    result_count = min(top_k, len(indexed_chunks))

    scores, indices = index.search(
        question_embedding,
        result_count,
    )

    results = []

    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue

        score = float(score)

        if min_score is not None and score < min_score:
            continue

        item = dict(indexed_chunks[int(chunk_index)])
        item["score"] = score
        results.append(item)

    return results


def clear_embedding_model():
    """Release the cached model reference and document indexes."""
    global _model

    with _model_lock:
        _model = None

    with _index_lock:
        _index_cache.clear()

