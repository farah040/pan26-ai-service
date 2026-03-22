import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

WINDOW_SIZE = 6
STRIDE = 2

def split_sentences(text):
    """
    Splits text into sentences and strips whitespace
    """
    return [s.strip() for s in sent_tokenize(text) if s.strip()]

def sliding_window(sentences, win=WINDOW_SIZE, stride=STRIDE):
    """
    Creates overlapping chunks from a list of sentences
    """
    if len(sentences) <= win:
        return [" ".join(sentences)]
    
    return [
        (" ".join(sentences[i:i+win]))
        for i in range(0, len(sentences) - win + 1, stride)
    ]

def chunk_document(doc_id, text):
    """
    Returns list of (doc_id, chunk_text) tuples for a single document
    """
    sentences = split_sentences(text)
    windows = sliding_window(sentences)
    return [(doc_id, chunk_text) for chunk_text in windows]
