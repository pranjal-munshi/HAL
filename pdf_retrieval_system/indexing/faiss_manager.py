import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSRetriever:
    def __init__(self, index, model, metadata, k=3):
        self.index = index
        self.model = model
        self.metadata = metadata
        self.k = k

    def retrieve(self, query):
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(query_embedding).astype("float32"), self.k)
        return [self.metadata[idx] for idx in indices[0]]

def build_index(texts, model):
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings
