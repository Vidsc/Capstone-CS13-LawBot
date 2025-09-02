import os, numpy as np
from dotenv import load_dotenv
load_dotenv()

class Embeddings:
    def embed(self, texts): ...
    def dim(self): ...

class LocalEmb(Embeddings):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._dim = 384
    def embed(self, texts):
        arr = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(arr, dtype="float32")
    def dim(self): return self._dim

def get_embeddings() -> Embeddings:
    b = os.environ.get("EMB_BACKEND", "local").lower()
    return LocalEmb()
