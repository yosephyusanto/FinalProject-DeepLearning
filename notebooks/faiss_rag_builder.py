import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent  # root project, because streamlit is executed from app/
FAISS_DIR = BASE_DIR / "data" / "faiss"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

class DrugRAGRetriever:
  def __init__(self, faiss_dir=None):
    if faiss_dir is None:
      faiss_dir = FAISS_DIR
    
    self.faiss_dir = Path(faiss_dir)

    # load config
    with open(self.faiss_dir / "config.json", "r") as f:
      self.config = json.load(f)
    
    # load model 
    print(f"Loading embedding model: {self.config['embedding_model']}")
    self.model = SentenceTransformer(self.config['embedding_model'])
    
    # Load FAISS index
    self.index = faiss.read_index(str(self.faiss_dir / "drug_knowledge.index"))
    
    # Load metadata
    with open(self.faiss_dir / "metadata.pkl", 'rb') as f:
        self.metadata = pickle.load(f)
    
    print(f"Loaded RAG retriever with {len(self.metadata)} chunks")
  
  def retrieve(self, query, k=3):
    # Retrieve top-k relevant chunks for a query
    # Encode query
    query_embedding = self.model.encode([query])
    
    # Search
    distances, indices = self.index.search(query_embedding, k)
    
    # Get results
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(self.metadata):
            result = self.metadata[idx].copy()
            result['distance'] = float(dist)
            results.append(result)
    
    return results
    
  def format_context(self, results):
    # Format retrieved results into context string
    context_parts = []
    
    for i, result in enumerate(results):
        context_parts.append(
            f"[Source {i+1}] {result['drug_name']} - {result['category']}:\n"
            f"{result['text']}\n"
        )
    
    return "\n".join(context_parts)