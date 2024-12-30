from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import json
from datetime import datetime
import asyncio
from pir import (
    SimplePIRParams, gen_params, gen_hint,
    answer as pir_answer
)
from update import update_embeddings
from utils import strings_to_matrix

app = FastAPI(title="Private Market Data Search")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class PIRQuery(BaseModel):
    query: List[int]

class UpdateRequest(BaseModel):
    type: str = "update"

class SetupResponse(BaseModel):
    params: Dict
    hint: List[List[float]]
    centroids: Optional[List[List[float]]] = None
    metadata: Optional[Dict] = None
    embeddings: Optional[List[List[float]]] = None
    num_articles: Optional[int] = None

class PIRResponse(BaseModel):
    answer: List[int]

class UpdateResponse(BaseModel):
    centroids: List[List[float]]
    metadata: Dict
    embeddings: List[List[float]]

# Server state
class ServerState:
    def __init__(self):
        self.load_data()
        self._update_task = None

    def load_data(self):
        """Load embeddings, metadata, and centroids from disk"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Loading embeddings database...")
        
        # Load embeddings data
        self.embeddings_db = np.load('embeddings/embeddings.npy')
        self.centroids = np.load('embeddings/centroids.npy')
        with open('embeddings/metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize PIR for embeddings
        self.embeddings_params = gen_params(m=self.embeddings_db.shape[0])
        self.embeddings_hint = gen_hint(self.embeddings_params, self.embeddings_db)
        
        # Load articles
        articles = []
        for article_info in self.metadata['articles']:
            with open(article_info['filepath'], 'r', encoding='utf-8') as f:
                articles.append(f.read())
        
        # Convert articles to matrix
        self.articles_db, matrix_size = strings_to_matrix(articles)
        self.articles_params = gen_params(m=matrix_size)
        self.articles_hint = gen_hint(self.articles_params, self.articles_db)
        self.num_articles = len(articles)
        
        print(f"[{timestamp}] Embeddings shape: {self.embeddings_db.shape}")
        print(f"[{timestamp}] Centroids shape: {self.centroids.shape}")
        print(f"[{timestamp}] Articles loaded: {self.num_articles}")

    async def update_loop(self):
        """Periodically update data"""
        while True:
            await asyncio.sleep(60)  # Wait for 1 minute
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] Starting scheduled update...")
                update_embeddings()
                self.load_data()
                print(f"[{timestamp}] Update complete!")
            except Exception as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Error during update: {e}")

# Initialize server state
state = ServerState()

@app.on_event("startup")
async def startup_event():
    """Start the update loop when the application starts"""
    state._update_task = asyncio.create_task(state.update_loop())

@app.on_event("shutdown")
async def shutdown_event():
    """Cancel the update loop when the application shuts down"""
    if state._update_task:
        state._update_task.cancel()
        try:
            await state._update_task
        except asyncio.CancelledError:
            pass

@app.get("/")
async def root():
    return {"message": "Private Market Data Search API"}

@app.get("/embedding/setup", response_model=SetupResponse)
async def embedding_setup():
    """Get initial setup data for embeddings"""
    return {
        'params': {
            'n': int(state.embeddings_params.n),
            'm': int(state.embeddings_params.m),
            'q': int(state.embeddings_params.q),
            'p': int(state.embeddings_params.p),
            'std_dev': float(state.embeddings_params.std_dev),
            'seed': int(state.embeddings_params.seed)
        },
        'hint': state.embeddings_hint.tolist(),
        'centroids': state.centroids.tolist(),
        'metadata': state.metadata,
        'embeddings': state.embeddings_db.tolist()
    }

@app.get("/article/setup", response_model=SetupResponse)
async def article_setup():
    """Get initial setup data for articles"""
    return {
        'params': {
            'n': int(state.articles_params.n),
            'm': int(state.articles_params.m),
            'q': int(state.articles_params.q),
            'p': int(state.articles_params.p),
            'std_dev': float(state.articles_params.std_dev),
            'seed': int(state.articles_params.seed)
        },
        'hint': state.articles_hint.tolist(),
        'num_articles': state.num_articles
    }

@app.post("/embedding/query", response_model=PIRResponse)
async def embedding_query(query: PIRQuery):
    """Handle PIR query for embeddings"""
    query_array = np.array(query.query)
    ans = pir_answer(query_array, state.embeddings_db, state.embeddings_params.q)
    return {'answer': ans.tolist()}

@app.post("/article/query", response_model=PIRResponse)
async def article_query(query: PIRQuery):
    """Handle PIR query for articles"""
    query_array = np.array(query.query)
    ans = pir_answer(query_array, state.articles_db, state.articles_params.q)
    return {'answer': ans.tolist()}

@app.post("/embedding/update", response_model=UpdateResponse)
async def embedding_update(request: UpdateRequest):
    """Get updated embedding data"""
    return {
        'centroids': state.centroids.tolist(),
        'metadata': state.metadata,
        'embeddings': state.embeddings_db.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 