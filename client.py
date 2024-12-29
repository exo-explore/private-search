import asyncio
import json
import numpy as np
from datetime import datetime
from typing import List, Optional, Union, Dict, Tuple
from embeddings import EmbeddingCreator
from pir import (
    SimplePIRParams, gen_secret, query as pir_query,
    recover as pir_recover, recover_row as pir_recover_row
)
from utils import numbers_to_string

class PIRClient:
    def __init__(self, embedding_host='127.0.0.1', embedding_port=8888,
                 article_host='127.0.0.1', article_port=8889):
        self.embedding_host = embedding_host
        self.embedding_port = embedding_port
        self.embedding_reader: Optional[asyncio.StreamReader] = None
        self.embedding_writer: Optional[asyncio.StreamWriter] = None
        
        self.article_host = article_host
        self.article_port = article_port
        self.article_reader: Optional[asyncio.StreamReader] = None
        self.article_writer: Optional[asyncio.StreamWriter] = None
        
        self.embeddings_params: Optional[SimplePIRParams] = None
        self.embeddings_hint: Optional[np.ndarray] = None
        self.embeddings_secret: Optional[np.ndarray] = None
        
        self.articles_params: Optional[SimplePIRParams] = None
        self.articles_hint: Optional[np.ndarray] = None
        self.articles_secret: Optional[np.ndarray] = None
        self.num_articles: Optional[int] = None

        self._embedding_creator = EmbeddingCreator()
        
        # New attributes for embeddings and metadata
        self.embeddings: Optional[np.ndarray] = None
        self.centroids: Optional[np.ndarray] = None
        self.metadata: Optional[Dict] = None
        self._update_task: Optional[asyncio.Task] = None
    
    async def _send_data(self, writer: asyncio.StreamWriter, data: dict):
        encoded = json.dumps(data).encode()
        writer.write(f"{len(encoded)}\n".encode())
        await writer.drain()
        writer.write(encoded)
        await writer.drain()
    
    async def _receive_data(self, reader: asyncio.StreamReader) -> dict:
        length = int((await reader.readline()).decode().strip())
        data = await reader.readexactly(length)
        return json.loads(data.decode())
    
    async def _update_loop(self):
        """Periodically request updated data from the server"""
        while True:
            await asyncio.sleep(60)  # Wait for 1 minute
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] Requesting updated data from server...")
                await self._send_data(self.embedding_writer, {'type': 'update'})
                data = await self._receive_data(self.embedding_reader)
                
                self.embeddings = np.array(data['embeddings'])
                self.centroids = np.array(data['centroids'])
                self.metadata = data['metadata']
                print(f"[{timestamp}] Successfully updated embeddings and metadata from server")
                print(f"[{timestamp}] New embeddings shape: {self.embeddings.shape}")
                print(f"[{timestamp}] New centroids shape: {self.centroids.shape}")
            except Exception as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Error updating data: {e}")
    
    async def retrieve_embedding(self, index: int) -> np.ndarray:
        if any(x is None for x in [self.embeddings_params, self.embeddings_secret, self.embeddings_hint]):
            raise RuntimeError("Client not connected. Call connect() first.")
        
        secret, query_cipher = pir_query(index, self.embeddings_params.m, self.embeddings_params)
        
        await self._send_data(self.embedding_writer, {
            'query': query_cipher.tolist()
        })
        
        response = await self._receive_data(self.embedding_reader)
        answer_cipher = np.array(response['answer'])
        
        recovered = pir_recover_row(secret, self.embeddings_hint, answer_cipher, 
                                  query_cipher, self.embeddings_params)
        return recovered
    
    async def retrieve_article(self, index: int) -> str:
        if any(x is None for x in [self.articles_params, self.articles_secret, self.articles_hint]):
            raise RuntimeError("Client not connected. Call connect() first.")
        
        if index >= self.num_articles:
            raise ValueError(f"Article index {index} out of range (max {self.num_articles-1})")
        
        secret, query_cipher = pir_query(index, self.articles_params.m, self.articles_params)
        
        await self._send_data(self.article_writer, {
            'query': query_cipher.tolist()
        })
        
        response = await self._receive_data(self.article_reader)
        answer_cipher = np.array(response['answer'])
        
        recovered_row = pir_recover_row(secret, self.articles_hint, answer_cipher, 
                                      query_cipher, self.articles_params)
        
        length = int(recovered_row[0])
        if length > 0:
            numbers = recovered_row[1:length+1].astype(np.int64).tolist()
            return numbers_to_string(numbers)
        return ""
    
    async def connect(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Connecting to servers...")
        
        self.embedding_reader, self.embedding_writer = await asyncio.open_connection(
            self.embedding_host, self.embedding_port
        )
        
        self.article_reader, self.article_writer = await asyncio.open_connection(
            self.article_host, self.article_port
        )
        
        print(f"[{timestamp}] Connected to servers, downloading initial data...")
        
        emb_data = await self._receive_data(self.embedding_reader)
        self.embeddings_params = SimplePIRParams(
            a=np.array(emb_data['a']),
            q=emb_data['params']['q'],
            p=emb_data['params']['p'],
            n=emb_data['params']['n'],
            m=emb_data['params']['m'],
            std_dev=emb_data['params']['std_dev']
        )
        self.embeddings_hint = np.array(emb_data['hint'])
        self.embeddings_secret = gen_secret(self.embeddings_params.q, self.embeddings_params.n)
        
        # Store embeddings and metadata
        self.embeddings = np.array(emb_data['embeddings'])
        self.centroids = np.array(emb_data['centroids'])
        self.metadata = emb_data['metadata']
        
        art_data = await self._receive_data(self.article_reader)
        self.articles_params = SimplePIRParams(
            a=np.array(art_data['a']),
            q=art_data['params']['q'],
            p=art_data['params']['p'],
            n=art_data['params']['n'],
            m=art_data['params']['m'],
            std_dev=art_data['params']['std_dev']
        )
        self.articles_hint = np.array(art_data['hint'])
        self.articles_secret = gen_secret(self.articles_params.q, self.articles_params.n)
        self.num_articles = art_data['num_articles']
        
        # Start the update loop
        self._update_task = asyncio.create_task(self._update_loop())
    
    def find_closest_embedding(self, query_embedding: np.ndarray, embeddings: Optional[np.ndarray] = None) -> int:
        if embeddings is None:
            embeddings = self.embeddings
        distances = np.linalg.norm(embeddings - query_embedding, axis=1)
        return np.argmin(distances)
    
    async def close(self):
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        if self.embedding_writer:
            self.embedding_writer.close()
            await self.embedding_writer.wait_closed()
        if self.article_writer:
            self.article_writer.close()
            await self.article_writer.wait_closed()

async def main():
    import time

    client = PIRClient()

    try:
        print("Connecting to servers...")
        await client.connect()
        print("Connected! Enter your queries (press Ctrl+C to exit):")
        
        while True:
            try:
                query = input("\nEnter your query: ").strip()
                if not query:
                    continue
                
                start_time = time.time()
                
                # Create embedding and find closest match
                query_embedding = client._embedding_creator.create_embedding(query)
                closest_embedding_index = client.find_closest_embedding(query_embedding)
                
                # Retrieve the article
                article = await client.retrieve_article(closest_embedding_index)
                
                end_time = time.time()
                duration = end_time - start_time
                
                print(f"\nResults (retrieved in {duration:.2f} seconds):")
                print("-" * 80)
                print(article)
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")