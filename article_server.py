import asyncio
import json
import numpy as np
import os
from pir import (
    SimplePIRParams, gen_params, gen_hint,
    answer as pir_answer
)
from utils import strings_to_matrix

class ArticleServer:
    def __init__(self, host='127.0.0.1', port=8889):  # Note different default port
        self.host = host
        self.port = port
        
        # Load and process article contents
        print("Loading article contents database...")
        articles = []
        article_dir = 'articles'
        
        # Load metadata to ensure correct article order
        with open('embeddings/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load articles in the same order as embeddings
        for article_info in metadata['articles']:
            with open(article_info['filepath'], 'r', encoding='utf-8') as f:
                articles.append(f.read())
        
        # Convert articles to matrix
        self.articles_db, matrix_size = strings_to_matrix(articles)
        self.articles_params = gen_params(m=matrix_size)
        self.articles_hint = gen_hint(self.articles_params, self.articles_db)
        self.num_articles = len(articles)
        
        print(f"Initialized with {len(articles)} articles")
        print(f"Articles matrix size: {matrix_size}x{matrix_size}")
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info('peername')
        print(f"New connection from {addr}")
        
        try:
            # Send setup data
            setup_data = {
                'params': {
                    'n': int(self.articles_params.n),
                    'm': int(self.articles_params.m),
                    'q': int(self.articles_params.q),
                    'p': int(self.articles_params.p),
                    'std_dev': float(self.articles_params.std_dev)
                },
                'hint': self.articles_hint.tolist(),
                'a': self.articles_params.a.tolist(),
                'num_articles': self.num_articles
            }
            
            # Send length-prefixed data
            data = json.dumps(setup_data).encode()
            length = len(data)
            writer.write(f"{length}\n".encode())
            await writer.drain()
            
            writer.write(data)
            await writer.drain()

            while True:
                # Read length-prefixed query
                length = await reader.readline()
                if not length:
                    break
                length = int(length.decode().strip())
                
                # Read query data
                data = await reader.readexactly(length)
                query_data = json.loads(data.decode())
                query = np.array(query_data['query'])
                
                ans = pir_answer(query, self.articles_db, self.articles_params.q)
                
                response = {
                    'answer': ans.tolist()
                }
                
                # Send length-prefixed response
                response_data = json.dumps(response).encode()
                writer.write(f"{len(response_data)}\n".encode())
                await writer.drain()
                
                writer.write(response_data)
                await writer.drain()
                
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
        finally:
            print(f"Connection closed for {addr}")
            writer.close()
            await writer.wait_closed()
    
    async def start(self):
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        
        addr = server.sockets[0].getsockname()
        print(f'Serving articles on {addr}')
        
        async with server:
            await server.serve_forever()

if __name__ == "__main__":
    server = ArticleServer()
    asyncio.run(server.start()) 