import asyncio
import json
import numpy as np
from pir import (
    SimplePIRParams, gen_params, gen_hint,
    answer as pir_answer
)

class EmbeddingServer:
    def __init__(self, host='127.0.0.1', port=8888):
        self.host = host
        self.port = port
        
        # Load embeddings database
        print("Loading embeddings database...")
        self.embeddings_db = np.load('embeddings/embeddings.npy')
        
        # Initialize PIR for embeddings
        self.embeddings_params = gen_params(m=self.embeddings_db.shape[0])
        self.embeddings_hint = gen_hint(self.embeddings_params, self.embeddings_db)
        
        print(f"Embeddings shape: {self.embeddings_db.shape}")
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info('peername')
        print(f"New connection from {addr}")
        
        try:
            # Send setup data
            setup_data = {
                'params': {
                    'n': int(self.embeddings_params.n),
                    'm': int(self.embeddings_params.m),
                    'q': int(self.embeddings_params.q),
                    'p': int(self.embeddings_params.p),
                    'std_dev': float(self.embeddings_params.std_dev)
                },
                'hint': self.embeddings_hint.tolist(),
                'a': self.embeddings_params.a.tolist(),
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
                
                ans = pir_answer(query, self.embeddings_db, self.embeddings_params.q)
                
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
        print(f'Serving embeddings on {addr}')
        
        async with server:
            await server.serve_forever()

if __name__ == "__main__":
    server = EmbeddingServer()
    asyncio.run(server.start())
