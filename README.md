# EXO Private Search

A privacy-preserving search system based on [MIT's Tiptoe paper](https://people.csail.mit.edu/nickolai/papers/henzinger-tiptoe.pdf). This implementation allows you to search through data while maintaining query privacy - the server never learns what you're searching for. For more details, see our [blog](https://blog.exolabs.net/day-8/).

## Features
- Privacy-preserving search using PIR
- Local embedding generation
- Clustering-based optimization for faster searches
- FastAPI-based REST API
- Interactive API documentation at `/docs`

## How It Works

1. Documents are converted into embeddings and clustered for efficient searching
2. The client downloads cluster centroids (~32 kB for a 1 GB database)
3. The client locally compares query vectors to centroids to find relevant clusters
4. Using SimplePIR, the client privately retrieves matching documents
5. All queries remain private - the server never sees what you're searching for

![Architecture Animations](architecture.svg)

*Architecture overview of private search with homomorphic encryption. The query is encrypted before being sent to the server, which processes it without being able to see the contents. The encrypted results are sent back to the client for decryption.*


## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python server.py
```

2. Run the client:
```bash
python client.py
```

3. Enter natural language queries to search through the documents privately.

## API Documentation

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t private-search .
```

2. Run the container:
```bash
docker run -p 8000:8000 private-search
```

## Architecture

The system uses a combination of:
- Sentence transformers for embedding generation
- K-means clustering for search optimization
- SimplePIR for private information retrieval
- FastAPI for the REST API interface

## Privacy Guarantees

- Queries are never revealed to the server
- Document retrieval patterns remain private
- All sensitive computations happen client-side
- Server only sees encrypted PIR queries

## Performance

The clustering-based approach provides significant performance improvements:
- Reduces the number of PIR operations needed
- Allows for efficient searching in large document collections
- Maintains privacy while providing fast results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License. 