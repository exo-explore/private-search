# Private Market Data Search

A privacy-preserving search system for real-time market prices using PIR (Private Information Retrieval). Get live stock and cryptocurrency prices while maintaining privacy.

## Features
- Live stock prices from major companies (Apple, NVIDIA, Microsoft, etc.)
- Live cryptocurrency prices (Bitcoin, Ethereum, Solana)
- Privacy-preserving queries using PIR
- Automatic price updates every minute
- FastAPI-based REST API
- Interactive API documentation at `/docs`

## Setup

```bash
pip install numpy yfinance requests tabulate sentence-transformers scikit-learn fastapi uvicorn aiohttp
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

3. Enter search queries to find market prices. Example queries:
   - "bitcoin price"
   - "tesla stock"
   - "nvidia shares"
   - "eth price"
   - "apple stock price"

The system automatically updates prices every minute.

## API Documentation

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## Docker Deployment

You can also run the server using Docker:

1. Build the Docker image:
```bash
docker build -t private-search .
```

2. Run the container:
```bash
docker run -p 8000:8000 private-search
```

3. Save the Docker image to a file:
```bash
docker save private-search > private-search.tar
```

4. Load the saved image on another machine:
```bash
docker load < private-search.tar
```

The server will be available at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`. 