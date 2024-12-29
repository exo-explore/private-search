# Private Market Data Search

A privacy-preserving search system for market prices using PIR (Private Information Retrieval).

## Setup

```bash
pip install numpy yfinance requests sentence-transformers
```

## Usage

1. Start the servers:
```bash
# Terminal 1
python embedding_server.py

# Terminal 2
python article_server.py
```

2. Run the client:
```bash
# Terminal 3
python client.py
```

3. Enter search queries to find market prices. Example queries:
   - "bitcoin price"
   - "tesla stock"
   - "nvidia shares"

The system automatically updates prices every minute. 