# Private Market Data Search

A privacy-preserving search system for real-time market prices using PIR (Private Information Retrieval). Get live stock and cryptocurrency prices while maintaining privacy.

## Features
- Live stock prices from major companies (Apple, NVIDIA, Microsoft, etc.)
- Live cryptocurrency prices (Bitcoin, Ethereum, Solana)
- Privacy-preserving queries using PIR
- Automatic price updates every minute

## Setup

```bash
pip install numpy yfinance requests tabulate sentence-transformers scikit-learn
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
   - "eth price"
   - "apple stock price"

The system automatically updates prices every minute. 