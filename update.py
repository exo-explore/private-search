import os
from market_prices import get_market_prices
from clustering import process_articles

def update_embeddings():
    """Updates price documents and rebuilds embeddings."""
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    articles_dir = os.path.join(base_dir, "articles")
    embeddings_dir = os.path.join(base_dir, "embeddings")
    os.makedirs(articles_dir, exist_ok=True)
    
    # Get prices and create documents
    stock_prices, crypto_prices, timestamp = get_market_prices()
    
    # Write stock documents
    for name, price in stock_prices.items():
        filename = name.replace(' ', '_').replace('.', '').replace(',', '').replace('(', '').replace(')', '') + '.txt'
        with open(os.path.join(articles_dir, filename), 'w') as f:
            f.write(f"{name}: ${price if price else 'N/A'}")
    
    # Write crypto documents
    for name, price in crypto_prices.items():
        filename = name.replace(' ', '_').replace('.', '').replace(',', '').replace('(', '').replace(')', '') + '.txt'
        with open(os.path.join(articles_dir, filename), 'w') as f:
            f.write(f"{name}: ${price if price else 'N/A'}")
    
    # Rebuild embeddings
    process_articles(articles_dir, embeddings_dir)

if __name__ == "__main__":
    update_embeddings()
