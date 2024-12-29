import os
import json
import numpy as np
from typing import Dict, List, Tuple
import glob
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import math
from embeddings import EmbeddingCreator
import argparse

def load_article(filepath: str) -> Dict:
    """Load an article from a text file and return its metadata."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extract content and metadata
    content = []
    url = ""
    title = os.path.basename(filepath).replace('.txt', '')
    
    for line in lines:
        line = line.strip()
        if line.startswith('URL:'):
            url = line[4:].strip()
        else:
            content.append(line)
    
    return {
        "title": title,
        "content": "\n".join(content),
        "url": url
    }

def load_price_document(filepath: str) -> Dict:
    """Load a price document and return its metadata."""
    with open(filepath, 'r') as f:
        content = f.read().strip()
    
    return {
        "title": os.path.basename(filepath).replace('.txt', ''),
        "content": content,
        "url": ""  # No URL needed for price documents
    }

def cluster_embeddings(embeddings: np.ndarray, metadata: List[Dict]) -> Tuple[np.ndarray, List[List[Dict]]]:
    """
    Cluster embeddings using k-means with k = sqrt(N).
    Returns centroids and groups of documents.
    """
    n_clusters = max(2, int(math.sqrt(len(embeddings))))
    
    # Normalize embeddings for cosine similarity
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_embeddings)
    
    # Get centroids
    centroids = kmeans.cluster_centers_
    
    # Group documents by cluster
    groups = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(cluster_labels):
        groups[label].append(metadata[idx])
    
    return centroids, groups

def process_articles(input_dir: str, output_dir: str, model_name: str = 'all-MiniLM-L6-v2') -> None:
    """
    Process price documents, create embeddings, and cluster them.
    """
    # Create embedding creator
    embedding_creator = EmbeddingCreator()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all text files
    article_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not article_files:
        print(f"No price documents found in {input_dir}!")
        return
    
    # Process each document
    articles_metadata = []
    all_embeddings = []
    
    for i, filepath in enumerate(article_files, 1):
        try:
            # Load and process document
            doc = load_price_document(filepath)
            
            # Create embedding for the document content
            embedding = embedding_creator.create_embedding(doc["content"])
            
            # Store metadata
            metadata = {
                "title": doc["title"],
                "url": doc["url"],
                "filepath": os.path.relpath(filepath, os.path.dirname(input_dir)),
                "embedding_index": i - 1
            }
            articles_metadata.append(metadata)
            all_embeddings.append(embedding)
            
            print(f"Processed document {i}/{len(article_files)}: {doc['title']}")
            
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(all_embeddings)
    
    # Cluster the embeddings
    centroids, groups = cluster_embeddings(embeddings_array, articles_metadata)
    
    # Save embeddings and metadata
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings_array)
    np.save(os.path.join(output_dir, "centroids.npy"), centroids)
    
    # Add cluster information to metadata
    cluster_metadata = {
        "articles": articles_metadata,
        "groups": [
            {
                "centroid_index": i,
                "articles": group
            }
            for i, group in enumerate(groups)
        ]
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(cluster_metadata, f, indent=2)
    
    # Print clustering summary
    print(f"\nProcessing complete!")
    print(f"Processed {len(articles_metadata)} documents")
    print(f"Created {len(groups)} clusters")
    print(f"Embeddings shape: {embeddings_array.shape}")
    print(f"Centroids shape: {centroids.shape}")
    print("\nCluster sizes:")
    for i, group in enumerate(groups):
        print(f"Cluster {i}: {len(group)} documents")
        print("Documents:", ", ".join(article["title"] for article in group[:3]))
        if len(group) > 3:
            print("...")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and cluster articles')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='Directory containing input text files')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save embeddings and clusters')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                      help='Model name to use for embeddings')
    
    args = parser.parse_args()
    process_articles(args.input_dir, args.output_dir, args.model) 