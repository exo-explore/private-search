from sentence_transformers import SentenceTransformer
import numpy as np
import argparse

class EmbeddingCreator:
    def __init__(self):
        """Initialize the embedding creator with the default model."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Create an embedding for a single text string."""
        return self.model.encode(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create embedding for text')
    parser.add_argument('text', type=str, help='Text to create embedding for')
    
    args = parser.parse_args()
    
    creator = EmbeddingCreator()
    embedding = creator.create_embedding(args.text)
    
    # Print the embedding as a space-separated list of numbers
    print(' '.join(map(str, embedding)))