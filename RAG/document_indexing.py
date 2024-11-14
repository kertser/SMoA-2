import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import json
import nltk
import warnings

warnings.filterwarnings("ignore")

def main():
    # Initialize the embedding model
    embedding_model = SentenceTransformer('all-mpnet-base-v2')  # Upgraded model

    # Download NLTK data
    nltk.download('punkt')

    # Paths
    DOCUMENTS_PATH = "Docs"
    INDEX_PATH = "passage_index.faiss"
    EMBEDDINGS_PATH = "passage_embeddings.npy"
    METADATA_PATH = "passage_metadata.json"

    passages = []
    metadata = []

    # Process documents and split into passages
    print("Processing documents and splitting into passages...")
    for file_path in tqdm(list(Path(DOCUMENTS_PATH).glob("*.txt"))):
        file_name = file_path.name
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Split content into sentences
            sentences = nltk.sent_tokenize(content)
            # Group sentences into passages (e.g., 5 sentences per passage)
            passage_size = 5
            for idx in range(0, len(sentences), passage_size):
                passage = ' '.join(sentences[idx:idx+passage_size])
                passages.append(passage)
                metadata.append({
                    'file_name': file_name,
                    'passage_idx': idx
                })

    # Generate embeddings in batches for better performance
    print("Encoding passages...")
    embeddings = embedding_model.encode(
        passages,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype('float32')

    # Save embeddings and metadata
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)

    # Create a FAISS index
    embedding_dim = embeddings.shape[1]
    num_embeddings = embeddings.shape[0]

    # Choose index type based on dataset size
    if num_embeddings < 10000:
        # For smaller datasets, use a flat index
        index = faiss.IndexFlatL2(embedding_dim)
    else:
        # For larger datasets, use an inverted file index
        nlist = int(np.sqrt(num_embeddings))  # Number of clusters
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)
        print("Training index...")
        index.train(embeddings)

    # Add embeddings to the index
    index.add(embeddings)

    # Save the index
    faiss.write_index(index, INDEX_PATH)

    print("Indexing completed and files saved.")

if __name__ == "__main__":
    main()
