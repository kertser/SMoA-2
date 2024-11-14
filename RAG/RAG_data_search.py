import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline, AutoTokenizer
from functools import lru_cache
from pathlib import Path
import json
import nltk
import warnings

warnings.filterwarnings("ignore")

def main():
    # Initialize models
    embedding_model = SentenceTransformer('all-mpnet-base-v2')  # Upgraded embedding model
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # For re-ranking
    qa_model_name = 'deepset/roberta-base-squad2'  # Define the QA model name
    qa_pipeline = pipeline("question-answering", model=qa_model_name)
    tokenizer = AutoTokenizer.from_pretrained(qa_model_name)  # Use the model name here
    max_length = tokenizer.model_max_length

    # Load the FAISS index and metadata
    INDEX_PATH = "passage_index.faiss"
    METADATA_PATH = "passage_metadata.json"

    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print("Index or metadata not found. Please run the indexing script first.")
        return

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        passage_metadata = json.load(f)

    # Interactive query loop
    while True:
        query = input("Enter query (or 'exit' to end): ")
        if query.lower() == 'exit':
            break
        answer = generate_answer(query, embedding_model, index, passage_metadata, qa_pipeline, cross_encoder, tokenizer, max_length)
        print(f"Answer: {answer}\n")

@lru_cache(maxsize=128)
def get_passage_content(file_name, passage_idx, passage_size=5):
    with open(f"Docs/{file_name}", "r", encoding="utf-8") as f:
        content = f.read()
        sentences = nltk.sent_tokenize(content)
        passage = ' '.join(sentences[passage_idx:passage_idx+passage_size])
        return passage

def search_similar_passages(query, embedding_model, index, passage_metadata, k=20):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype('float32')

    distances, indices = index.search(query_embedding, k)
    results = []
    for idx in indices[0]:
        metadata = passage_metadata[idx]
        file_name = metadata['file_name']
        passage_idx = metadata['passage_idx']
        passage_content = get_passage_content(file_name, passage_idx)
        results.append({
            'file_name': file_name,
            'passage_idx': passage_idx,
            'content': passage_content
        })
    return results

def re_rank_passages(query, passages, cross_encoder):
    pairs = [(query, passage['content']) for passage in passages]
    scores = cross_encoder.predict(pairs)
    for i, passage in enumerate(passages):
        passage['score'] = scores[i]
    passages.sort(key=lambda x: x['score'], reverse=True)
    return passages

def get_combined_context(passages, tokenizer, max_tokens):
    combined_context = ''
    total_tokens = 0
    for passage in passages:
        passage_tokens = tokenizer.tokenize(passage['content'])
        if total_tokens + len(passage_tokens) > max_tokens - 50:  # Reserve tokens for the question
            break
        combined_context += passage['content'] + ' '
        total_tokens += len(passage_tokens)
    return combined_context.strip()

def generate_answer(query, embedding_model, index, passage_metadata, qa_pipeline, cross_encoder, tokenizer, max_length):
    retrieved_passages = search_similar_passages(query, embedding_model, index, passage_metadata)
    if not retrieved_passages:
        return "Nothing is found"

    # Re-rank passages
    re_ranked_passages = re_rank_passages(query, retrieved_passages, cross_encoder)

    # Combine top passages into context
    combined_context = get_combined_context(re_ranked_passages, tokenizer, max_length)
    if not combined_context:
        return "Context is too long."

    try:
        answer = qa_pipeline(question=query, context=combined_context)
        return answer['answer']
    except Exception as e:
        return "Generation Error."

if __name__ == "__main__":
    main()
