# embedder.py
from sentence_transformers import SentenceTransformer
import json
import os

def create_embeddings(text_chunks, model_name="all-MiniLM-L6-v2"):
    """
    Creates embeddings for a list of text chunks using a Sentence Transformer model.
    :param text_chunks: List of dictionaries, each with 'id' and 'text'.
    :param model_name: Name of the Sentence Transformer model to use.
    :return: List of dictionaries, each with 'id', 'text', 'source_url', and 'embedding'.
    """
    print(f"Loading Sentence Transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    print("Model loaded.")

    embedded_chunks = []
    for chunk in text_chunks:
        text = chunk['text']
        embedding = model.encode(text).tolist() # Convert numpy array to list for JSON serialization
        embedded_chunks.append({
            "id": chunk['id'],
            "text": text,
            "source_url": chunk.get('source_url', 'N/A'), # Include source_url if available
            "embedding": embedding
        })
    print(f"Created embeddings for {len(embedded_chunks)} chunks.")
    return embedded_chunks

if __name__ == "__main__":
    input_filename = "cleaned_website_content.json"
    output_filename = "embedded_content.json"

    if not os.path.exists(input_filename):
        print(f"Error: {input_filename} not found. Please run scraper.py first.")
    else:
        with open(input_filename, 'r', encoding='utf-8') as f:
            cleaned_data = json.load(f)

        if cleaned_data:
            embedded_data = create_embeddings(cleaned_data)
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(embedded_data, f, ensure_ascii=False, indent=4)
            print(f"Embeddings saved to {output_filename}")
        else:
            print("No cleaned data found to embed.")