import os
import json
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  # e.g., "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")    # e.g., "changi-chatbot-index"
EMBEDDED_CONTENT_FILE = "embedded_content.json"
DIMENSION = 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
METRIC = "cosine" # Cosine similarity is common for embeddings

def initialize_pinecone():
    """Initializes Pinecone client and creates the index if it doesn't exist."""
    print("Initializing Pinecone...")
    if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT or not PINECONE_INDEX_NAME:
        raise ValueError("Pinecone API key, environment, or index name not set in .env")

    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT) # Assuming serverless on AWS
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

    return pc.Index(PINECONE_INDEX_NAME)

def upsert_embeddings(index, embedded_data):
    """Upserts embeddings to the Pinecone index."""
    print(f"Upserting {len(embedded_data)} embeddings to Pinecone...")
    # Pinecone upsert expects a list of (id, vector, metadata) tuples
    vectors_to_upsert = []
    for i, item in enumerate(embedded_data):
        # Ensure 'id' is a string and unique
        unique_id = f"chunk-{i}-{item.get('source_url_hash', '')}"
        vectors_to_upsert.append(
            (
                unique_id,
                item["embedding"],
                {"text": item["text"], "source_url": item.get("source_url", "N/A"), "chunk_index": item.get("chunk_index", i)}
            )
        )

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted batch {int(i/batch_size) + 1}/{(len(vectors_to_upsert) + batch_size - 1) // batch_size}")

    print("All embeddings upserted successfully.")

if __name__ == "__main__":
    try:
        # Load embedded content
        with open(EMBEDDED_CONTENT_FILE, 'r', encoding='utf-8') as f:
            embedded_chunks = json.load(f)

        if not embedded_chunks:
            print("No embedded content found. Please run embedder.py first.")
        else:
            pinecone_index = initialize_pinecone()
            upsert_embeddings(pinecone_index, embedded_chunks)
            print("Database manager process complete.")

    except FileNotFoundError:
        print(f"Error: {EMBEDDED_CONTENT_FILE} not found. Please run embedder.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")