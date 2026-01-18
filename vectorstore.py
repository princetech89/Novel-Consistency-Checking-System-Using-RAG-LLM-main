from pinecone import Pinecone, ServerlessSpec
import os, uuid
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in environment variables")

if not INDEX_NAME:
    raise ValueError("INDEX_NAME is not set in environment variables")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072, #TODO: Dimention should be dynamic based on the model
        metric="cosine",
        spec=ServerlessSpec(cloud="AWS", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

def store_chunks(chunks, embed_fn):
    batch_size = 100  # Process chunks in batches of 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        vectors = []
        for chunk in batch_chunks:
            vec = embed_fn(chunk)
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": vec,
                "metadata": {"text": chunk}
            })
        if vectors:
            index.upsert(vectors=vectors)
