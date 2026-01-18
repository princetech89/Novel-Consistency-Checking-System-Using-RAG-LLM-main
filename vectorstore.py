from pinecone import Pinecone, ServerlessSpec
import os, uuid

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ["INDEX_NAME"]

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="AWS", region="us-east-1")
    )

index = pc.Index(index_name)

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
