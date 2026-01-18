from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["INDEX_NAME"])

def retrieve_evidence(claim, embed_fn, top_k=5):
    vec = embed_fn(claim)
    results = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches
