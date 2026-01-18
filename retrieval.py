from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in environment variables")

if not INDEX_NAME:
    raise ValueError("INDEX_NAME is not set in environment variables")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def retrieve_evidence(claim, embed_fn, top_k=5):
    vec = embed_fn(claim)
    results = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches
