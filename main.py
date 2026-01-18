from chunker import chunk_text
from embedder import embed_text
from vectorstore import store_chunks
from classifier import extract_claims, classify_consistency
import os

def index_novel():
    novel_path = "data/The Count of Monte Cristo.txt"

    if not os.path.exists(novel_path):
        raise FileNotFoundError(f"Novel file not found at {novel_path}")

    print("📘 Loading novel...")
    novel = open(novel_path, "r", encoding="utf-8").read()

    print("✂️ Chunking novel...")
    chunks = chunk_text(novel)

    print("📥 Storing chunks into Pinecone...")
    store_chunks(chunks, embed_text)

    print("✅ Novel successfully indexed in Pinecone!")

if __name__ == "__main__":
    index_novel()
