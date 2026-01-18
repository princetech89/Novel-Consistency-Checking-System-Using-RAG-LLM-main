import csv
import os
from retrieval import retrieve_evidence
from classifier import classify_consistency
from embedder import embed_text

def process_claims(input_csv, output_csv):
    results = []

    if not os.path.exists(input_csv):
        print(f"❌ Input file not found: {input_csv}")
        return
    
    print(f"📥 Loading claims from: {input_csv}")

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            claim = row[0].strip()
            print(f"\n🔍 Processing claim: {claim}")

            evidence = retrieve_evidence(claim, embed_text, top_k=5)

            result, confidence = classify_consistency(claim, evidence)

            results.append([claim, result, confidence, evidence])

    print(f"\n💾 Saving results to: {output_csv}")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["claim", "result", "confidence", "evidence"])
        writer.writerows(results)

    print("\n✅ Processing complete!")


if __name__ == "__main__":
    input_csv = "data/claims.csv"
    output_csv = "data/claims_results.csv"

    process_claims(input_csv, output_csv)
