from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def classify_consistency(claim, evidence):
    prompt = f"""
You are a consistency evaluator.

Given the following claim:

Claim: "{claim}"

And the retrieved evidence from the novel:

Evidence:
{evidence}

Determine if the evidence SUPPORTS, CONTRADICTS, or is UNKNOWN with respect to the claim.

Also provide a confidence score between 0 and 1.

Respond ONLY in JSON with this format:
{{
  "label": "SUPPORT" or "CONTRADICT" or "UNKNOWN",
  "confidence": float
}}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )

    output = response.choices[0].message.content

    # Parse JSON safely
    import json
    result = json.loads(output)

    label = result["label"]
    confidence = result["confidence"]

    return label, confidence
