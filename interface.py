import streamlit as st
from retrieval import retrieve_evidence
from embedder import embed_text
from classifier import classify_consistency

st.title("📚 Novel Consistency Checker — Chat Interface")

user_claim = st.text_input("Enter a claim to check against the novel:")

if st.button("Check Consistency"):
    evidence = retrieve_evidence(user_claim, embed_text)
    result = classify_consistency(user_claim, evidence)

    st.subheader("Result:")
    st.write(result)

    st.subheader("Retrieved Evidence:")
    for m in evidence:
        st.write(m.metadata["text"])
        st.write("---")
