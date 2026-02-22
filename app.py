import streamlit as st
from rag_engine import create_rag

st.set_page_config(page_title="ResearchInsight-AI", layout="wide")

st.title("📄 ResearchInsight-AI")
st.write("AI assistant for understanding research papers")

uploaded_file = st.file_uploader("Upload research paper (PDF)", type="pdf")

if uploaded_file is not None:

    pdf_path = f"temp_{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("Paper processed successfully!")

    ask, summarize, contributions, explain_simple = create_rag(pdf_path)

    st.divider()

    # 🔥 Feature Buttons
    col1, col2, col3 = st.columns(3)

    if col1.button("🧠 Smart Summary"):
        with st.spinner("Analyzing paper..."):
            st.write(summarize())

    if col2.button("📌 Key Contributions"):
        with st.spinner("Extracting contributions..."):
            st.write(contributions())

    if col3.button("🧒 Explain for Beginner"):
        with st.spinner("Simplifying..."):
            st.write(explain_simple())

    st.divider()

    # 🔎 Ask Questions
    question = st.text_input("Ask anything about the paper")

    if question:
        with st.spinner("Thinking..."):
            answer = ask(question)

        st.markdown("### 🔎 Answer")
        st.write(answer)