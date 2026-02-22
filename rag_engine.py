from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline


def create_rag(pdf_path):

    # ----------------------------
    # 1️⃣ Load PDF
    # ----------------------------
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # ----------------------------
    # 2️⃣ Split into chunks
    # ----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    texts = splitter.split_documents(documents)

    # ----------------------------
    # 3️⃣ Embedding model (CPU friendly)
    # ----------------------------
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ----------------------------
    # 4️⃣ Create / reuse vector DB
    # ----------------------------
    client = chromadb.Client()

    collection = client.get_or_create_collection("research_paper")

    # Add documents only if collection is empty
    if collection.count() == 0:
        for i, doc in enumerate(texts):
            embedding = embed_model.encode(doc.page_content).tolist()

            collection.add(
                documents=[doc.page_content],
                embeddings=[embedding],
                ids=[str(i)]
            )

    # ----------------------------
    # 5️⃣ Local LLM (FREE)
    # ----------------------------
    generator = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    max_length=256,
    temperature=0.2
)

    # ======================================================
    # 🧠 FEATURE 1 — Ask Questions
    # ======================================================
    def ask(question):

        q_embed = embed_model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[q_embed],
            n_results=3
        )

        context = "\n".join(results["documents"][0])

        prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

        response = generator(prompt)[0]["generated_text"]
        return response

    # ======================================================
    # 🧠 FEATURE 2 — Smart Summary
    # ======================================================
    def summarize():

        results = collection.get()
        text = " ".join(results["documents"])

        prompt = f"""
Give a clear research paper summary including:
- Main problem
- Method used
- Key contributions
- Results
- Why it matters

Paper Content:
{text[:4000]}
"""

        return generator(prompt)[0]["generated_text"]

    # ======================================================
    # 📌 FEATURE 3 — Key Contributions
    # ======================================================
    def contributions():

        results = collection.get()
        text = " ".join(results["documents"])

        prompt = f"""
List the main contributions of this research paper as bullet points.

Paper Content:
{text[:4000]}
"""

        return generator(prompt)[0]["generated_text"]

    # ======================================================
    # 🧒 FEATURE 4 — Explain Simply
    # ======================================================
    def explain_simple():

        results = collection.get()
        text = " ".join(results["documents"])

        prompt = f"""
Explain this research paper in very simple terms for a beginner.

Paper Content:
{text[:3000]}
"""

        return generator(prompt)[0]["generated_text"]

    # Return all capabilities
    return ask, summarize, contributions, explain_simple