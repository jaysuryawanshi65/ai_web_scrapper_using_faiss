import requests
from bs4 import BeautifulSoup
import streamlit as st
import faiss
import numpy as np
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Load AI model (only once)
llm = OllamaLLM(model="mistral")

# FAISS index and context memory
index = faiss.IndexFlatL2(384)  # 384-dim for MiniLM
context_store = []  # Stores (url, text_chunk) tuples

# ✅ Lazy-load the embedding model (fixes UI freeze)
@st.cache_resource
def load_embeddings():
    st.write("📦 Loading HuggingFace embeddings model...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Scrape website
def scrape_website(url):
    try:
        st.info(f"🌍 Scraping: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"⚠️ Failed to fetch (Status code: {response.status_code})"
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text(strip=True) for p in paragraphs])[:5000]
    except Exception as e:
        return f"❌ Error: {e}"

# Store text in FAISS
def store_in_faiss(text, url):
    global index, context_store
    st.info("📦 Embedding and storing text...")

    embeddings = load_embeddings()  # Lazy-load here
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    vectors = embeddings.embed_documents(chunks)
    vectors = np.array(vectors).astype(np.float32)

    index.add(vectors)
    for chunk in chunks:
        context_store.append((url, chunk))

    return f"✅ Stored {len(chunks)} chunks in FAISS."

# Retrieve and answer question
def retrieve_and_answer(query):
    global index, context_store

    if index.ntotal == 0:
        return "⚠️ No documents stored yet."

    embeddings = load_embeddings()  # Lazy-load here too
    query_vec = np.array(embeddings.embed_query(query)).astype(np.float32).reshape(1, -1)
    D, I = index.search(query_vec, k=3)

    context = ""
    used_chunks = set()
    for i in I[0]:
        if i < len(context_store):
            url, chunk = context_store[i]
            if chunk not in used_chunks:
                context += chunk + "\n\n"
                used_chunks.add(chunk)

    if not context:
        return "🤖 No relevant context found."

    prompt = f"Based on the context below, answer the question:\n\n{context}\nQuestion: {query}\nAnswer:"
    return llm.invoke(prompt)

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="AI Web Scraper + Q&A", layout="centered")
st.title("🤖 AI-Powered Web Scraper + Q&A using FAISS")
st.markdown("🔗 Enter a website URL to store its knowledge, then ask questions about it!")

# Input for website
url = st.text_input("🌐 Enter Website URL:")
if url:
    content = scrape_website(url)
    if "⚠️" in content or "❌" in content:
        st.error(content)
    else:
        with st.spinner("Embedding and storing..."):
            message = store_in_faiss(content, url)
            st.success(message)

# Input for question
query = st.text_input("❓ Ask your question about stored content:")
if query:
    with st.spinner("🔍 Retrieving answer..."):
        answer = retrieve_and_answer(query)
        st.subheader("🤖 Answer:")
        st.write(answer)
