import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain_ollama import OllamaLLM
from urllib.parse import urlparse

# Load AI Model (Ollama must be running locally with 'mistral' model)
llm = OllamaLLM(model="mistral")

# Helper: Validate URL
def is_valid_url(url):
    parsed = urlparse(url)
    return all([parsed.scheme, parsed.netloc])

# Function to scrape website content
def scrape_website(url):
    try:
        st.info(f"🌍 Scraping website: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return f"⚠️ Failed to fetch {url} (Status code: {response.status_code})"
        
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text(strip=True) for p in paragraphs])

        return text if text else "⚠️ No paragraph text found on the page."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Function to summarize content using AI
def summarize_content(content):
    try:
        st.info("✍️ Summarizing content using AI...")
        prompt = f"Summarize this website content:\n\n{content[:1000]}"
        return llm.invoke(prompt)
    except Exception as e:
        return f"❌ AI summarization failed: {str(e)}"

# Streamlit UI
st.title("🤖 AI-Powered Website Summarizer")
st.write("Enter a valid URL to fetch and summarize content using AI (Mistral via Ollama).")

# Input field
url = st.text_input("🔗 Enter website URL (e.g., https://example.com):")

# Handle request
if url:
    if not is_valid_url(url):
        st.error("🚫 Please enter a valid URL (must include https://)")
    else:
        with st.spinner("🔍 Fetching and summarizing..."):
            content = scrape_website(url)

            if content.startswith("⚠️") or content.startswith("❌"):
                st.error(content)
            else:
                # Optional: show part of original content
                with st.expander("📝 Preview Extracted Text"):
                    st.write(content[:1500] + "..." if len(content) > 1500 else content)
                
                # Summarize
                summary = summarize_content(content)
                st.subheader("📄 AI Summary")
                st.write(summary)
                st.success("✅ Summary generated successfully!")