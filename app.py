import streamlit as st
from modules.chat import ask_ai
from modules.document_reader import read_pdf, read_docx
import os
import PyPDF2
from modules.paper_search import search_arxiv
from modules.data_analysis import analyze_csv
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from pdf_export import generate_pdf
import tempfile



st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("📚 AI Research Assistant")

# Chat Section
st.header("💬 Ask AI")
user_query = st.text_input("Enter your research question:")
if st.button("Ask AI") and user_query:
    with st.spinner("Thinking..."):
        response = ask_ai(user_query)
    st.subheader("AI Response")
    st.write(response)

from modules.paper_search import search_arxiv

# Research Paper Search Section
st.header("🔍 Search for Latest Research Papers")
search_topic = st.text_input("Enter topic for research paper search:")

if st.button("Search Papers") and search_topic:
    with st.spinner("Searching arXiv..."):
        papers = search_arxiv(search_topic, max_results=5)
    
    if papers and "error" not in papers[0]:
        for idx, paper in enumerate(papers, start=1):
            st.markdown(f"### {idx}. {paper['title']}")
            st.markdown(f"**Link:** {paper['link']}")
            st.write(paper['summary'])

            if st.button(f"Summarize Paper {idx}"):
                with st.spinner("Summarizing with AI..."):
                    summary = ask_ai(f"Summarize this research paper in an academic tone:\n{paper['summary']}")
                st.markdown("**AI Summary:**")
                st.write(summary)
    else:
        st.error("No papers found or API error.")

# Document Summarization Section
st.header("📄 Upload Research Paper for Summarization")
uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

if uploaded_file:
    # Save file locally
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text
    if uploaded_file.name.endswith(".pdf"):
        text = read_pdf(file_path)
    else:
        text = read_docx(file_path)

    if text.strip():
        with st.spinner("Summarizing document..."):
            summary = ask_ai(f"Summarize the following research paper in an academic tone:\n{text[:3000]}")
        st.subheader("Document Summary")
        st.write(summary)
    else:
        st.warning("No text could be extracted from the document.")
    
# Data Analysis Section
st.header("📊 Research Dataset Analysis")
dataset_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if dataset_file:
    os.makedirs("data", exist_ok=True)
    dataset_path = os.path.join("data", dataset_file.name)
    with open(dataset_path, "wb") as f:
        f.write(dataset_file.getbuffer())

    with st.spinner("Analyzing dataset..."):
        info, charts = analyze_csv(dataset_path)
    
    st.subheader("Dataset Information")
    st.write(f"Shape: {info['shape']}")
    st.write(f"Columns: {info['columns']}")
    st.write("Missing Values:", info["missing_values"])
    
    st.subheader("Summary Statistics")
    st.json(info["summary_stats"])
    
    st.subheader("Charts")
    for chart in charts:
        st.image(chart, use_column_width=True)

# Load models (use lightweight ones for local testing)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to create FAISS index from text
def create_faiss_index(text_chunks):
    embeddings = embedder.encode(text_chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Function to retrieve most relevant chunks
def retrieve_chunks(query, index, text_chunks, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [text_chunks[i] for i in indices[0]]

# Streamlit UI
st.title("📚 Research Assistant - Question Answering")
uploaded_pdf = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

if uploaded_pdf is not None:
    text = extract_text_from_pdf(uploaded_pdf)
    text_chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    index, _ = create_faiss_index(text_chunks)
    st.success("✅ PDF processed successfully. You can now ask questions.")

    query = st.text_input("Ask a question about the document:")
    if query:
        relevant_chunks = retrieve_chunks(query, index, text_chunks)
        context = " ".join(relevant_chunks)
        answer = qa_pipeline(question=query, context=context)
        st.write("**Answer:**", answer['answer'])

# Load summarization model (you can replace with your own)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def summarize_text(text):
    # Limit to 1024 tokens for summarization
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)

def extract_citations(text):
    import re
    citations = re.findall(r'\[(\d+)\]', text)  # Matches [1], [2], etc.
    return list(set(citations))

# Streamlit UI
st.subheader("📄 Research Paper Summarization & Citation Extractor")

uploaded_pdf = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])

if uploaded_pdf:
    with st.spinner("Extracting text..."):
        extracted_text = extract_text_from_pdf(uploaded_pdf)
    
    st.write("✅ **Text Extracted**")
    st.text_area("Extracted Text", extracted_text[:2000] + "...", height=200)
    
    if st.button("Summarize Paper"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(extracted_text)
        st.success("📌 Summary:")
        st.write(summary)
    
    if st.button("Extract Citations"):
        citations = extract_citations(extracted_text)
        st.success(f"Found {len(citations)} citations:")
        st.write(citations)

# Inside your main app code after analysis
if st.button("Generate PDF Report"):
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "research_report.pdf")
    
    # Example: 'summary_text' from AI output, 'chart_files' list from saved matplotlib charts
    pdf_file = generate_pdf(summary_text, chart_files, output_path=pdf_path)
    
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download PDF",
            data=f,
            file_name="research_report.pdf",
            mime="application/pdf"
        )