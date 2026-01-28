import os
import re
import json
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import altair as alt

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------- CONFIG ----------------
PDF_DIR = "data/pdfs"
INDEX_DIR = "data/faiss"
META_FILE = "data/metadata.json"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

st.set_page_config(page_title="Research Intelligence", page_icon="📚", layout="wide")

# ---------------- UI STYLING ----------------
st.markdown(
    """
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    [data-testid="stHeader"] { height: 0px !important; background: transparent !important; }
    [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 1px solid #222; }
    .main-title { color: #ffffff !important; font-weight: 800; text-transform: uppercase; margin-top: 0px !important; }
    [data-testid="stFileUploader"] { background-color: #000000 !important; }
    [data-testid="stFileUploaderDropzone"] { background-color: #000000 !important; border: 1px dashed #E50914 !important; }
    [data-testid="stFileUploader"] button { background-color: #000000 !important; color: white !important; border: 1px solid #333 !important; }
    [data-testid="stFileUploaderDropzone"] div div span { color: white !important; }
    canvas { filter: invert(100%) hue-rotate(180deg) brightness(1.1); }
    div.stButton > button { background-color: #E50914 !important; color: white !important; font-weight: 700 !important; }
    .stSpinner, [data-testid="stStatusWidget"] { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- HELPERS ----------------
def extract_year(text):
    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group()) if match else None

def extract_references(text):
    refs = []
    for line in text.split("\n"):
        if len(line) > 30 and re.search(r"\d{4}", line):
            refs.append(line.strip())
    return refs[:20]

def save_metadata(meta):
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("<h2 style='color: #E50914;'>N-RESEARCH</h2>", unsafe_allow_html=True)
st.sidebar.subheader("📄 Upload Center")

uploaded_files = st.sidebar.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        path = os.path.join(PDF_DIR, f.name)
        with open(path, "wb") as out: out.write(f.getbuffer())
    st.sidebar.success("Library Updated")
    st.cache_resource.clear()

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Library"):
    for f in os.listdir(PDF_DIR): os.remove(os.path.join(PDF_DIR, f))
    if os.path.exists(INDEX_DIR):
        for f in os.listdir(INDEX_DIR): os.remove(os.path.join(INDEX_DIR, f))
    if os.path.exists(META_FILE): os.remove(META_FILE)
    st.cache_resource.clear()
    st.rerun()

# ---------------- BUILD SYSTEM ----------------
@st.cache_resource
def build_system():
    metadata_store = {}
    all_docs = []
    if not os.path.exists(PDF_DIR) or not os.listdir(PDF_DIR): return None, None, {}, None, None, None

    for pdf in os.listdir(PDF_DIR):
        if not pdf.endswith(".pdf"): continue
        loader = PyPDFLoader(os.path.join(PDF_DIR, pdf))
        docs = loader.load()
        full_text = " ".join(d.page_content for d in docs)
        year = extract_year(full_text)
        references = extract_references(full_text)
        metadata_store[pdf] = {"paper_id": pdf, "year": year, "references": references}
        for d in docs: d.metadata.update({"paper": pdf, "year": year})
        all_docs.extend(docs)

    save_metadata(metadata_store)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    embeddings = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_DIR)

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
    
    deep_dive_prompt = PromptTemplate(
        template="""You are an expert research analyst. Extract information from the provided paper.
        Clean the text: remove headers, footers, page numbers, and complex LaTeX equations.
        
        Focus strictly on:
        - Title: The full formal title of the paper.
        - Abstract: High-level summary.
        - Introduction: Core motivation.
        - Methods: How it was done.
        - Experiments/Results: Data and findings.
        - Conclusion: Final takeaways.
        
        Context: {context}
        Format the output clearly with these headings using bold text.
        """,
        input_variables=["context"]
    )

    return vectorstore, metadata_store, llm, deep_dive_prompt

# ---------------- MAIN UI ----------------
st.markdown("<h1 class='main-title'>RESEARCH INTELLIGENCE SYSTEM</h1>", unsafe_allow_html=True)

if not os.listdir(PDF_DIR):
    st.warning("Welcome. Please upload research papers in the sidebar to begin analysis.")
    st.stop()

vectorstore, metadata_store, llm, deep_dive_prompt = build_system()

# Metrics
df = pd.DataFrame([{"Paper": k, "Year": v["year"], "References": len(v["references"])} for k, v in metadata_store.items()])
m1, m2, m3 = st.columns(3)
m1.metric("Library Size", f"{len(df)} Papers")
m2.metric("Latest Release", int(df['Year'].max()) if not df.empty and df['Year'].max() else "N/A")
m3.metric("Status", "Ready for Deep Dive")

st.markdown("---")

# ---------------- SELECTION UI ----------------
col_lib, col_trend = st.columns([1.2, 0.8])

with col_lib:
    st.subheader("📊 Paper Library")
    event = st.dataframe(
        df, 
        use_container_width=True, 
        hide_index=True, 
        on_select="rerun", 
        selection_mode="multi-row", # Enables Checkboxes
        column_config={
            "Paper": st.column_config.TextColumn("Paper Name", width="large"),
            "Year": st.column_config.NumberColumn("Year", format="%d"),
        }
    )

with col_trend:
    st.subheader("📈 Research Trends")
    year_counts = Counter(v["year"] for v in metadata_store.values() if v["year"])
    trend_df = pd.DataFrame(sorted(year_counts.items()), columns=["Year", "Count"])
    chart = alt.Chart(trend_df).mark_area(line={'color':'#E50914'}, color=alt.Gradient(
        gradient='linear', stops=[alt.GradientStop(color='#E50914', offset=0), alt.GradientStop(color='black', offset=1)],
        x1=1, x2=1, y1=1, y2=0)).encode(
            x=alt.X("Year:O", axis=alt.Axis(labelColor="white", titleColor="white")),
            y=alt.Y("Count:Q", axis=alt.Axis(labelColor="white", titleColor="white")),
        ).properties(height=250, background="#000000")
    st.altair_chart(chart, use_container_width=True)

# ---------------- LOGIC FOR SCOPED ANALYSIS ----------------
selected_indices = event.selection.rows

if selected_indices:
    selected_papers = df.iloc[selected_indices]["Paper"].tolist()
    # The last selected paper acts as the "Active/Radio" choice for Deep Dive
    active_paper = selected_papers[-1] 
    
    st.markdown(f"---")
    st.subheader(f"🎯 Active Analysis: {active_paper}")
    
    # 1. Automatic Deep Dive for Active Paper
    with st.status(f"Processing Deep Dive for {active_paper}...", expanded=True):
        loader = PyPDFLoader(os.path.join(PDF_DIR, active_paper))
        paper_docs = loader.load()
        paper_context = "\n\n".join([d.page_content for d in paper_docs])
        
        deep_chain = deep_dive_prompt | llm | StrOutputParser()
        analysis = deep_chain.invoke({"context": paper_context[:15000]}) 
        
        st.info(analysis)
        
        # 2. Dropdown for citations as requested
        with st.expander("🔗 Reference List (Citations)"):
            for r in metadata_store[active_paper]["references"]:
                st.write(f"• {r}")

    # 3. Global Assistant Scoped to Checkboxed Papers
    st.markdown("---")
    st.subheader(f"💬 Assistant Scope: {', '.join(selected_papers)}")
    query = st.text_input("Ask a question focusing ONLY on the selected papers...", placeholder="e.g. Compare the findings of these papers...")

    if st.button("RUN SCOPED ANALYSIS"):
        # MMR Retriever with metadata filter for selected papers
        retriever = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 6, "filter": {"paper": selected_papers}}
        )
        
        rag_chain = (
            RunnableParallel({
                "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
                "question": RunnablePassthrough()
            })
            | PromptTemplate.from_template("Using only the selected papers, answer: {question}\n\nContext:\n{context}") 
            | llm 
            | StrOutputParser()
        )
        
        answer = rag_chain.invoke(query)
        st.markdown("#### 🧠 AI Insights")
        st.success(answer)

else:
    st.info("💡 Select one or more papers from the table to begin analysis.")

st.markdown("---")