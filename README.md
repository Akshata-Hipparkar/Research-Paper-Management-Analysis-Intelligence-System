# 📚 Research Intelligence System  
### GenAI-Powered Research Paper Management & Analysis Platform

> A production-style **AI research assistant** that ingests academic papers and enables **semantic search, structured summarization, citation tracking, cross-paper comparison, and research trend analysis** using LLMs, RAG, FAISS, and Streamlit.

---

## 🚀 Project Overview

Modern researchers are overwhelmed by the explosive growth of scientific literature across domains such as **AI, healthcare, finance, physics, and social sciences**. Thousands of new papers are published every month, making it difficult to:

- Discover relevant research efficiently  
- Track citations and related work  
- Understand papers without reading them end-to-end  
- Identify emerging research trends across years  

The **Research Intelligence System** addresses these challenges by combining **Document AI, Retrieval-Augmented Generation (RAG), semantic search, and LLM-based analysis** into a single interactive platform.

This project closely mirrors real-world **academic search engines and research intelligence tools** used by universities, research labs, think tanks, and R&D teams.

---

## 🎯 Objectives

The system is designed to:

1. Ingest and manage research papers (PDFs + metadata)  
2. Automatically generate structured academic summaries  
3. Enable semantic search and question answering across papers  
4. Track references and citation information  
5. Analyze research trends over time  
6. Provide a clean, researcher-focused Streamlit UI  
7. Demonstrate end-to-end GenAI system design skills  

---

## 🧠 Core Features

### 📄 Research Paper Ingestion
- Upload multiple research PDFs  
- Automatic extraction of:
  - Full text
  - Publication year
  - Reference section
- Persistent metadata storage per paper  

---

### 🧩 Intelligent Document Processing
- Cleans raw PDF text by removing:
  - Headers, footers, page numbers
  - Formatting noise
- Splits papers into **semantic chunks**
- Preserves metadata such as:
  - Paper name
  - Year
  - Section context

---

### 🔎 Semantic Search & Retrieval (FAISS + RAG)
- Embeds document chunks using **HuggingFace sentence embeddings**
- Stores vectors in **FAISS**
- Enables:
  - Topic-based paper discovery
  - Context-aware question answering
  - Multi-paper retrieval using metadata filters

---

### 📝 Automatic Paper Deep-Dive
For any selected paper, the system generates a **structured academic analysis** covering:

- **Title**
- **Abstract**
- **Introduction / Motivation**
- **Methods**
- **Experiments & Results**
- **Conclusion**

All outputs are **LLM-generated but strictly grounded in the paper content**.

---

### 💬 Context-Aware Research Assistant
- Ask questions about:
  - A single paper
  - Multiple selected papers
- Example queries:
  - *“Compare the methodologies used in these papers”*
  - *“What datasets are used across these studies?”*
  - *“How do the findings differ?”*

Implemented using **Retrieval-Augmented Generation (RAG)** to minimize hallucinations.

---

### 🔗 Citation & Reference Tracking
- Automatically extracts reference sections
- Displays citation lists per paper
- Enables citation-aware research analysis

---

### 📈 Research Trend Analysis
- Aggregates papers by publication year
- Visualizes publication trends using **Altair**
- Helps identify:
  - Rapid growth areas
  - Emerging research directions

---

## 🖥️ Streamlit User Interface

The application provides:

- 📊 **Paper Library Dashboard**
  - Paper name
  - Publication year
  - Reference count
- ✅ Multi-paper selection
- 🎯 Active paper deep-dive view
- 💬 Interactive research chat assistant
- 📈 Research trend visualization
- 🧹 Library reset and re-indexing controls

Designed with a **dark, research-grade UI** optimized for long reading and analysis sessions.

---


