# 🚀 TenderIQ — AI-Powered Tender Intelligence & QA System

> An end-to-end NLP Capstone System featuring custom spaCy NER, Hybrid Rule Extraction, Page-Aware Semantic Search, and RAG Grounded Summarization.

📍 **Live Demo:** [Click Here to Launch TenderIQ App](https://tender-iq-mzwt9hmwzi2dxudf5auhh8.streamlit.app/)

---

## 🛠️ Architecture Overview
1. **PDF Parsing & Cleaning:** PyMuPDF layout and table extraction.
2. **Custom NER:** Fine-tuned spaCy model (`models/tender_ner_v1`) trained on domain-specific procurement entities (`FINANCIAL_REQUIREMENT`, `TECHNICAL_REQUIREMENT`, `DEADLINE`, etc.).
3. **Hybrid Engine:** Regex-rule fallback + NER for high recall on GeM & CPPP tenders.
4. **Vector Search:** `sentence-transformers/all-MiniLM-L6-v2` with incremental embeddings & document-level scope filtering.
5. **Grounded RAG:** Dynamic Gemini AI retrieval with direct page-level evidence citation.

---

## 💻 Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/tender-iq.git
cd tender-iq
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run streamlit_app.py