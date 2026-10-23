import os
import sys
import json
import shutil
import time
from pathlib import Path
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pdf_extractor import extract_pdf, save_extraction
from src.text_cleaner import clean_document, save_cleaned_document, create_text_chunks
from src.hybrid_pipeline import HybridPipeline
from src.embeddings import build_embeddings_from_chunks
from src.rag import TenderRAG

st.set_page_config(
    page_title="TenderIQ | RFP Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════
# PRODUCTION-GRADE CSS & THEME
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Global Container & Fonts */
    .main .block-container { 
        padding-top: 2rem; 
        max-width: 1400px; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

    /* Animated Hero Title */
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(270deg, #10b981, #3b82f6, #8b5cf6, #10b981);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 8s ease infinite;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    
    .hero-sub {
        color: #64748b;
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    /* Glassmorphic Metrics */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 16px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* Entity Tags with Hover States */
    .entity-tag {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        margin: 4px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        transition: all 0.2s ease;
        cursor: default;
    }
    .entity-tag:hover { filter: brightness(1.1); transform: scale(1.02); }
    
    .tag-cert { background: rgba(220, 38, 38, 0.1); color: #ef4444; border: 1px solid rgba(220, 38, 38, 0.3); }
    .tag-fin  { background: rgba(245, 158, 11, 0.1); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .tag-tech { background: rgba(59, 130, 246, 0.1); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
    .tag-date { background: rgba(139, 92, 246, 0.1); color: #8b5cf6; border: 1px solid rgba(139, 92, 246, 0.3); }
    .tag-org  { background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .tag-doc  { background: rgba(236, 72, 153, 0.1); color: #ec4899; border: 1px solid rgba(236, 72, 153, 0.3); }

    /* RAG Answer Box */
    .answer-box {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.05), rgba(59, 130, 246, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        line-height: 1.7;
        font-size: 1.05rem;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02);
    }
    .answer-cursor {
        display: inline-block;
        width: 8px;
        height: 18px;
        background-color: #10b981;
        animation: blink 1s step-end infinite;
        vertical-align: middle;
        margin-left: 4px;
    }
    @keyframes blink { 50% { opacity: 0; } }

    /* Evidence Cards */
    .evidence-card {
        background: var(--secondary-background-color);
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 16px;
        margin: 8px 0;
        font-size: 0.95rem;
        color: var(--text-color);
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: background 0.2s ease;
    }
    .evidence-card:hover { background: rgba(59, 130, 246, 0.03); }

    /* Smooth Gradient Divider */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.3), transparent);
        margin: 32px 0;
    }
    
    /* Button Overrides */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def normalize_doc_id(name: str) -> str:
    """ONE canonical id format used everywhere (fixes infinite upload loop)."""
    stem = Path(name).stem if name.lower().endswith(".pdf") else name
    stem = stem.replace("_cleaned", "").replace("_extracted", "")
    return stem.lower().replace(" ", "_").replace("%20", "_")


# ═══════════════════════════════════════════════════════
# CACHED LOADERS
# ═══════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⚡ Booting AI Search & RAG Engine...")
def get_cached_rag(_embed_version: int = 0):
    return TenderRAG()


@st.cache_resource(show_spinner="⚡ Loading Custom NER Model...")
def get_cached_hybrid():
    return HybridPipeline("models/tender_ner_v1")


@st.cache_data(show_spinner=False)
def get_hybrid_results(doc_path_str: str, mtime: float):
    pipeline = get_cached_hybrid()
    return pipeline.process_document(doc_path_str)


# ═══════════════════════════════════════════════════════
# PIPELINE OPERATIONS
# ═══════════════════════════════════════════════════════
def run_single_pipeline(pdf_path: Path) -> str:
    from src.embeddings import append_document_embeddings

    doc_id = normalize_doc_id(pdf_path.name)
    
    progress_bar = st.progress(0, text="Initializing extraction...")

    progress_bar.progress(20, text="📄 Extracting text & layouts...")
    result = extract_pdf(str(pdf_path), document_id=doc_id)
    extracted_path = save_extraction(result, "data/extracted")

    progress_bar.progress(50, text="🧹 Cleaning & normalizing text...")
    cleaned = clean_document(extracted_path)
    save_cleaned_document(cleaned, "data/extracted")

    progress_bar.progress(70, text="📦 Creating semantic chunks...")
    new_chunks = create_text_chunks(cleaned)

    all_chunks_path = Path("data/extracted/all_chunks.json")
    if all_chunks_path.exists():
        with open(all_chunks_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    else:
        all_chunks = []

    all_chunks = [c for c in all_chunks if c.get("document_id") != doc_id]
    all_chunks.extend(new_chunks)

    with open(all_chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    progress_bar.progress(90, text=f"⚡ Encoding {len(new_chunks)} chunks for semantic search...")
    append_document_embeddings(new_chunks)
    
    progress_bar.progress(100, text="✅ Processing complete!")
    time.sleep(0.5)
    progress_bar.empty()

    st.session_state["embed_version"] = st.session_state.get("embed_version", 0) + 1
    st.toast(f"🎉 '{pdf_path.name}' successfully indexed!", icon="✅")
    return doc_id


def delete_specific_document(doc_id: str):
    doc_id = normalize_doc_id(doc_id)

    for pdf in Path("data/raw_pdfs").glob("*.pdf"):
        if normalize_doc_id(pdf.name) == doc_id:
            pdf.unlink()

    for f in Path("data/extracted").glob("*.json"):
        if normalize_doc_id(f.stem.replace("_cleaned", "").replace("_extracted", "")) == doc_id:
            if f.name in ("all_chunks.json", "chunk_metadata.json"):
                continue
            try:
                f.unlink()
            except Exception:
                pass

    all_chunks_path = Path("data/extracted/all_chunks.json")
    if all_chunks_path.exists():
        with open(all_chunks_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        all_chunks = [c for c in all_chunks if c.get("document_id") != doc_id]
        with open(all_chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    try:
        from src.embeddings import EmbeddingEngine
        engine = EmbeddingEngine()
        vectors, metadata = engine.load_embeddings("data/extracted")
        if len(metadata) > 0:
            keep_indices = [i for i, m in enumerate(metadata) if m.get("document_id") != doc_id]
            if keep_indices:
                new_vectors = vectors[keep_indices]
                new_meta = [metadata[i] for i in keep_indices]
                engine.save_embeddings(new_vectors, new_meta, "data/extracted")
            else:
                for fname in ["embeddings.npy", "chunk_metadata.json", "all_chunks.json"]:
                    p = Path("data/extracted") / fname
                    if p.exists():
                        p.unlink()
    except Exception:
        pass

    st.session_state["embed_version"] = st.session_state.get("embed_version", 0) + 1
    get_hybrid_results.clear()
    st.toast("🗑️ Document deleted successfully.", icon="🛑")


def wipe_all_data():
    for d in ["data/raw_pdfs", "data/extracted"]:
        p = Path(d)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    st.cache_resource.clear()
    get_hybrid_results.clear()
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.toast("♻️ Workspace completely wiped.", icon="🧹")


def stream_text(text: str, speed: float = 0.005):
    placeholder = st.empty()
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(
            f'<div class="answer-box">{displayed}<span class="answer-cursor"></span></div>',
            unsafe_allow_html=True
        )
        time.sleep(speed)
    placeholder.markdown(
        f'<div class="answer-box">{displayed}</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════
def main():
    st.markdown('<div class="hero-title">TenderIQ</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">AI-Powered RFP Intelligence & Requirement Extraction System</p>', unsafe_allow_html=True)

    Path("data/raw_pdfs").mkdir(parents=True, exist_ok=True)
    Path("data/extracted").mkdir(parents=True, exist_ok=True)

    if "embed_version" not in st.session_state:
        st.session_state["embed_version"] = 0
    if "processed_upload_keys" not in st.session_state:
        st.session_state["processed_upload_keys"] = set()

    hybrid = get_cached_hybrid()
    rag = get_cached_rag(st.session_state["embed_version"])

    # ───────────────────────────────────────────────────
    # SIDEBAR
    # ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Engine Status")
        if getattr(rag, "provider", None):
            st.success(f"🟢 **LLM Active:** {rag.provider.upper()} \n\n*{getattr(rag, 'active_model_name', '')}*")
        else:
            st.warning("⚠️ **LLM Offline:** Search-only mode active")
            
        st.markdown("---")
        st.markdown("### 📄 Document Pipeline")

        uploaded_file = st.file_uploader("Upload New Tender (PDF)", type=["pdf"], key="pdf_uploader")

        if uploaded_file is not None:
            file_key = f"{uploaded_file.name}::{uploaded_file.size}"
            doc_id = normalize_doc_id(uploaded_file.name)

            already_done = file_key in st.session_state["processed_upload_keys"]
            already_current = st.session_state.get("processed_file_key") == file_key

            if not already_done and not already_current:
                pdf_path = Path("data/raw_pdfs") / uploaded_file.name
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                processed_id = run_single_pipeline(pdf_path)

                st.session_state["processed_upload_keys"].add(file_key)
                st.session_state["processed_file_key"] = file_key
                st.session_state["current_doc_id"] = processed_id
                st.session_state["selected_doc"] = processed_id
                st.rerun()
            else:
                st.session_state["selected_doc"] = doc_id
                st.session_state["current_doc_id"] = doc_id

        extracted_dir = Path("data/extracted")
        cleaned_files = sorted(list(extracted_dir.glob("*_cleaned.json")))

        if not cleaned_files:
            st.info("👋 Workspace is empty. Upload a PDF to begin.")
            return

        doc_options = {normalize_doc_id(f.stem.replace("_cleaned", "")): f for f in cleaned_files}
        doc_keys = list(doc_options.keys())

        default_index = 0
        sel = st.session_state.get("selected_doc")
        if sel and normalize_doc_id(sel) in doc_keys:
            default_index = doc_keys.index(normalize_doc_id(sel))

        selected_doc_key = st.selectbox("📂 Active Document", options=doc_keys, index=default_index)
        st.session_state["selected_doc"] = selected_doc_key
        selected_doc_path = doc_options[selected_doc_key]

        with open(selected_doc_path, "r", encoding="utf-8") as f:
            cleaned_doc = json.load(f)

        st.markdown("---")
        st.markdown("### 🛠️ Maintenance")
        if st.button("🗑️ Delete Active File", use_container_width=True):
            delete_specific_document(selected_doc_key)
            for k in ("selected_doc", "current_doc_id", "processed_file_key"):
                st.session_state.pop(k, None)
            st.rerun()

        if st.button("🚨 Wipe Database", type="primary", use_container_width=True):
            wipe_all_data()
            st.rerun()

    # ───────────────────────────────────────────────────
    # DATA PROCESSING & TABS
    # ───────────────────────────────────────────────────
    doc_results = None
    try:
        mtime = selected_doc_path.stat().st_mtime
        doc_results = get_hybrid_results(str(selected_doc_path), mtime)
    except Exception as e:
        st.error(f"Pipeline Execution Error: {e}")

    tab1, tab2, tab3 = st.tabs([
        "📊 Document Intelligence",
        "🏷️ NLP Entity Visualizer",
        "💬 Semantic RAG Chat"
    ])

    # ═══ TAB 1: OVERVIEW ═══
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Total Pages", cleaned_doc.get("total_pages", "—"))
        col2.metric("📝 Word Count", f"{cleaned_doc.get('total_words', 0):,}")
        
        if doc_results:
            stats = doc_results.get("statistics", {})
            col3.metric("🧩 Entities Detected", stats.get("total_ner_entities", 0))
            col4.metric("🔎 Key Rules Matched", stats.get("total_rule_fields", 0))

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📋 Structured Requirements Extraction")

        if doc_results:
            rules_data = doc_results.get("rule_extracted_with_pages", {})
            if rules_data:
                df_rules = []
                for field, val_data in rules_data.items():
                    if isinstance(val_data, dict):
                        value = val_data.get("value", val_data)
                        page = val_data.get("page", "—")
                    else:
                        value, page = val_data, "—"
                    df_rules.append({
                        "Extraction Field": field.replace("_", " ").title(),
                        "Identified Value": value,
                        "Source Page": f"Page {page}"
                    })
                
                st.dataframe(
                    pd.DataFrame(df_rules), 
                    width=1400, 
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No deterministic rules matched on this document.")

    # ═══ TAB 2: NER VISUALIZER ═══
    with tab2:
        st.markdown("### 🧠 Named Entity Recognition (NER)")
        st.markdown("Custom spaCy pipeline combined with domain-specific heuristics.")

        if doc_results:
            grouped_entities = doc_results.get("structured_data", {})

            tag_map = [
                ("📜 Certifications", "certifications", "tag-cert"),
                ("💰 Financial Terms", "financial_requirements", "tag-fin"),
                ("🛠️ Technical Specs", "technical_requirements", "tag-tech"),
                ("📅 Critical Dates", "deadlines", "tag-date"),
                ("🏢 Organizations", "eligibility", "tag-org"),
                ("📑 Required Docs", "documents_required", "tag-doc"),
            ]

            cols = st.columns(3)
            for i, (title, key, tag_cls) in enumerate(tag_map):
                with cols[i % 3]:
                    st.markdown(f"#### {title}")
                    items = grouped_entities.get(key, [])
                    if items:
                        for item in items[:8]:  # Limit display to top 8 to prevent UI clutter
                            st.markdown(f'<span class="entity-tag {tag_cls}">{item}</span>', unsafe_allow_html=True)
                    else:
                        st.caption("No entities detected.")
                    st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📖 In-Document Entity Highlighter")

            pages = cleaned_doc.get("pages", [])
            if pages:
                page_num = st.selectbox("Select Page to Inspect:", options=[p["page_number"] for p in pages])
                page_text = pages[page_num - 1]["text"]
                pred_ents = hybrid.ner.predict(page_text)

                if pred_ents:
                    st.caption(f"Found **{len(pred_ents)}** semantic entities on Page {page_num}")
                    highlighted = page_text
                    
                    # Highlight in reverse order to preserve string indices
                    for ent in sorted(pred_ents, key=lambda x: -x["start"]):
                        s, e, label, val = ent["start"], ent["end"], ent["label"], ent["text"]
                        tc = (
                            "tag-cert" if "CERT" in label else
                            "tag-fin" if "FIN" in label else
                            "tag-tech" if "TECH" in label else
                            "tag-date" if "DEADLINE" in label else
                            "tag-org" if "ORG" in label or "ELIG" in label else "tag-doc"
                        )
                        badge = f'<span class="entity-tag {tc}">{val} [{label}]</span>'
                        highlighted = highlighted[:s] + badge + highlighted[e:]
                        
                    st.markdown(
                        f'<div style="line-height:2.2; font-size:15px; padding:24px; '
                        f'background: var(--secondary-background-color); border-radius:12px; '
                        f'border:1px solid rgba(128,128,128,0.2); box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);">'
                        f'{highlighted.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("No highlightable entities detected on this page.")
                    st.markdown(f'<div style="padding:20px; background: var(--secondary-background-color); border-radius:8px;">{page_text[:2000]}...</div>', unsafe_allow_html=True)

    # ═══ TAB 3: RAG CHAT ═══
    with tab3:
        st.markdown("### 💬 Generative Q&A (RAG)")
        
        scope = st.radio(
            "🔍 Search Context:",
            options=["Active document only", "Global (All indexed tenders)"],
            index=0,
            horizontal=True
        )

        query = st.chat_input("Ask a question about the requirements, deadlines, or scope...")

        if query:
            st.markdown(f"**You:** {query}")
            filter_id = selected_doc_key if scope == "Active document only" else None

            with st.spinner("Analyzing context & formulating response..."):
                if getattr(rag, "provider", None) or getattr(rag, "llm_client", None):
                    response = rag.ask(query, top_k=8, filter_document_id=filter_id)
                else:
                    response = rag.ask_without_llm(query, top_k=5, filter_document_id=filter_id)
                    st.toast("LLM disabled. Returning semantic search snippets.", icon="⚠️")

            stream_text(response.get("answer", "I couldn't find a definitive answer in the document context."), speed=0.005)

            if response.get("evidence"):
                st.markdown(f"#### 📑 Verified Sources ({len(response['evidence'])})")
                for i, ev in enumerate(response["evidence"]):
                    with st.expander(f"Source {i+1}: {ev['document_id']} (Page {ev['page_number']}) - Match Score: {ev['score']:.2f}"):
                        st.markdown(f'<div class="evidence-card">"{ev["text"]}..."</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()