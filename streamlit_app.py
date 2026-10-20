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
    page_title="TenderIQ - RFP Intelligence",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════
# PROFESSIONAL CSS
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 1400px; }

    /* Animated gradient title */
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #10b981, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 4px;
    }

    /* Metric cards */
    .stMetric { 
        background: var(--secondary-background-color); 
        padding: 12px; 
        border-radius: 10px; 
        border-left: 4px solid #10b981; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Entity pills */
    .entity-tag {
        display: inline-block;
        padding: 5px 12px;
        margin: 3px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }
    .tag-cert { background: #dc262622; color: #f87171; border: 1px solid #dc262655; }
    .tag-fin  { background: #f59e0b22; color: #fbbf24; border: 1px solid #f59e0b55; }
    .tag-tech { background: #3b82f622; color: #60a5fa; border: 1px solid #3b82f655; }
    .tag-date { background: #8b5cf622; color: #a78bfa; border: 1px solid #8b5cf655; }
    .tag-org  { background: #10b98122; color: #34d399; border: 1px solid #10b98155; }
    .tag-doc  { background: #ec489922; color: #f472b6; border: 1px solid #ec489955; }
    .tag-pen  { background: #f9731622; color: #fb923c; border: 1px solid #f9731655; }
    .tag-elig { background: #06b6d422; color: #22d3ee; border: 1px solid #06b6d455; }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #064e3b22, #065f4622);
        border: 1px solid #10b98144;
        border-radius: 12px;
        padding: 18px;
        margin: 10px 0;
        line-height: 1.8;
    }

    /* Evidence card */
    .evidence-card {
        background: var(--secondary-background-color);
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
    }

    /* Divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #10b981, transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# LAZY CACHED LOADERS (only load when needed)
# ═══════════════════════════════════════════════════════
# 1. Cache the Heavy RAG Engine & Embedding Model (Loads ONCE when app boots)
@st.cache_resource(show_spinner="⚡ Booting AI Search & RAG Engine...")
def get_cached_rag():
    from src.rag import TenderRAG
    return TenderRAG()


# 2. Cache the Hybrid NER Pipeline (Loads ONCE when app boots)
@st.cache_resource(show_spinner="⚡ Loading Custom NER Model...")
def get_cached_hybrid():
    from src.hybrid_pipeline import HybridPipeline
    return HybridPipeline("models/tender_ner_v1")


# Initialize cached instances
rag = get_cached_rag()
hybrid = get_cached_hybrid()


# ═══════════════════════════════════════════════════════
# PIPELINE OPERATIONS
# ═══════════════════════════════════════════════════════
def run_single_pipeline(pdf_path: Path) -> str:
    """Process a new PDF with INCREMENTAL embedding (fast)."""
    from src.embeddings import append_document_embeddings

    doc_id = pdf_path.stem.lower().replace(" ", "_")

    # 1. Extract
    with st.spinner("📄 Extracting text..."):
        result = extract_pdf(str(pdf_path), document_id=doc_id)
        extracted_path = save_extraction(result, "data/extracted")

    # 2. Clean
    with st.spinner("🧹 Cleaning..."):
        cleaned = clean_document(extracted_path)
        save_cleaned_document(cleaned, "data/extracted")

    # 3. Create chunks for THIS document only
    with st.spinner("📦 Creating chunks..."):
        new_chunks = create_text_chunks(cleaned)

        # Also update all_chunks.json
        all_chunks_path = Path("data/extracted/all_chunks.json")
        if all_chunks_path.exists():
            with open(all_chunks_path, "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
        else:
            all_chunks = []

        # Remove any old chunks for this doc (if re-uploading)
        all_chunks = [c for c in all_chunks if c["document_id"] != doc_id]
        all_chunks.extend(new_chunks)

        with open(all_chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # 4. INCREMENTAL embedding — only encode the new chunks
    with st.spinner(f"⚡ Encoding {len(new_chunks)} new chunks (fast)..."):
        append_document_embeddings(new_chunks)

    # Clear RAG cache so it reloads the new embeddings
    st.cache_resource.clear()
    st.success(f"🎉 '{pdf_path.name}' indexed in seconds!")
    return doc_id


def delete_specific_document(doc_id: str):
    """Delete a document and its embeddings without full rebuild."""
    import numpy as np
    from src.embeddings import EmbeddingEngine

    # Delete PDF
    for pdf in Path("data/raw_pdfs").glob("*.pdf"):
        if pdf.stem.lower().replace(" ", "_") == doc_id:
            pdf.unlink()

    # Delete extracted/cleaned JSON
    for f in Path("data/extracted").glob(f"{doc_id}_*.json"):
        f.unlink()

    # Remove from all_chunks.json
    all_chunks_path = Path("data/extracted/all_chunks.json")
    if all_chunks_path.exists():
        with open(all_chunks_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        all_chunks = [c for c in all_chunks if c["document_id"] != doc_id]
        with open(all_chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Remove from embeddings without full rebuild
    engine = EmbeddingEngine()
    vectors, metadata = engine.load_embeddings("data/extracted")
    if len(metadata) > 0:
        keep_indices = [i for i, m in enumerate(metadata) if m["document_id"] != doc_id]
        if keep_indices:
            new_vectors = vectors[keep_indices]
            new_meta = [metadata[i] for i in keep_indices]
            engine.save_embeddings(new_vectors, new_meta, "data/extracted")
        else:
            # No documents left
            for f in ["embeddings.npy", "chunk_metadata.json", "all_chunks.json"]:
                p = Path("data/extracted") / f
                if p.exists():
                    p.unlink()

    st.cache_resource.clear()


def wipe_all_data():
    for d in ["data/raw_pdfs", "data/extracted"]:
        p = Path(d)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    st.cache_resource.clear()
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def stream_text(text: str, speed: float = 0.012):
    """Typing animation for RAG answers."""
    placeholder = st.empty()
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(
            f'<div class="answer-box">{displayed}▌</div>',
            unsafe_allow_html=True
        )
        time.sleep(speed)
    placeholder.markdown(
        f'<div class="answer-box">{displayed}</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════
def main():
    # Hero section
    st.markdown('<p class="hero-title">📋 TenderIQ</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">RFP Intelligence & Requirement Extraction System • NLP Capstone</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    Path("data/raw_pdfs").mkdir(parents=True, exist_ok=True)
    Path("data/extracted").mkdir(parents=True, exist_ok=True)

    # ───────────────────────────────────────────────────
    # SIDEBAR
    # ───────────────────────────────────────────────────
    st.sidebar.markdown("## 📁 Documents")

    uploaded_file = st.sidebar.file_uploader("Upload Tender PDF", type=["pdf"])
    
    # Handle PDF Upload / Extraction safely inside session state
    if uploaded_file is not None:
        doc_id = os.path.splitext(uploaded_file.name)[0]
        
        # Check if this doc is ALREADY processed in memory
        if st.session_state.get("current_doc_id") != doc_id:
            with st.spinner("📄 Processing PDF text & extracting entities..."):
                # Save uploaded PDF
                pdf_path = os.path.join("data/raw_pdfs", uploaded_file.name)
                os.makedirs("data/raw_pdfs", exist_ok=True)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process extraction ONLY ONCE per document
                processed_id = run_single_pipeline(Path(pdf_path))
                
                # Store result in session_state so reruns don't repeat this step
                st.session_state["current_doc_id"] = processed_id
                st.session_state["selected_doc"] = processed_id
                st.session_state["last_uploaded"] = uploaded_file.name
                
                st.success(f"Successfully processed {uploaded_file.name}!")
                st.rerun()
        else:
            if st.session_state.get("last_uploaded") != uploaded_file.name:
                st.session_state["selected_doc"] = doc_id.lower().replace(" ", "_")
                st.session_state["last_uploaded"] = uploaded_file.name

    extracted_dir = Path("data/extracted")
    cleaned_files = sorted(list(extracted_dir.glob("*_cleaned.json")))

    if not cleaned_files:
        st.info("👋 Welcome! Upload a tender PDF from the sidebar to begin analysis.")
        return

    doc_options = {f.stem.replace("_cleaned", ""): f for f in cleaned_files}
    doc_keys = list(doc_options.keys())

    default_index = 0
    if "selected_doc" in st.session_state and st.session_state["selected_doc"] in doc_keys:
        default_index = doc_keys.index(st.session_state["selected_doc"])

    selected_doc_key = st.sidebar.selectbox(
        "📄 Active Document", options=doc_keys, index=default_index
    )
    st.session_state["selected_doc"] = selected_doc_key
    selected_doc_path = doc_options[selected_doc_key]

    with open(selected_doc_path, "r", encoding="utf-8") as f:
        cleaned_doc = json.load(f)

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Delete Active Document", use_container_width=True):
        with st.spinner("Deleting..."):
            delete_specific_document(selected_doc_key)
            if "selected_doc" in st.session_state:
                del st.session_state["selected_doc"]
            if "current_doc_id" in st.session_state:
                del st.session_state["current_doc_id"]
        st.rerun()

    st.sidebar.subheader("🧹 Maintenance")
    if st.sidebar.button("♻️ Wipe All Data", type="secondary", use_container_width=True):
        with st.spinner("Wiping..."):
            wipe_all_data()
        st.rerun()

    # ───────────────────────────────────────────────────
    # TABS
    # ───────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📊 Document Overview",
        "🏷️ Custom NER Visualizer",
        "💬 Semantic Q&A (RAG)"
    ])

    # Lazy load: pipeline only when Tab 1 or 2 is active
    pipeline = hybrid
    doc_results = None
    try:
        doc_results = pipeline.process_document(str(selected_doc_path))
    except Exception as e:
        st.error(f"Pipeline Error: {e}")

    # ═══════════════════════════════════════════════
    # TAB 1: METADATA & RULES
    # ═══════════════════════════════════════════════
    with tab1:
        st.header(f"📊 {cleaned_doc['document_id']}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Total Pages", cleaned_doc['total_pages'])
        col2.metric("📝 Word Count", f"{cleaned_doc['total_words']:,}")
        if doc_results:
            col3.metric("🧩 Entities Found", doc_results['statistics']['total_ner_entities'])
            col4.metric("🔎 Rules Matched", doc_results['statistics']['total_rule_fields'])

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("🔎 Extracted Structured Fields")
        
        if doc_results:
            rules_data = doc_results.get("rule_extracted_with_pages", {})
            if rules_data:
                df_rules = []
                for field, val_data in rules_data.items():
                    df_rules.append({
                        "📌 Field": field.replace("_", " ").title(),
                        "✅ Extracted Value": val_data["value"],
                        "📄 Source Page": f"Page {val_data['page']}"
                    })
                st.dataframe(pd.DataFrame(df_rules), use_container_width=True, hide_index=True)
            else:
                st.info("No deterministic rules matched. Try the NER or Q&A tabs.")

    # ═══════════════════════════════════════════════
    # TAB 2: NER VISUALIZER
    # ═══════════════════════════════════════════════
    with tab2:
        st.header("🏷️ Named Entity Recognition")
        st.write("Entities extracted using the hybrid spaCy + domain rules.")

        if doc_results:
            grouped_entities = doc_results.get("structured_data", {})
            
            tag_map = [
                ("📜 Certifications", "certifications", "tag-cert"),
                ("💰 Financial Terms", "financial_requirements", "tag-fin"),
                ("🛠️ Technical Specs", "technical_requirements", "tag-tech"),
                ("📅 Deadlines", "deadlines", "tag-date"),
                ("🏢 Organizations", "eligibility", "tag-org"),
                ("📑 Documents", "documents_required", "tag-doc"),
            ]
            
            cols = st.columns(3)
            for i, (title, key, tag_cls) in enumerate(tag_map):
                with cols[i % 3]:
                    st.markdown(f"##### {title}")
                    items = grouped_entities.get(key, [])
                    if items:
                        for item in items[:5]:
                            st.markdown(
                                f'<span class="entity-tag {tag_cls}">{item}</span>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.caption("None detected")

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("📖 Page-by-Page Entity Highlighter")
            
            page_num = st.selectbox(
                "Select Page:",
                options=[p["page_number"] for p in cleaned_doc["pages"]]
            )
            page_text = cleaned_doc["pages"][page_num - 1]["text"]
            pred_ents = pipeline.ner.predict(page_text)
            
            if pred_ents:
                st.caption(f"✨ {len(pred_ents)} entities on Page {page_num}")
                highlighted = page_text
                for ent in sorted(pred_ents, key=lambda x: -x["start"]):
                    s, e, label, val = ent["start"], ent["end"], ent["label"], ent["text"]
                    tc = (
                        "tag-cert" if "CERT" in label else
                        "tag-fin" if "FIN" in label else
                        "tag-tech" if "TECH" in label else
                        "tag-date" if "DEADLINE" in label else
                        "tag-org" if "ORG" in label else "tag-doc"
                    )
                    badge = f'<span class="entity-tag {tc}">{val} [{label}]</span>'
                    highlighted = highlighted[:s] + badge + highlighted[e:]
                st.markdown(
                    f'<div style="line-height:2.0; font-size:14px; padding:18px; '
                    f'background: var(--secondary-background-color); color: var(--text-color); '
                    f'border-radius:8px; border:1px solid rgba(128,128,128,0.2);">'
                    f'{highlighted.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.caption("No entities on this page.")
                st.text(page_text[:2000])

    # ═══════════════════════════════════════════════
    # TAB 3: RAG Q&A (LAZY LOADS LLM)
    # ═══════════════════════════════════════════════
    with tab3:
        st.header("💬 Semantic Search & RAG")
        st.markdown('<p style="color:#94a3b8;">Ask natural-language questions. Answers include document citations.</p>', unsafe_allow_html=True)

        scope = st.radio(
            "🎯 Search Scope:",
            options=["Active document only", "Search across all uploaded tenders"],
            index=0,
            horizontal=True
        )

        query = st.text_input(
            "💭 Ask a question:",
            placeholder="e.g., What is the submission deadline? Who issued this tender?"
        )

        if query:
            filter_id = selected_doc_key if scope == "Active document only" else None
            
            with st.spinner("🔍 Searching & generating..."):
                if rag.llm_client:
                    response = rag.ask(query, top_k=8, filter_document_id=filter_id)
                else:
                    response = rag.ask_without_llm(query, top_k=5, filter_document_id=filter_id)
                    st.warning("⚠️ LLM not active. Showing raw search results.")

            # Animated answer
            st.subheader("🤖 Answer")
            stream_text(response["answer"], speed=0.012)

            # Evidence
            if response.get("evidence"):
                st.subheader(f"📑 Evidence & Citations ({len(response['evidence'])} sources)")
                for ev in response["evidence"]:
                    with st.expander(
                        f"📖 {ev['document_id']} — Page {ev['page_number']} (Relevance: {ev['score']:.3f})"
                    ):
                        st.markdown(
                            f'<div class="evidence-card">"{ev["text"]}..."</div>',
                            unsafe_allow_html=True
                        )


if __name__ == "__main__":
    main()