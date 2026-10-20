"""
TenderIQ — Fully Dynamic RAG Pipeline
======================================
100% Dynamic Model Discovery with Safe Token Probing.
"""

import sys
import os
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.logger_config import get_logger
from src.semantic_search import SemanticSearchEngine

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are TenderIQ, an expert tender document analyst.

Your job is to answer questions about tender/RFP documents accurately.

RULES:
1. Answer ONLY based on the provided context from the tender document.
2. If the answer is not in the context, say: "This information is not explicitly mentioned in the provided tender context."
3. Do NOT invent information.
4. Extract specific dates, organization names, quantities, and requirements from the context.
5. Keep answers direct, concise, and professional.
"""

CONTEXT_TEMPLATE = """Here is the relevant context extracted from the tender document:

{context}

Based ONLY on the above context, answer this question accurately:
{question}"""


def _get_api_key(key_name: str) -> str:
    """Helper to fetch API key from Streamlit Cloud Secrets OR local environment variables."""
    val = os.getenv(key_name, "").strip()
    if val:
        return val

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if key_name in st.secrets and st.secrets[key_name]:
                return str(st.secrets[key_name]).strip()
            for k in st.secrets:
                if k.upper() == key_name.upper() and st.secrets[k]:
                    return str(st.secrets[k]).strip()
    except Exception:
        pass

    return ""


class TenderRAG:
    """Universal RAG pipeline with safe dynamic model discovery."""

    def __init__(
        self,
        embeddings_dir="data/extracted",
        model_name=None,
    ):
        logger.info("Initializing RAG Pipeline...")
        self.search_engine = SemanticSearchEngine(embeddings_dir, model_name)
        
        self.gemini_key = _get_api_key("GEMINI_API_KEY")
        self.groq_key = _get_api_key("GROQ_API_KEY")

        if self.gemini_key:
            os.environ["GEMINI_API_KEY"] = self.gemini_key
        if self.groq_key:
            os.environ["GROQ_API_KEY"] = self.groq_key
        
        self.provider = None
        self.gemini_model = None
        self.active_model_name = None
        self.groq_client = None
        self.last_error = None

        # ── 1. SAFE GEMINI DISCOVERY ──
        if self.gemini_key:
            self._init_dynamic_gemini()
        else:
            self.last_error = "GEMINI_API_KEY is missing or empty in Secrets."

        # ── 2. GROQ FALLBACK ──
        if not self.provider and self.groq_key:
            self._init_dynamic_groq()

        if not self.provider:
            logger.warning(f"No active LLM provider found. Error: {self.last_error}")

        self.llm_client = self.provider

    def _init_dynamic_gemini(self):
        """Fast & safe model discovery."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            
            discovered = []
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    name = m.name
                    if any(x in name.lower() for x in ["tts", "image", "robot", "research", "lyria", "antigravity", "computer-use", "transcribe"]):
                        continue
                    discovered.append(name)
            
            logger.info(f"Found {len(discovered)} text models")

            if not discovered:
                self.last_error = "No generateContent models available on key."
                return

            prioritized = sorted(
                discovered,
                key=lambda n: (0 if "flash" in n.lower() else 1 if "pro" in n.lower() else 2)
            )

            for model_id in prioritized[:5]:
                try:
                    logger.info(f"Testing Gemini model: {model_id}...")
                    test_model = genai.GenerativeModel(model_id)
                    probe = test_model.generate_content(
                        "Hi",
                        generation_config={"max_output_tokens": 15, "temperature": 0.0}
                    )
                    
                    # Safe check for response text
                    has_parts = False
                    if probe and hasattr(probe, "candidates") and probe.candidates:
                        cand = probe.candidates[0]
                        if hasattr(cand, "content") and cand.content and cand.content.parts:
                            has_parts = True

                    if has_parts or (probe and hasattr(probe, "text")):
                        self.gemini_model = genai.GenerativeModel(
                            model_id, 
                            system_instruction=SYSTEM_PROMPT
                        )
                        self.active_model_name = model_id.replace("models/", "")
                        self.provider = "gemini"
                        self.last_error = None
                        logger.info(f"✓ Connected to Gemini: {self.active_model_name}")
                        return
                except Exception as e:
                    logger.debug(f"Skipped {model_id}: {e}")
                    self.last_error = f"Probe failed for {model_id}: {e}"
                    continue

        except Exception as e:
            self.last_error = f"Gemini init error: {e}"
            logger.warning(f"Gemini error: {e}")

    def _init_dynamic_groq(self):
        """Dynamically probes Groq models."""
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=self.groq_key)
            all_groq = [m.id for m in self.groq_client.models.list().data]
            
            chat_candidates = [
                m for m in all_groq 
                if not any(x in m.lower() for x in ["guard", "whisper", "vision", "orpheus"])
            ]

            for model_id in chat_candidates:
                try:
                    logger.info(f"Testing Groq model probe: {model_id}...")
                    self.groq_client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=5
                    )
                    self.active_model_name = model_id
                    self.provider = "groq"
                    self.last_error = None
                    logger.info(f"✓ Connected to live Groq model: {self.active_model_name}")
                    return
                except Exception as e:
                    self.last_error = f"Groq probe failed for {model_id}: {e}"
                    continue
        except Exception as e:
            self.last_error = f"Groq configuration error: {e}"
            logger.warning(f"Groq discovery error: {e}")

    def _build_context(self, results):
        if not results:
            return "No relevant context found in the tender documents."

        context_parts = []
        for r in results:
            context_parts.append(
                f"[Document: {r['document_id']}, Page {r['page_number']} | Relevance: {r['score']:.2f}]\n"
                f"{r['text']}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _call_llm(self, prompt):
        if self.provider == "gemini":
            try:
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.1, "max_output_tokens": 600}
                )
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini generation error: {e}")
                return f"[Gemini Error: {e}]"

        elif self.provider == "groq":
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.active_model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=600,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Groq generation error: {e}")
                return f"[Groq Error: {e}]"

        return f"[LLM not active. Status: {self.last_error}]"

    def ask(self, question, top_k=8, min_score=0.05, filter_document_id=None):
        if not question or not question.strip():
            return {
                "question": question,
                "answer": "Please enter a valid question.",
                "evidence": [],
                "sources": [],
                "found": False
            }

        logger.info(f"RAG query: '{question}' (doc_filter: {filter_document_id})")

        results = self.search_engine.search(
            query=question,
            top_k=top_k,
            min_score=min_score,
            filter_document_id=filter_document_id
        )

        if not results:
            return {
                "question": question,
                "answer": "This information was not found in the selected document.",
                "evidence": [],
                "sources": [],
                "found": False
            }

        context = self._build_context(results)
        prompt = CONTEXT_TEMPLATE.format(context=context, question=question)
        answer = self._call_llm(prompt)

        evidence = [
            {
                "document_id": r["document_id"],
                "page_number": r["page_number"],
                "text": r["text"][:250],
                "score": r["score"]
            }
            for r in results
        ]

        sources = list(set(
            f"{r['document_id']} (Page {r['page_number']})"
            for r in results
        ))

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence,
            "sources": sources,
            "found": True
        }

    def ask_without_llm(self, question, top_k=3, filter_document_id=None):
        results = self.search_engine.search(
            query=question, 
            top_k=top_k,
            filter_document_id=filter_document_id
        )

        if not results:
            return {
                "question": question,
                "answer": "No relevant context found in this document.",
                "evidence": [],
                "sources": [],
                "found": False
            }

        answer_parts = [
            f"• [Page {r['page_number']}]: {r['text']}"
            for r in results
        ]

        return {
            "question": question,
            "answer": "\n\n".join(answer_parts),
            "evidence": [
                {
                    "document_id": r["document_id"],
                    "page_number": r["page_number"],
                    "text": r["text"][:250],
                    "score": r["score"]
                }
                for r in results
            ],
            "sources": list(set(
                f"{r['document_id']} (Page {r['page_number']})"
                for r in results
            )),
            "found": True
        }