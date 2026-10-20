"""
TenderIQ — Fully Dynamic RAG Pipeline
======================================
100% Dynamic Model Discovery with Streamlit Cloud Secrets & .env support.
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
    # 1. Try Streamlit Secrets (Cloud Deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key_name in st.secrets:
            val = st.secrets[key_name]
            if val:
                return str(val).strip()
    except Exception:
        pass

    # 2. Try OS Environment / .env file (Local execution)
    return os.getenv(key_name, "").strip()


class TenderRAG:
    """Universal RAG pipeline with 100% dynamic model discovery."""

    def __init__(
        self,
        embeddings_dir="data/extracted",
        model_name=None,
    ):
        logger.info("Initializing RAG Pipeline...")
        self.search_engine = SemanticSearchEngine(embeddings_dir, model_name)
        
        # Dual key resolution (Cloud Secrets + Local .env)
        self.gemini_key = _get_api_key("GEMINI_API_KEY")
        self.groq_key = _get_api_key("GROQ_API_KEY")
        
        self.provider = None
        self.gemini_model = None
        self.active_model_name = None
        self.groq_client = None

        # ── 1. 100% DYNAMIC GEMINI DISCOVERY & PROBING ──
        if self.gemini_key:
            self._init_dynamic_gemini()

        # ── 2. DYNAMIC GROQ FALLBACK IF GEMINI IS NOT ACTIVE ──
        if not self.provider and self.groq_key:
            self._init_dynamic_groq()

        if not self.provider:
            logger.warning("  No active LLM provider found. Running in retrieval-only mode.")

        self.llm_client = self.provider

    def _init_dynamic_gemini(self):
        """Fast model discovery — picks first working flash model."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            
            # Get all models that support text generation
            discovered = []
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    name = m.name
                    # Skip TTS, image, robotics, research models
                    if any(x in name.lower() for x in ["tts", "image", "robot", "research", "lyria", "antigravity", "computer-use", "transcribe"]):
                        continue
                    discovered.append(name)
            
            logger.info(f"  Found {len(discovered)} text models")

            # Sort: flash first, then pro
            prioritized = sorted(
                discovered,
                key=lambda n: (0 if "flash" in n.lower() else 1 if "pro" in n.lower() else 2)
            )

            # Probe ONLY the first 3 candidates (not all)
            for model_id in prioritized[:3]:
                try:
                    logger.info(f"  Probing: {model_id}...")
                    candidate = genai.GenerativeModel(
                        model_id, 
                        system_instruction=SYSTEM_PROMPT
                    )
                    probe = candidate.generate_content(
                        "Say OK",
                        generation_config={"max_output_tokens": 3, "temperature": 0.0}
                    )
                    if probe and probe.text:
                        self.gemini_model = candidate
                        self.active_model_name = model_id.replace("models/", "")
                        self.provider = "gemini"
                        logger.info(f"  ✓ Active: {self.active_model_name}")
                        return
                except Exception as e:
                    logger.debug(f"  Skipped {model_id}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"  Gemini error: {e}")

    def _init_dynamic_groq(self):
        """Dynamically probes Groq models."""
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=self.groq_key)
            all_groq = [m.id for m in self.groq_client.models.list().data]
            
            # Filter for text chat models
            chat_candidates = [
                m for m in all_groq 
                if not any(x in m.lower() for x in ["guard", "whisper", "vision", "orpheus"])
            ]

            for model_id in chat_candidates:
                try:
                    logger.info(f"  Testing Groq model probe: {model_id}...")
                    self.groq_client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=5
                    )
                    self.active_model_name = model_id
                    self.provider = "groq"
                    logger.info(f"  ✓ Connected to live Groq model: {self.active_model_name}")
                    return
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"  Groq discovery error: {e}")

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
        # 1. Dynamic Gemini Call
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

        # 2. Dynamic Groq Call
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

        return "[LLM not configured. Please check your API keys in Streamlit Secrets or .env]"

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