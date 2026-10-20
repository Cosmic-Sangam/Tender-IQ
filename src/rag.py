"""
TenderIQ — Zero-Probe RAG Engine with Rate-Limit Handling & Groq Fallback
========================================================================
Prevents 429 Quota Exceeded by eliminating dummy probe calls on startup.
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
    """Universal RAG pipeline with Zero-Probe Instant Boot."""

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

        # Always initialize Groq client if key is present for instant fallback
        if self.groq_key:
            self._init_groq()

        # Direct Gemini selection without dummy probe calls
        if self.gemini_key:
            self._init_zero_probe_gemini()

        if not self.provider and self.groq_client:
            self.provider = "groq"

        if not self.provider:
            logger.warning(f"No active LLM provider found. Error: {self.last_error}")

        self.llm_client = self.provider

    def _init_zero_probe_gemini(self):
        """Pick best model without making dummy API probe calls (saves quota!)."""
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
            
            if not discovered:
                self.last_error = "No generateContent models available on key."
                return

            # Prioritize stable flash models over preview ones
            prioritized = sorted(
                discovered,
                key=lambda n: (
                    0 if "flash-lite" in n.lower() else
                    1 if "1.5-flash" in n.lower() else
                    2 if "2.5-flash" in n.lower() else
                    3 if "flash" in n.lower() else 4
                )
            )

            # Instant assignment without calling generate_content()
            selected_model_id = prioritized[0]
            self.gemini_model = genai.GenerativeModel(
                selected_model_id,
                system_instruction=SYSTEM_PROMPT
            )
            self.active_model_name = selected_model_id.replace("models/", "")
            self.provider = "gemini"
            self.last_error = None
            logger.info(f"✓ Instantly assigned Gemini model: {self.active_model_name} (0 quota used on boot)")

        except Exception as e:
            self.last_error = f"Gemini init error: {e}"
            logger.warning(f"Gemini error: {e}")

    def _init_groq(self):
        """Initialize Groq client."""
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=self.groq_key)
            all_groq = [m.id for m in self.groq_client.models.list().data]
            chat_candidates = [
                m for m in all_groq 
                if not any(x in m.lower() for x in ["guard", "whisper", "vision", "orpheus"])
            ]
            if chat_candidates:
                self.groq_model_name = chat_candidates[0]
                logger.info(f"✓ Groq fallback ready: {self.groq_model_name}")
        except Exception as e:
            logger.warning(f"Groq setup notice: {e}")

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
        # 1. Try Gemini
        if self.provider == "gemini":
            try:
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.1, "max_output_tokens": 500}
                )
                return response.text.strip()
            except Exception as e:
                err_str = str(e)
                logger.error(f"Gemini error: {err_str}")
                
                # If Gemini hits 429 Rate Limit, attempt Groq fallback automatically
                if "429" in err_str or "quota" in err_str.lower():
                    if self.groq_client and hasattr(self, "groq_model_name"):
                        logger.info("⚡ Gemini 429 quota reached. Switching seamlessly to Groq fallback!")
                        return self._call_groq(prompt)
                    return f"⚠️ Gemini Free Tier Rate Limit (5 requests/min) reached. Please wait ~30 seconds and try again, or add a `GROQ_API_KEY` in Secrets for auto-fallback!"
                
                return f"[Gemini Generation Error: {e}]"

        # 2. Try Groq
        elif self.provider == "groq" or self.groq_client:
            return self._call_groq(prompt)

        return f"[LLM not active. Status: {self.last_error}]"

    def _call_groq(self, prompt):
        try:
            response = self.groq_client.chat.completions.create(
                model=getattr(self, "groq_model_name", "llama-3.3-70b-versatile"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            return f"[Groq Error: {e}]"

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