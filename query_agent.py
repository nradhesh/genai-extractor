# query_agent.py
import os
import sys
import re
from pathlib import Path

try:
    from ddgs import DDGS  # new package name
except Exception:  # fallback if not installed yet
    from duckduckgo_search import DDGS

# ✅ Correct imports for LangChain v1.0.3 modular setup
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# ==============================================================
#  CONFIG
# ==============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY environment variable.")

DB_DIR = "vector_db_v2"
SOURCE_DIR = "extractor-2"

# ==============================================================
#  LOAD VECTOR DB (Lazy Loading)
# ==============================================================
_embedding_model_cache = None
_vectordb_cache = None


class SentenceTransformerEmbeddings(Embeddings):
    """Thin wrapper around `sentence-transformers` to avoid TensorFlow dependencies."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts):
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self._model.encode(text, convert_to_numpy=True).tolist()

def get_embedding_model():
    """Lazy load embedding model with retry logic for PyTorch meta tensor issues."""
    global _embedding_model_cache
    if _embedding_model_cache is not None:
        return _embedding_model_cache
    import shutil

    model_name = "sentence-transformers/all-mpnet-base-v2"
    
    # Try loading normally first
    try:
        _embedding_model_cache = SentenceTransformerEmbeddings(
            model_name=model_name,
            device="cpu",
        )
        return _embedding_model_cache
    except (NotImplementedError, RuntimeError) as e:
        error_str = str(e).lower()
        if "meta tensor" in error_str or "to_empty" in error_str:
            # Clear potentially corrupted cache and retry
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_cache = cache_dir / f"models--{model_name.replace('/', '--')}"
            if model_cache.exists():
                try:
                    shutil.rmtree(model_cache)
                    print("Warning: Cleared corrupted model cache. Re-downloading model...")
                except Exception as cleanup_err:
                    print(f"Warning: Could not clear cache: {cleanup_err}")
            
            # Retry after cache clear with explicit device
            _embedding_model_cache = SentenceTransformerEmbeddings(
                model_name=model_name,
                device="cpu",
            )
            return _embedding_model_cache
        raise

def get_vectordb():
    """Lazy load vector database."""
    global _vectordb_cache
    if _vectordb_cache is not None:
        return _vectordb_cache
    
    _vectordb_cache = Chroma(
        persist_directory=DB_DIR,
        embedding_function=get_embedding_model()
    )
    return _vectordb_cache

# Note: Use get_embedding_model() and get_vectordb() functions instead of direct access

# ==============================================================
#  LLM (Groq via OpenAI-compatible API)
# ==============================================================
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    temperature=0
)

# ==============================================================
#  RAG PROMPT
# ==============================================================
prompt_template = """
You are a scholarly research assistant for Dr. R. C. A. Naidu.
Answer the question based only on the given context.
If the context is insufficient, reply with "not sure".

Context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ==============================================================
#  FACULTY LOOKUP
# ==============================================================
def _slug(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())

def _normalise_faculty_name(raw_name: str) -> str:
    name = (raw_name or "").replace("google_scholar_", "")
    name = name.replace("_", " ").replace("  ", " ").strip()
    if name.lower().startswith("dr "):
        name = name.replace("Dr ", "Dr. ", 1)
    return name.strip()

def _build_faculty_lookup():
    pretty_to_raw = {}
    slug_to_pretty = {}
    base = Path(SOURCE_DIR)
    if base.exists():
        for csv_path in base.glob("*.csv"):
            raw = csv_path.stem
            pretty = _normalise_faculty_name(raw)
            pretty_to_raw[pretty] = raw
            slug_to_pretty[_slug(pretty)] = pretty
            slug_to_pretty[_slug(raw)] = pretty
    return pretty_to_raw, slug_to_pretty

FACULTY_LOOKUP, FACULTY_SLUG = _build_faculty_lookup()

def _match_faculties(question: str):
    slug_question = _slug(question)
    matches = []
    for slug_val, pretty in FACULTY_SLUG.items():
        if slug_val and slug_val in slug_question and pretty not in matches:
            matches.append(pretty)
    return matches

QUESTION_STOPS = {
    "what", "which", "when", "where", "who", "whose", "whom", "why", "how",
    "summarize", "list", "identify", "show", "provide", "give", "explain",
    "compare", "analyze", "report", "detail", "outline", "describe",
    "recent", "cloud", "data", "security", "best", "practices", "supports",
    "evidence", "work", "on",
}

def _extract_name_hints(text: str):
    cleaned = text.replace("?", " ").replace(",", " ")
    tokens = cleaned.split()
    hints = []
    current = []
    for token in tokens:
        word = token.strip()
        if not word:
            continue
        normalized = word.strip(".")
        lower = normalized.lower()
        if lower in QUESTION_STOPS and not current:
            continue
        if any(ch.isalpha() for ch in normalized) and normalized[0].isupper():
            current.append(word)
        else:
            if len(current) >= 2:
                hints.append(" ".join(current))
            current = []
    if len(current) >= 2:
        hints.append(" ".join(current))
    cleaned_hints = []
    for hint in hints:
        hint = re.sub(r"\s+", " ", hint).strip()
        if hint and hint not in cleaned_hints:
            cleaned_hints.append(hint)
    return cleaned_hints

def retrieve_documents(question: str, k: int = 6):
    matches = _match_faculties(question)
    name_hints = _extract_name_hints(question)
    docs = []
    seen_ids = set()
    if matches:
        for pretty in matches:
            raw = FACULTY_LOOKUP.get(pretty, "")
            filter_kwargs = {"faculty_pretty": pretty}
            faculty_docs = []
            filtered_docs = get_vectordb().similarity_search(question, k=k, filter=filter_kwargs)
            for doc in filtered_docs:
                rec_id = doc.metadata.get("citation_id") or doc.metadata.get("title") or doc.page_content
                if rec_id not in seen_ids:
                    seen_ids.add(rec_id)
                    faculty_docs.append(doc)
            raw_dump = get_vectordb().get(where=filter_kwargs, limit=500)
            for doc_text, meta in zip(raw_dump.get("documents", []), raw_dump.get("metadatas", [])):
                rec_id = meta.get("citation_id") or meta.get("title") or doc_text
                if rec_id in seen_ids:
                    continue
                seen_ids.add(rec_id)
                faculty_docs.append(Document(page_content=doc_text, metadata=meta))
            def sort_key(document: Document):
                raw_year = document.metadata.get("year") if isinstance(document.metadata, dict) else None
                try:
                    year_val = int(str(raw_year)[:4])
                except Exception:
                    year_val = -1
                cited = document.metadata.get("cited_by.value") or document.metadata.get("citations") or 0
                try:
                    cited_val = int(float(cited))
                except Exception:
                    cited_val = -1
                return (year_val, cited_val)
            faculty_docs.sort(key=sort_key, reverse=True)
            docs.extend(faculty_docs[:k])
    if not docs:
        docs = get_vectordb().similarity_search(question, k=k)
    return docs, matches, name_hints

def rag_answer(question: str, k: int = 6):
    docs, matches, name_hints = retrieve_documents(question, k=k)
    context = format_docs(docs)
    prompt_value = PROMPT.format_prompt(context=context, question=question)
    ai_message = llm.invoke(prompt_value.to_messages())
    response_text = getattr(ai_message, "content", ai_message)
    return response_text, docs, context, matches, name_hints

# ==============================================================
#  DUCKDUCKGO FALLBACK
# ==============================================================
def _contains_hint(record, hints):
    text = " ".join([
        record.get("title", ""),
        record.get("body", ""),
        record.get("href", ""),
    ]).lower()
    return any(h.lower() in text for h in hints)

def web_search(query, num_results=5, hints=None):
    hints = hints or []
    # Primary attempt
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(
                query,
                max_results=num_results,
                safesearch="moderate",
                region="wt-wt",
                timelimit="y"
            )]
    except Exception:
        results = []

    # Retry with alternative phrasing if empty or missing hints
    need_retry = not results
    if hints and results:
        need_retry = not any(_contains_hint(r, hints) for r in results)
    if need_retry:
        alt_queries = [
            f"{query} citations",
            "R C A Naidu cloud citations",
            "R C A Naidu cloud database citations",
            "site:scholar.google.com R C A Naidu cloud"
        ]
        for hint in hints:
            alt_queries.extend([
                f"\"{hint}\" cloud security 2024",
                f"\"{hint}\" cloud data security best practices",
                f"{hint} cloud security research 2023",
            ])
        for alt in alt_queries:
            try:
                with DDGS() as ddgs:
                    alt_results = [r for r in ddgs.text(
                        alt,
                        max_results=num_results,
                        safesearch="moderate",
                        region="wt-wt"
                    )]
                if alt_results:
                    results = alt_results
                    break
            except Exception:
                continue
    return results

# ==============================================================
#  HYBRID AGENT LOGIC
# ==============================================================
def _is_weak_answer(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    weak_markers = [
        "not sure",
        "insufficient",
        "don't have",
        "cannot find",
        "n/a",
    ]
    if any(m in lowered for m in weak_markers):
        return True
    return len(text.strip()) < 80

def _build_vector_references(docs):
    vector_lines = []
    records = []
    for idx, doc in enumerate(docs[:8], start=1):
        meta = doc.metadata or {}
        title = meta.get("title") or (doc.page_content.splitlines()[0][:120] if doc.page_content else "Untitled")
        authors = meta.get("authors", "").strip()
        year = meta.get("year", "").strip()
        citations = meta.get("citations")
        citation_text = f"{citations} citations" if citations not in (None, "", "None") else ""
        link = meta.get("link", "").strip()
        pieces = [title]
        if year:
            pieces.append(f"({year})")
        if authors:
            pieces.append(f"Authors: {authors}")
        if citation_text:
            pieces.append(citation_text)
        if link:
            pieces.append(f"URL: {link}")
        line = f"[V{idx}] " + " — ".join(pieces)
        vector_lines.append(line)
        records.append({"label": f"V{idx}", "title": title, "authors": authors, "year": year, "citations": citations, "link": link, "raw": doc.page_content})
    return "\n".join(vector_lines) if vector_lines else "No vector-derived documents found.", records

def _build_web_references(results):
    web_lines = []
    records = []
    for idx, result in enumerate((results or [])[:8], start=1):
        title = result.get("title", "Untitled")
        body = result.get("body", "")
        href = result.get("href", "")
        summary = body[:220] + ("..." if len(body) > 220 else "")
        pieces = [title]
        if summary:
            pieces.append(summary)
        if href:
            pieces.append(f"URL: {href}")
        line = f"[W{idx}] " + " — ".join(pieces)
        web_lines.append(line)
        records.append({"label": f"W{idx}", "title": title, "summary": summary, "url": href})
    return "\n".join(web_lines) if web_lines else "No web evidence found.", records

def agentic_answer(
    query: str,
    *,
    use_web: bool = True,
    web_k: int = 8,
    vector_k: int = 6,
) -> dict:
    _, docs, context, matches, name_hints = rag_answer(query, k=vector_k)
    all_hints = matches + [hint for hint in name_hints if hint not in matches]

    web_results = []
    if use_web and web_k > 0:
        web_results = web_search(query, num_results=web_k, hints=all_hints)

    # Build a combined, formatting-focused prompt
    # Domain filtering to scholarly sources
    allowed_domains = [
        "scholar.google.com", "researchgate.net", "semanticscholar.org",
        "ieeexplore.ieee.org", "springer.com", "link.springer.com",
    "sciencedirect.com", "acm.org", "dl.acm.org", "arxiv.org",
    "taylorfrancis.com", "igi-global.com", "scientificprogramming.com"
    ]
    def is_allowed(url: str) -> bool:
        return any(d in (url or "") for d in allowed_domains)

    filtered = [r for r in web_results if is_allowed(r.get("href", ""))] if use_web else []
    chosen_results = filtered

    if use_web and not filtered:
        # Retry with targeted site queries
        site_queries = [
            f'"R C A Naidu" cloud site:scholar.google.com',
            f'"R C A Naidu" cloud site:researchgate.net',
            f'"R C A Naidu" cloud site:semanticscholar.org',
        ]
        for sq in site_queries:
            more = web_search(sq, num_results=10)
            filtered = [r for r in more if is_allowed(r.get("href", ""))]
            if filtered:
                chosen_results = filtered
                break

    if use_web and all_hints and chosen_results:
        hinted = [r for r in chosen_results if _contains_hint(r, all_hints)]
        if not hinted:
            hinted = [r for r in chosen_results if _contains_hint(r, [h.split()[-1] for h in all_hints])]
        if hinted:
            chosen_results = hinted
    elif use_web and all_hints and not chosen_results:
        hint_text = ", ".join(all_hints)
        answer = f"not sure — I couldn't find recent web evidence mentioning {hint_text}."
        vector_block, vector_meta = _build_vector_references(docs)
        web_block, web_meta = _build_web_references(chosen_results)
        return {
            "answer": answer,
            "context": context,
            "vector_block": vector_block,
            "web_block": web_block,
            "vector_docs": docs,
            "web_results": chosen_results or [],
            "vector_meta": vector_meta,
            "web_meta": web_meta,
        }

    vector_block, vector_meta = _build_vector_references(docs)
    web_block, web_meta = _build_web_references(chosen_results if use_web else [])

    synth_prompt = (
        "You are a scholarly assistant synthesizing research insights.\n"
        "Craft a detailed answer (2 short paragraphs plus bullet points if appropriate) using BOTH the vector context and web evidence below.\n"
        "Always cite facts with [V#] for vector documents and [W#] for web results. If only one source type is available, say so explicitly.\n"
        "Finish with a 'Sources:' section listing only the items you cited.\n\n"
        f"Question:\n{query}\n\n"
        f"Vector context excerpts:\n{context or 'None'}\n\n"
        f"Vector references:\n{vector_block}\n\n"
        f"Web references:\n{web_block}\n"
    )

    response = llm.invoke([HumanMessage(content=synth_prompt)]).content
    return {
        "answer": response,
        "context": context,
        "vector_block": vector_block,
        "web_block": web_block,
        "vector_docs": docs,
        "web_results": chosen_results if use_web else [],
        "vector_meta": vector_meta,
        "web_meta": web_meta,
    }

def agentic_query(query):
    print(f"\nQuery: {query}\n")
    result = agentic_answer(query)
    print(f"\nFinal Answer:\n{result['answer']}\n")

def run_streamlit_app():
    try:
        import streamlit as st
    except Exception as exc:
        raise RuntimeError("Streamlit is required to run the UI. Install with `pip install streamlit`.") from exc

    st.set_page_config(page_title="Scholarly RAG Assistant", page_icon="📚", layout="wide")
    st.title("📚 Scholarly RAG Assistant")
    st.write(
        "Ask research-focused questions to retrieve answers synthesized from both the local scholarly vector store and live web evidence."
    )

    with st.sidebar:
        st.header("How it works")
        st.markdown(
            "- Retrieves relevant documents from the Chroma vector database.\n"
            "- Optionally searches the web (DuckDuckGo) for recent scholarly evidence.\n"
            "- Combines both sources through the Groq LLM, citing vector items as `[V#]` and web items as `[W#]`."
        )
        st.divider()
        st.markdown(
            "**Tip:** Include researcher names, topics, and years to help the agent focus both retrieval and web search."
        )
        st.divider()
        st.subheader("Performance")
        fast_mode = st.checkbox("Fast mode (skip web search)", value=True)
        vector_k = st.slider(
            "Vector matches",
            min_value=2,
            max_value=10,
            value=6,
            help="Number of top vector documents to retrieve before synthesis.",
        )
        web_k = 0
        if not fast_mode:
            web_k = st.slider(
                "Web results",
                min_value=3,
                max_value=10,
                value=6,
                help="Higher values improve coverage but increase latency.",
            )
        use_web = not fast_mode

    st.divider()
    query = st.text_area(
        "Enter your question",
        height=120,
        placeholder="e.g., Summarize 2022-2024 cloud security work by Dr. S. Rajarajeswari with citations",
    )

    if st.button("Run query", type="primary", use_container_width=True, disabled=not query.strip()):
        with st.spinner("Running hybrid retrieval and synthesis..."):
            result = agentic_answer(
                query.strip(),
                use_web=use_web,
                web_k=web_k,
                vector_k=vector_k,
            )

        st.subheader("Answer")
        st.markdown(result["answer"], unsafe_allow_html=True)

        with st.expander("Vector context excerpts", expanded=False):
            st.markdown(result["context"] or "_(No vector context retrieved.)_")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Vector references")
            if result["vector_meta"]:
                for item in result["vector_meta"]:
                    details = [
                        f"**{item['label']} — {item['title']}**",
                        f"Authors: {item['authors'] or 'N/A'}",
                        f"Year: {item['year'] or 'N/A'}",
                        f"Citations: {item['citations'] if item['citations'] not in (None, '', 'None') else 'N/A'}",
                    ]
                    if item["link"]:
                        details.append(f"[Source link]({item['link']})")
                    st.markdown("\n".join(details))
                    st.markdown("---")
            else:
                st.caption("No vector references available.")

        with col2:
            st.subheader("Web references")
            if result["web_meta"]:
                for item in result["web_meta"]:
                    details = [
                        f"**{item['label']} — {item['title']}**",
                        item["summary"] or "No snippet available.",
                    ]
                    if item["url"]:
                        details.append(f"[Visit URL]({item['url']})")
                    st.markdown("\n".join(details))
                    st.markdown("---")
            else:
                st.caption("No web references available.")
    else:
        st.info("Enter a question above and click *Run query* to begin.")

# ==============================================================
#  ENTRY POINTS
# ==============================================================

def _interactive_loop():
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break
        agentic_query(query)

# ==============================================================
#  INTERACTIVE LOOP
# ==============================================================
if __name__ == "__main__":
    if "--streamlit" in sys.argv or os.environ.get("STREAMLIT_SERVER_PORT"):
        run_streamlit_app()
    else:
        _interactive_loop()
