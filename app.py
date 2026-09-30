"""
AboveDeck: Below Deck RAG Demo
"""

import streamlit as st
import time
import os

st.set_page_config(
    page_title="AboveDeck: Below Deck RAG",
    page_icon="⚓",
    layout="wide",
)

#styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Source+Sans+3:wght@400;600&family=Source+Code+Pro:wght@400;600&display=swap');

    /* Root colors */
    :root {
        --navy: #1B2A4A;
        --ocean: #2E6F8E;
        --sky: #E8F1F8;
        --sand: #F5F0E8;
        --rope: #C4956A;
        --foam: #F0F6FA;
        --text: #1B2A4A;
        --text-light: #5A6B7D;
    }

    /* Global font */
    .block-container { padding-top: 1rem; max-width: 920px; }
    .stApp { font-family: 'Source Sans 3', sans-serif; color: var(--text); }
    .stMarkdown p, .stMarkdown li { font-family: 'Source Sans 3', sans-serif; }

    /* Top bar */
    .top-bar {
        background: linear-gradient(135deg, var(--navy) 0%, var(--ocean) 100%);
        padding: 1.5rem 2rem 1.25rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .top-bar::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 6px;
        background: var(--rope);
        opacity: 0.6;
    }
    .top-bar-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: white;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .top-bar-sub {
        color: rgba(255,255,255,0.75);
        font-size: 0.95rem;
        margin-top: 0.35rem;
        font-family: 'Source Sans 3', sans-serif;
    }

    /* Anchor SVG icon */
    .anchor-icon {
        width: 28px; height: 28px;
        fill: var(--rope);
        flex-shrink: 0;
        opacity: 0.9;
    }

    /* Wave divider */
    .wave-divider {
        width: 100%;
        height: 20px;
        margin: 0.75rem 0;
        opacity: 0.15;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: var(--foam); }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    section[data-testid="stSidebar"] h4 {
        font-family: 'DM Serif Display', serif;
        color: var(--navy);
        font-size: 1.05rem;
    }

    /* Config tag */
    .config-tag {
        display: inline-block;
        background: var(--sky);
        color: var(--ocean);
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'Source Code Pro', monospace;
    }

    /* Answer box */
    .answer-box {
        background: var(--foam);
        border-left: 3px solid var(--ocean);
        padding: 1.1rem 1.4rem;
        margin: 0.75rem 0 1.25rem 0;
        border-radius: 0 6px 6px 0;
        line-height: 1.65;
        font-size: 1rem;
    }

    /* Metrics row */
    .metrics-row {
        display: flex;
        gap: 2.5rem;
        margin: 1rem 0;
        padding: 0.75rem 1rem;
        background: var(--sky);
        border-radius: 6px;
    }
    .metric-item { display: flex; flex-direction: column; }
    .metric-label {
        font-size: 0.72rem;
        color: var(--text-light);
        letter-spacing: 0.03em;
        margin-bottom: 0.1rem;
    }
    .metric-value {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--navy);
        font-family: 'Source Code Pro', monospace;
    }

    /* Chunk display */
    .chunk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.25rem;
    }
    .chunk-source {
        font-size: 0.8rem;
        color: var(--text-light);
    }
    .chunk-score {
        font-size: 0.75rem;
        color: var(--text-light);
        font-family: 'Source Code Pro', monospace;
    }
    .chunk-text {
        font-size: 0.82rem;
        line-height: 1.55;
        color: var(--text);
        background: var(--foam);
        padding: 0.75rem;
        border-radius: 4px;
        border: 1px solid #dce8f0;
        font-family: 'Source Code Pro', monospace;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 200px;
        overflow-y: auto;
    }

    /* Example questions */
    .example-q {
        padding: 0.6rem 0.9rem;
        margin: 0.3rem 0;
        background: var(--foam);
        border-radius: 5px;
        font-size: 0.92rem;
        color: var(--text);
        border: 1px solid #dce8f0;
    }

    /* Sidebar stats */
    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        font-size: 0.85rem;
        font-family: 'Source Sans 3', sans-serif;
    }
    .sidebar-stat-label { color: var(--text-light); }
    .sidebar-stat-value { font-weight: 600; color: var(--navy); }

    /* Section headers */
    .section-label {
        font-family: 'DM Serif Display', serif;
        font-size: 1rem;
        color: var(--navy);
        margin: 1rem 0 0.5rem 0;
    }

    /* Hide default metric cards */
    [data-testid="stMetric"] { display: none; }
</style>
""", unsafe_allow_html=True)


#SVG elements
ANCHOR_SVG = '''<svg class="anchor-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2C10.34 2 9 3.34 9 5C9 6.3 9.84 7.4 11 7.82V10H8V12H11V19.92C8.16 19.48 6 17.74 5.35 15.25L3.26 15.84C4.23 19.3 7.81 22 12 22C16.19 22 19.77 19.3 20.74 15.84L18.65 15.25C18 17.74 15.84 19.48 13 19.92V12H16V10H13V7.82C14.16 7.4 15 6.3 15 5C15 3.34 13.66 2 12 2ZM12 4C12.55 4 13 4.45 13 5C13 5.55 12.55 6 12 6C11.45 6 11 5.55 11 5C11 4.45 11.45 4 12 4Z"/>
</svg>'''

WAVE_SVG = '''<svg class="wave-divider" viewBox="0 0 1200 20" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
<path d="M0,10 C150,0 150,20 300,10 C450,0 450,20 600,10 C750,0 750,20 900,10 C1050,0 1050,20 1200,10"
      stroke="#2E6F8E" fill="none" stroke-width="2"/>
</svg>'''


import ablation
from google import genai


#configs
BASELINE = {
    "chunk_size": 800, "chunking_strategy": "fixed",
    "embedding_model": "all-MiniLM-L6-v2", "top_k": 5,
    "retrieval_method": "dense", "reranker": "off", "query_transform": "raw",
}

def _cfg(label, description, **overrides):
    config = BASELINE.copy()
    config.update(overrides)
    return {"label": label, "description": description, "config": config}

AVAILABLE_CONFIGS = {
    "baseline": _cfg("Baseline (dense, k=5)", "Dense retrieval, MiniLM embeddings, top-5 chunks."),
    "bm25": _cfg("BM25 only", "Keyword retrieval. Strong on multi-hop, weak on lookups.", retrieval_method="bm25"),
    "hybrid": _cfg("Hybrid (dense + BM25)", "Reciprocal Rank Fusion of dense and BM25.", retrieval_method="hybrid"),
    "topk10": _cfg("Dense, k=10", "Baseline retrieval with 10 chunks instead of 5.", top_k=10),
    "hybrid+topk10": _cfg("Hybrid + k=10 (winner)", "Best config: +21% recall, multi-hop tripled.", retrieval_method="hybrid", top_k=10),
    "dense+rerank": _cfg("Dense + reranker", "Cross-encoder reranker. +229ms, zero metric gain.", reranker="on"),
}

def _setup_key(config):
    return (config["chunk_size"], config["chunking_strategy"], config["embedding_model"])


#cached setup
@st.cache_resource(show_spinner="Loading models and building indexes...")
def load_pipelines():
    from collections import defaultdict
    groups = defaultdict(list)
    for key, entry in AVAILABLE_CONFIGS.items():
        sk = _setup_key(entry["config"])
        groups[sk].append((key, entry["config"]))
    pipelines = {}
    for sk, config_list in groups.items():
        setup_config = config_list[0][1].copy()
        if any(c["reranker"] == "on" for _, c in config_list):
            setup_config["reranker"] = "on"
        pipeline = ablation.setup_pipeline(setup_config)
        pipelines[sk] = pipeline
    return pipelines

@st.cache_resource
def get_client():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("No GOOGLE_API_KEY found. Add it to .streamlit/secrets.toml or .env")
        st.stop()
    return genai.Client(api_key=api_key)


#sidebar
with st.sidebar:
    st.markdown("#### Configuration")
    config_key = st.selectbox(
        "Retrieval config",
        options=list(AVAILABLE_CONFIGS.keys()),
        index=list(AVAILABLE_CONFIGS.keys()).index("hybrid+topk10"),
        format_func=lambda k: AVAILABLE_CONFIGS[k]["label"],
        label_visibility="collapsed",
    )
    selected = AVAILABLE_CONFIGS[config_key]
    st.caption(selected["description"])

    st.markdown("---")
    st.markdown("#### Ablation results")
    for label, value in [
        ("Winner", "hybrid + k=10"),
        ("Recall@k", "0.404 (+21%)"),
        ("Multi-hop recall", "0.129 → 0.400"),
        ("Reranker cost", "+440ms, no gain"),
    ]:
        st.markdown(f'<div class="sidebar-stat"><span class="sidebar-stat-label">{label}</span><span class="sidebar-stat-value">{value}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Judge validation")
    for label, value in [("Exact match", "87%"), ("Within ±1", "100%"), ("Method", "30 items, blind")]:
        st.markdown(f'<div class="sidebar-stat"><span class="sidebar-stat-label">{label}</span><span class="sidebar-stat-value">{value}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Corpus: 22 Below Deck wiki pages")
    st.caption("Generator: Gemini 3.6 Flash")
    st.caption("Judge: GPT-4o-mini")


#header
st.markdown(
    f'<div class="top-bar">'
    f'<div class="top-bar-title">{ANCHOR_SVG} AboveDeck</div>'
    f'<div class="top-bar-sub">RAG over Below Deck wiki content, with a 12-config ablation study and validated LLM-as-judge eval.</div>'
    f'</div>',
    unsafe_allow_html=True,
)

query = st.text_input(
    "Query",
    placeholder="Ask a question about Below Deck...",
    label_visibility="collapsed",
)

if query:
    config = selected["config"]
    pipelines = load_pipelines()
    client = get_client()
    sk = _setup_key(config)
    pipeline = pipelines[sk]

    with st.spinner(""):
        t0 = time.time()
        _, retrieval_result = ablation.run_query(config, query, client, pipeline, return_generation=False)
        retrieval_ms = (time.time() - t0) * 1000
        t_gen = time.time()
        response, _ = ablation.run_query(config, query, client, pipeline, return_generation=True)
        generation_ms = (time.time() - t_gen) * 1000

    st.markdown(f'<div class="answer-box">{response}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="metrics-row">'
        f'<div class="metric-item"><span class="metric-label">config</span><span class="metric-value">{config_key}</span></div>'
        f'<div class="metric-item"><span class="metric-label">retrieval</span><span class="metric-value">{retrieval_ms:.0f}ms</span></div>'
        f'<div class="metric-item"><span class="metric-label">generation</span><span class="metric-value">{generation_ms:.0f}ms</span></div>'
        f'<div class="metric-item"><span class="metric-label">chunks</span><span class="metric-value">{len(retrieval_result)}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(WAVE_SVG, unsafe_allow_html=True)
    st.markdown('<p class="section-label">Retrieved Chunks</p>', unsafe_allow_html=True)

    for i, chunk in enumerate(retrieval_result, 1):
        source = chunk["meta"].get("source_page", "unknown")
        season = chunk["meta"].get("season", "?")
        show = chunk["meta"].get("show", "?")
        score = chunk.get("dist", 0)
        show_short = "Med" if "Mediterranean" in show else "BD"

        with st.expander(f"{i}. {show_short} S{season} — {source}", expanded=(i <= 2)):
            st.markdown(
                f'<div class="chunk-header"><span class="chunk-source">{show}, Season {season}</span>'
                f'<span class="chunk-score">score: {score:.4f}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="chunk-text">{chunk["doc"][:800]}</div>', unsafe_allow_html=True)
else:
    st.markdown(WAVE_SVG, unsafe_allow_html=True)
    st.markdown('<p class="section-label">Example Questions</p>', unsafe_allow_html=True)
    for q in [
        "Who was the chief stewardess in Below Deck Season 1?",
        "Which chef appeared on Below Deck Season 6 and later returned?",
        "Was there a Below Deck Mediterranean Season 11?",
        "Who was the captain in Below Deck Mediterranean Season 3?",
        "How many seasons of Below Deck Mediterranean have aired?",
    ]:
        st.markdown(f'<div class="example-q">{q}</div>', unsafe_allow_html=True)