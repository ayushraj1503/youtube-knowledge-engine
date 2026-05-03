# frontend/app.py
"""
YouTube Knowledge Engine — Streamlit Frontend

Features:
  - Channel ingestion with real-time progress polling
  - Multi-mode query interface (Hybrid / Semantic / BM25)
  - Source cards with timestamp deep-links
  - Video library browser with pagination
  - System health dashboard

Run: streamlit run frontend/app.py
"""

import time
from typing import Optional

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Knowledge Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config ────────────────────────────────────────────────────────────────────
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/api/v1"


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .main-header h1 {
        color: #fff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: rgba(255,255,255,0.65);
        margin: 0.4rem 0 0;
        font-size: 1rem;
    }

    .source-card {
        background: #1a1a2e;
        border: 1px solid rgba(124, 77, 255, 0.3);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }

    .source-card:hover {
        border-color: rgba(124, 77, 255, 0.7);
    }

    .source-card .video-title {
        font-weight: 600;
        color: #c084fc;
        font-size: 0.95rem;
    }

    .source-card .timestamp {
        font-family: 'JetBrains Mono', monospace;
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.3rem;
    }

    .source-card .excerpt {
        color: #e2e8f0;
        font-size: 0.88rem;
        margin-top: 0.6rem;
        line-height: 1.5;
        border-left: 3px solid rgba(124, 77, 255, 0.4);
        padding-left: 0.75rem;
    }

    .score-badge {
        display: inline-block;
        background: rgba(124, 77, 255, 0.15);
        color: #c084fc;
        border: 1px solid rgba(124, 77, 255, 0.3);
        border-radius: 20px;
        padding: 0.15rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    .answer-box {
        background: linear-gradient(135deg, #0d1117, #161b22);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        color: #e6edf3;
        line-height: 1.7;
        font-size: 0.95rem;
    }

    .metric-card {
        background: #1a1a2e;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .metric-card .label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-card .value {
        color: #c084fc;
        font-size: 1.6rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .stButton>button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: opacity 0.2s;
    }

    .stButton>button:hover {
        opacity: 0.85;
    }

    .tag {
        display: inline-block;
        background: rgba(79, 70, 229, 0.2);
        color: #818cf8;
        border: 1px solid rgba(79, 70, 229, 0.3);
        border-radius: 4px;
        padding: 0.1rem 0.4rem;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 0.3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="main-header">
    <h1>🎬 YouTube Knowledge Engine</h1>
    <p>Production-grade RAG system — search across entire YouTube channels with AI</p>
</div>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_post(endpoint: str, payload: dict) -> Optional[dict]:
    try:
        r = requests.post(f"{API_BASE}/{endpoint}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Is the FastAPI server running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_get(endpoint: str, params: dict = None) -> Optional[dict]:
    try:
        r = requests.get(f"{API_BASE}/{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    search_mode = st.selectbox(
        "Search Mode",
        ["hybrid", "semantic", "bm25"],
        help="Hybrid = BM25 + Vector (best quality). Semantic = dense vectors only. BM25 = keyword only.",
    )

    top_k = st.slider("Results (Top-K)", min_value=1, max_value=20, value=5)

    col1, col2 = st.columns(2)
    rerank = col1.toggle("Re-rank", value=True, help="Cross-encoder re-ranking")
    rewrite = col2.toggle("Query rewrite", value=True, help="LLM query optimization")

    st.divider()

    # System health
    st.markdown("## 🏥 System Health")
    if st.button("Check Health"):
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=5)
            h = r.json()
            status_emoji = "🟢" if h["status"] == "ok" else "🟡"
            st.markdown(f"{status_emoji} **{h['status'].upper()}** — v{h['version']}")
            for name, comp in h.get("components", {}).items():
                emoji = "✅" if comp["status"] == "ok" else ("⚠️" if comp["status"] == "degraded" else "❌")
                latency = f" ({comp['latency_ms']}ms)" if comp.get("latency_ms") else ""
                st.markdown(f"{emoji} `{name}`{latency}")
        except Exception:
            st.error("Backend unreachable")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_ingest, tab_video, tab_query, tab_library = st.tabs(
    ["📥 Ingest Channel", "🎬 Ingest Video", "🔍 Ask a Question", "📚 Video Library"]
)


# ════════════════════════════════════════════════════════════════════════
# TAB 1: INGEST
# ════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.markdown("### Ingest a YouTube Channel")
    st.markdown(
        "Enter a YouTube channel URL or `@handle`. "
        "The system will fetch all videos, extract transcripts, and index them."
    )

    with st.form("ingest_form"):
        channel_url = st.text_input(
            "Channel URL",
            placeholder="https://www.youtube.com/@3blue1brown",
        )
        col1, col2 = st.columns(2)
        max_videos = col1.number_input("Max videos", min_value=1, max_value=5000, value=50)
        extract_frames = col2.toggle("Extract frames (BLIP)", value=True)
        force_reingest = st.toggle("Force re-ingest (overwrite existing)", value=False)
        submit = st.form_submit_button("🚀 Start Ingestion", use_container_width=True)

    if submit:
        if not channel_url.strip():
            st.warning("Please enter a channel URL.")
        else:
            with st.spinner("Starting ingestion job..."):
                resp = api_post(
                    "ingest/channel",
                    {
                        "channel_url": channel_url,
                        "max_videos": max_videos,
                        "force_reingest": force_reingest,
                        "extract_frames": extract_frames,
                    },
                )

            if resp:
                job_id = resp.get("job_id")
                st.success(f"✅ Job started! ID: `{job_id}`")
                st.session_state["current_job_id"] = job_id

    # Progress polling
    if "current_job_id" in st.session_state:
        job_id = st.session_state["current_job_id"]
        st.divider()
        st.markdown(f"#### Progress — Job `{job_id}`")

        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()

        auto_poll = st.toggle("Auto-poll every 3s", value=True)

        if auto_poll or st.button("Refresh Status"):
            job = api_get(f"ingest/status/{job_id}")
            if job:
                pct = job.get("progress_percent", 0)
                progress_placeholder.progress(pct / 100.0)

                status_color = {
                    "completed": "🟢", "running": "🔵",
                    "failed": "🔴", "partial": "🟡", "pending": "⚪"
                }.get(job["status"], "⚪")

                status_placeholder.markdown(
                    f"{status_color} **{job['status'].upper()}** — "
                    f"Channel: {job.get('channel_name', 'Discovering...')}"
                )

                with metrics_placeholder.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Videos Total", job.get("videos_total", 0))
                    c2.metric("Processed", job.get("videos_processed", 0))
                    c3.metric("Failed", job.get("videos_failed", 0))
                    c4.metric("Chunks Created", job.get("chunks_created", 0))

                if job["status"] == "running" and auto_poll:
                    time.sleep(3)
                    st.rerun()


# ════════════════════════════════════════════════════════════════════════
# TAB 2: QUERY
# ════════════════════════════════════════════════════════════════════════
with tab_query:
    st.markdown("### Ask a Question")
    st.markdown(
        "Ask anything about the ingested YouTube content. "
        "The AI retrieves relevant transcript segments and generates a cited answer."
    )

    # Example queries
    with st.expander("💡 Example queries"):
        examples = [
            "How does backpropagation work?",
            "What is the difference between supervised and unsupervised learning?",
            "Explain the attention mechanism in transformers",
            "What tools does he recommend for beginners?",
            "How can I improve model performance without more data?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}"):
                st.session_state["query_input"] = ex

    query_text = st.text_area(
        "Your question",
        value=st.session_state.get("query_input", ""),
        placeholder="What is explained in these videos?",
        height=80,
    )

    if st.button("🔍 Search & Answer", type="primary", use_container_width=True):
        if not query_text.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                resp = api_post(
                    "query",
                    {
                        "query": query_text,
                        "top_k": top_k,
                        "search_mode": search_mode,
                        "rerank": rerank,
                        "rewrite_query": rewrite,
                    },
                )

            if resp:
                # Query rewriting notice
                if resp.get("rewritten_query") and resp["rewritten_query"] != query_text:
                    st.info(f"🔄 Query rewritten to: *{resp['rewritten_query']}*")

                # Metrics row
                col1, col2, col3 = st.columns(3)
                col1.metric("Sources found", len(resp.get("sources", [])))
                col2.metric("Latency", f"{resp.get('latency_ms', 0):.0f}ms")
                col3.metric("Tokens used", resp.get("tokens_used") or "—")

                st.divider()

                # Answer
                st.markdown("#### 💬 Answer")
                st.markdown(
                    f'<div class="answer-box">{resp["answer"]}</div>',
                    unsafe_allow_html=True,
                )

                # Sources
                sources = resp.get("sources", [])
                if sources:
                    st.divider()
                    st.markdown(f"#### 📌 Sources ({len(sources)})")

                    for i, src in enumerate(sources, 1):
                        ts = format_timestamp(src["start_time"])
                        ts_end = format_timestamp(src["end_time"])
                        score_pct = int(src["score"] * 100)

                        st.markdown(
                            f"""<div class="source-card">
<div class="video-title">
  [{i}] {src['video_title']}
  <span class="score-badge">score: {score_pct}%</span>
</div>
<div class="timestamp">⏱ {ts} → {ts_end} &nbsp;|&nbsp; 📺 {src['channel_name']}</div>
{"" if not src.get('frame_caption') else f'<div class="timestamp">👁 Visual: {src["frame_caption"]}</div>'}
<div class="excerpt">{src['text'][:300]}{'...' if len(src['text']) > 300 else ''}</div>
</div>""",
                            unsafe_allow_html=True,
                        )

                        st.link_button(
                            f"▶ Watch at {ts}",
                            src["timestamp_url"],
                        )


# ════════════════════════════════════════════════════════════════════════
# TAB: SINGLE VIDEO
# ════════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown("### 🎬 Ingest a Single YouTube Video")
    st.markdown("Paste any YouTube video URL to add just that video to your knowledge base.")

    with st.form("ingest_video_form"):
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )
        col1, col2 = st.columns(2)
        extract_frames_v = col1.toggle("Extract frames (BLIP)", value=False)
        force_reingest_v = col2.toggle("Force re-ingest", value=False)
        submit_video = st.form_submit_button("🎬 Ingest Video", use_container_width=True)

    if submit_video:
        if not video_url.strip():
            st.warning("Please enter a YouTube video URL.")
        else:
            with st.spinner("Starting video ingestion..."):
                resp = api_post(
                    "ingest/video",
                    {
                        "video_url": video_url,
                        "force_reingest": force_reingest_v,
                        "extract_frames": extract_frames_v,
                    },
                )
            if resp:
                job_id = resp.get("job_id")
                st.success(f"✅ Job started! ID: `{job_id}`")
                st.session_state["video_job_id"] = job_id

    if "video_job_id" in st.session_state:
        job_id = st.session_state["video_job_id"]
        st.divider()
        st.markdown(f"#### Progress — Job `{job_id}`")

        if st.button("Refresh Status", key="refresh_video"):
            job = api_get(f"ingest/status/{job_id}")
            if job:
                pct = job.get("progress_percent", 0)
                st.progress(pct / 100.0)
                status = job.get("status")
                status_emoji = {"completed": "✅", "running": "🔵", "failed": "❌", "pending": "⚪"}.get(status, "⚪")
                st.markdown(f"{status_emoji} **{status.upper()}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Status", status)
                c2.metric("Chunks Created", job.get("chunks_created", 0))
                c3.metric("Progress", f"{pct:.0f}%")
                if job.get("error_message"):
                    st.error(f"Error: {job['error_message']}")
                if status == "completed":
                    st.balloons()
                    st.success("✅ Video ingested! Go to 'Ask a Question' tab to query it.")

# ════════════════════════════════════════════════════════════════════════
# TAB 3: VIDEO LIBRARY
# ════════════════════════════════════════════════════════════════════════
with tab_library:
    st.markdown("### Video Library")
    st.markdown("Browse all indexed videos in the knowledge base.")

    col1, col2 = st.columns([3, 1])
    page = col2.number_input("Page", min_value=1, value=1)

    data = api_get("videos", params={"page": page, "page_size": 20})

    if data:
        st.markdown(
            f"**{data['total']} videos** indexed across all channels "
            f"(page {data['page']} of {max(1, -(-data['total'] // 20))})"
        )

        videos = data.get("videos", [])
        if not videos:
            st.info("No videos indexed yet. Ingest a channel first!")
        else:
            for video in videos:
                with st.expander(f"📹 {video['title']}"):
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**Channel:** {video.get('channel_name', '—')}")
                    c1.markdown(f"**Published:** {video.get('published_at', '—')}")
                    duration = video.get("duration_seconds", 0)
                    m, s = divmod(duration, 60)
                    c2.metric("Duration", f"{m}m {s}s")
                    if video.get("url"):
                        st.link_button("▶ Open on YouTube", video["url"])

        if data.get("has_next"):
            st.info("More videos available — increase the page number.")
    else:
        st.info("Could not load video library. Make sure the backend is running.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color: #475569; font-size: 0.8rem;'>"
    "YouTube Knowledge Engine • Production RAG System • "
    "FastAPI + ChromaDB + Groq + sentence-transformers"
    "</p>",
    unsafe_allow_html=True,
)
