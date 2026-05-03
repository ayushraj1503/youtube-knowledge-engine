#!/usr/bin/env python3
# scripts/quickstart_demo.py
"""
Quickstart demo script — ingests a small channel and runs example queries.
Run: python scripts/quickstart_demo.py
Requires: backend running on localhost:8000
"""

import time
import requests

BASE = "http://localhost:8000/api/v1"

DEMO_CHANNEL  = "https://www.youtube.com/@3blue1brown"
DEMO_QUERIES  = [
    "How does neural network backpropagation work?",
    "What is the intuition behind gradient descent?",
    "Explain the relationship between linear algebra and machine learning",
    "How are eigenvalues useful in data science?",
    "What makes transformers different from RNNs?",
]

def banner(text):
    print(f"\n{'═'*60}")
    print(f"  {text}")
    print('═'*60)

def check_backend():
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        h = r.json()
        print(f"✅ Backend healthy — status: {h['status']}, version: {h['version']}")
        for name, comp in h.get("components", {}).items():
            status = "✅" if comp["status"] == "ok" else "⚠️"
            print(f"   {status} {name}: {comp['status']}")
        return True
    except Exception as e:
        print(f"❌ Backend not reachable: {e}")
        print("   Start with: uvicorn backend.main:app --port 8000")
        return False

def ingest_channel(channel_url, max_videos=10):
    banner(f"Ingesting channel (max {max_videos} videos)")
    print(f"  URL: {channel_url}")

    r = requests.post(f"{BASE}/ingest/channel", json={
        "channel_url": channel_url,
        "max_videos": max_videos,
        "force_reingest": False,
        "extract_frames": False,   # faster for demo
    })

    if r.status_code not in (200, 202):
        print(f"❌ Ingest failed: {r.text}")
        return None

    job = r.json()
    job_id = job["job_id"]
    print(f"  Job ID: {job_id}")

    # Poll for completion
    print("\n  Polling for progress...")
    while True:
        status_r = requests.get(f"{BASE}/ingest/status/{job_id}")
        s = status_r.json()
        pct = s.get("progress_percent", 0)
        status = s.get("status")
        processed = s.get("videos_processed", 0)
        total = s.get("videos_total", "?")
        chunks = s.get("chunks_created", 0)

        print(f"  [{pct:5.1f}%] {status:10s} — {processed}/{total} videos — {chunks} chunks", end="\r")

        if status in ("completed", "failed", "partial"):
            print()
            if status == "completed":
                print(f"\n  ✅ Ingestion complete!")
                print(f"     Videos: {processed} | Chunks: {chunks}")
            else:
                print(f"\n  ⚠️  Status: {status}")
                if s.get("error_message"):
                    print(f"     Error: {s['error_message']}")
            return job_id

        time.sleep(3)

def run_query(query_text, top_k=3):
    print(f"\n  📝 Query: {query_text}")

    r = requests.post(f"{BASE}/query", json={
        "query": query_text,
        "top_k": top_k,
        "search_mode": "hybrid",
        "rerank": True,
        "rewrite_query": True,
    })

    if r.status_code != 200:
        print(f"  ❌ Query failed ({r.status_code}): {r.text[:200]}")
        return

    data = r.json()

    if data.get("rewritten_query") and data["rewritten_query"] != query_text:
        print(f"  🔄 Rewritten: {data['rewritten_query']}")

    print(f"\n  💬 Answer:\n  {'─'*50}")
    # Print answer with line wrapping
    answer = data.get("answer", "No answer")
    for line in answer.split("\n"):
        print(f"  {line}")

    sources = data.get("sources", [])
    print(f"\n  📌 Sources ({len(sources)}):")
    for i, src in enumerate(sources[:3], 1):
        ts = int(src["start_time"])
        m, s = divmod(ts, 60)
        score = int(src["score"] * 100)
        print(f"  [{i}] {src['video_title'][:50]} @ {m:02d}:{s:02d} (score: {score}%)")
        print(f"      🔗 {src['timestamp_url']}")

    print(f"\n  ⏱ Latency: {data.get('latency_ms', 0):.0f}ms | "
          f"Tokens: {data.get('tokens_used') or '—'}")

def list_videos():
    banner("Video Library")
    r = requests.get(f"{BASE}/videos?page=1&page_size=10")
    if r.status_code != 200:
        print(f"❌ Failed: {r.text}")
        return

    data = r.json()
    print(f"  Total videos indexed: {data['total']}\n")
    for v in data["videos"][:10]:
        dur = v.get("duration_seconds", 0)
        m, s = divmod(dur, 60)
        print(f"  📹 {v['title'][:60]}")
        print(f"     Channel: {v['channel_name']} | Duration: {m}m {s}s")


if __name__ == "__main__":
    banner("YouTube Knowledge Engine — Quickstart Demo")

    # 1. Health check
    if not check_backend():
        exit(1)

    # 2. Ingest
    job_id = ingest_channel(DEMO_CHANNEL, max_videos=5)

    if not job_id:
        print("\nSkipping queries — ingestion failed.")
        exit(1)

    # 3. Show library
    list_videos()

    # 4. Run demo queries
    banner("Running Example Queries")
    for q in DEMO_QUERIES[:3]:
        run_query(q, top_k=3)
        print()
        time.sleep(1)  # be gentle on Groq rate limits

    banner("Demo Complete 🎉")
    print("  Open http://localhost:8501 for the full Streamlit UI")
