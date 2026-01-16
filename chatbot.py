# chatbot.py
# Run with: streamlit run chatbot.py

import os
import re
import json
import time
import base64
import hashlib
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI

# =========================
# Optional native deps
# =========================
try:
    import faiss
except Exception:
    faiss = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="The Vaillant Brain", layout="wide")

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_pdfs")
KB_DIR = os.path.join(BASE_DIR, "kb")
LOG_DIR = os.path.join(BASE_DIR, "logs")

KB_INDEX = os.path.join(KB_DIR, "index.faiss")
KB_META = os.path.join(KB_DIR, "meta.json")
IMAGE_FEEDBACK_FILE = os.path.join(LOG_DIR, "image_feedback.jsonl")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KB_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# 🔑 OpenAI key (hidden)
# =========================
def get_openai_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

def require_openai_key() -> bool:
    if not get_openai_key():
        st.error(
            "❌ OpenAI API key missing.\n\n"
            "Set OPENAI_API_KEY as an environment variable or in Streamlit secrets."
        )
        return False
    return True

def get_client() -> OpenAI:
    return OpenAI(api_key=get_openai_key())

# =========================
# Session state
# =========================
def init_state():
    defaults = {
        "manual_pdf_path": None,
        "kb_ready": False,
        "retrieved_items": [],
        "answer": None,
        "image_feedback_given": set(),  # {(chunk_id, img_idx)}
        "kb_rebuild_complete": False,
        "kb_last_stats": None,
        # feedback cache
        "img_feedback_cache": None,  # dict
        "img_feedback_cache_mtime": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =========================
# Helpers
# =========================
def bytes_to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode()

def b64_to_bytes(s: str) -> bytes:
    return base64.b64decode(s.encode())

def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]

def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

# =========================
# Image helpers
# =========================
def looks_like_blank(img_bytes: bytes) -> bool:
    return not img_bytes or len(img_bytes) < 8000

def extract_page_images(pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
    if fitz is None:
        return []

    out: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)

        # embedded raster
        for img in page.get_images(full=True):
            try:
                base = doc.extract_image(img[0])
                b = base.get("image")
                if not b or looks_like_blank(b):
                    continue
                out.append({"b64": bytes_to_b64(b), "kind": "embedded", "page": page_num})
            except Exception:
                continue

        # render fallback
        if not out:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                b = pix.tobytes("png")
                if not looks_like_blank(b):
                    out.append({"b64": bytes_to_b64(b), "kind": "render", "page": page_num})
            except Exception:
                pass

        doc.close()
    except Exception:
        pass

    return out

# =========================
# Embeddings
# =========================
def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    vecs: List[np.ndarray] = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        r = client.embeddings.create(model="text-embedding-3-large", input=batch)
        data = list(r.data)
        # keep order if index exists
        if data and hasattr(data[0], "index"):
            data.sort(key=lambda d: d.index)  # type: ignore
        for d in data:
            vecs.append(np.array(d.embedding, dtype="float32"))
    return np.vstack(vecs) if vecs else np.zeros((0, 0), dtype="float32")

# =========================
# FAISS
# =========================
@st.cache_resource
def load_faiss():
    if faiss is None or not os.path.exists(KB_INDEX) or not os.path.exists(KB_META):
        return None, None
    try:
        return (
            faiss.read_index(KB_INDEX),
            json.load(open(KB_META, "r", encoding="utf-8")),
        )
    except Exception:
        return None, None

# =========================
# Feedback aggregation (images)
# =========================
def _read_feedback_file() -> List[Dict[str, Any]]:
    if not os.path.exists(IMAGE_FEEDBACK_FILE):
        return []
    rows: List[Dict[str, Any]] = []
    with open(IMAGE_FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def get_image_feedback_stats(force: bool = False) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Returns stats keyed by (chunk_id, image_index):
      {
        (cid, idx): {
            "pos": int,
            "neg": int,
            "n": int,
            "score": float,   # smoothed helpfulness in [0,1]
        }
      }
    Uses a simple Bayesian smoothing prior so early votes don't swing too hard:
      score = (pos + 1) / (pos + neg + 2)
    """
    try:
        mtime = os.path.getmtime(IMAGE_FEEDBACK_FILE) if os.path.exists(IMAGE_FEEDBACK_FILE) else None
    except Exception:
        mtime = None

    if (not force) and st.session_state.get("img_feedback_cache") is not None:
        if st.session_state.get("img_feedback_cache_mtime") == mtime:
            return st.session_state.img_feedback_cache  # type: ignore

    stats: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in _read_feedback_file():
        cid = row.get("chunk_id")
        idx = row.get("image_index")
        helpful = row.get("helpful")
        if not cid or idx is None:
            continue
        key = (str(cid), int(idx))
        if key not in stats:
            stats[key] = {"pos": 0, "neg": 0, "n": 0, "score": 0.5}
        if helpful is True:
            stats[key]["pos"] += 1
        elif helpful is False:
            stats[key]["neg"] += 1
        stats[key]["n"] += 1

    # smoothed score
    for key, d in stats.items():
        pos, neg = d["pos"], d["neg"]
        d["score"] = (pos + 1) / (pos + neg + 2)  # Laplace smoothing

    st.session_state.img_feedback_cache = stats
    st.session_state.img_feedback_cache_mtime = mtime
    return stats

def image_penalty_factor(score_0_1: float, n_votes: int) -> float:
    """
    Convert image helpfulness score into a multiplicative factor for retrieval ranking.
    - score near 1 => boost slightly
    - score near 0 => penalize
    Make effect mild when vote count is low.
    """
    # vote confidence scale in [0.0..1.0], ramps up by ~8 votes
    conf = min(1.0, n_votes / 8.0)
    # map score into [-1..+1]
    centered = (score_0_1 - 0.5) * 2.0
    # max +/- 25% effect at high confidence
    return 1.0 + (0.25 * centered * conf)

def chunk_image_quality(chunk_id: str, num_images: int, stats: Dict[Tuple[str, int], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize image feedback for a chunk:
      avg_score, n_votes_total, penalty_factor
    If no votes, neutral.
    """
    if num_images <= 0:
        return {"avg_score": 0.5, "n_votes": 0, "factor": 1.0}

    scores = []
    n_total = 0
    for idx in range(num_images):
        d = stats.get((chunk_id, idx))
        if not d:
            continue
        scores.append(float(d.get("score", 0.5)))
        n_total += int(d.get("n", 0))

    if not scores:
        return {"avg_score": 0.5, "n_votes": 0, "factor": 1.0}

    avg = float(np.mean(scores))
    factor = image_penalty_factor(avg, n_total)
    return {"avg_score": avg, "n_votes": n_total, "factor": factor}

# =========================
# KB rebuild
# =========================
def rebuild_kb():
    if not require_openai_key():
        return

    pdf = st.session_state.manual_pdf_path
    if not pdf:
        st.error("Upload a PDF first.")
        return

    if faiss is None:
        st.error("faiss-cpu not installed")
        return

    client = get_client()

    try:
        reader = PdfReader(pdf)
    except Exception as e:
        st.error(f"Could not open PDF: {e}")
        return

    items: List[Dict[str, Any]] = []
    img_pages = 0
    img_total = 0
    prog = st.progress(0)
    status = st.empty()
    n_pages = len(reader.pages)

    for i, page in enumerate(reader.pages, start=1):
        status.write(f"Processing page {i}/{n_pages} ...")
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""

        if len(text) < 120:
            prog.progress(int(i / n_pages * 70))
            continue

        images = extract_page_images(pdf, i)
        if images:
            img_pages += 1
            img_total += len(images)

        for ch in re.split(r"\n{2,}", text):
            ch = ch.strip()
            if not ch:
                continue
            items.append({
                "page": i,
                "chunk_id": f"p{i}-{stable_hash(ch)}",
                "text": ch,
                "images": images,
            })

        prog.progress(int(i / n_pages * 70))

    if not items:
        st.error("No usable text extracted from PDF.")
        return

    status.write(f"Embedding {len(items)} chunks ...")
    emb = embed_texts(client, [x["text"] for x in items])
    if emb.size == 0:
        st.error("Embedding matrix empty.")
        return

    faiss.normalize_L2(emb)
    status.write("Building FAISS index ...")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    prog.progress(90)

    faiss.write_index(index, KB_INDEX)
    with open(KB_META, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    st.cache_resource.clear()
    st.session_state.kb_ready = True
    st.session_state.kb_rebuild_complete = True
    st.session_state.kb_last_stats = {
        "chunks": len(items),
        "img_pages": img_pages,
        "img_total": img_total,
    }
    # refresh feedback cache (not strictly required)
    st.session_state.img_feedback_cache = None
    st.session_state.img_feedback_cache_mtime = None

# =========================
# Retrieval (down-rank bad diagrams)
# =========================
def retrieve(query: str, k: int = 6):
    if not require_openai_key():
        return []

    index, meta = load_faiss()
    if not index or not meta or faiss is None:
        return []

    client = get_client()
    r = client.embeddings.create(model="text-embedding-3-large", input=query)
    qv = np.array(r.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(qv)

    scores, ids = index.search(qv, max(k * 4, 12))  # overfetch so rerank has room
    raw: List[Dict[str, Any]] = []
    for s, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(meta):
            continue
        it = dict(meta[idx])
        it["similarity"] = float(s)
        raw.append(it)

    # Apply image-quality rerank
    stats = get_image_feedback_stats()
    for it in raw:
        cid = str(it.get("chunk_id", ""))
        imgs = it.get("images") or []
        qsum = chunk_image_quality(cid, len(imgs), stats)
        it["img_avg_score"] = qsum["avg_score"]
        it["img_votes"] = qsum["n_votes"]
        it["img_factor"] = qsum["factor"]
        # final score = similarity * image factor
        it["final_score"] = float(it.get("similarity", 0.0)) * float(it["img_factor"])

    raw.sort(key=lambda x: float(x.get("final_score", x.get("similarity", 0.0))), reverse=True)
    return raw[:k]

# =========================
# Sidebar
# =========================
def sidebar_ui():
    st.header("📘 Knowledge Base")

    pdf = st.file_uploader("Upload manual PDF", type=["pdf"])
    if pdf:
        path = os.path.join(UPLOAD_DIR, safe_filename(pdf.name))
        with open(path, "wb") as f:
            f.write(pdf.read())
        st.session_state.manual_pdf_path = path
        st.success(f"Loaded {pdf.name}")

    if st.button("🔁 Rebuild Knowledge Base"):
        rebuild_kb()

    if st.session_state.get("kb_rebuild_complete", False):
        s = st.session_state.get("kb_last_stats") or {}
        st.toast("✅ Knowledge Base rebuild COMPLETE", icon="✅")
        st.success(
            f"✅ Knowledge base rebuild COMPLETE\n\n"
            f"- Chunks: {s.get('chunks')}\n"
            f"- Pages with images: {s.get('img_pages')}\n"
            f"- Total images: {s.get('img_total')}"
        )
        st.session_state.kb_rebuild_complete = False

    st.markdown("---")
    st.subheader("🖼️ Diagram quality controls")
    st.checkbox("Hide diagrams that are strongly down-voted", value=True, key="hide_bad_images")
    st.slider("Hide threshold (helpfulness score)", 0.0, 1.0, 0.25, 0.05, key="hide_bad_images_threshold")
    st.caption("Helpfulness is learned from your 👍/👎 votes (smoothed).")

# =========================
# Image feedback persistence
# =========================
def save_image_feedback(chunk_id: str, img_index: int, helpful: bool):
    entry = {"ts": time.time(), "chunk_id": chunk_id, "image_index": img_index, "helpful": helpful}
    with open(IMAGE_FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    # bust cache so new votes affect ranking immediately
    st.session_state.img_feedback_cache = None
    st.session_state.img_feedback_cache_mtime = None

# =========================
# Chat UI
# =========================
def chat_ui():
    st.title("💬 AI Simulation Assistant")

    if not st.session_state.kb_ready:
        st.info("Upload a PDF and rebuild the knowledge base to begin.")
        return

    q = st.text_input("Ask a question")
    if st.button("Ask") and q:
        ctx = retrieve(q)
        st.session_state.retrieved_items = ctx

        manual_text = "\n\n".join(
            f"[Page {c['page']}]\n{c['text']}" for c in ctx
        )

        client = get_client()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer ONLY from the manual text. Cite page numbers."},
                {"role": "user", "content": manual_text + "\n\n" + q},
            ],
        )
        st.session_state.answer = r.choices[0].message.content

    if st.session_state.answer:
        st.markdown("### ✅ Answer")
        st.markdown(st.session_state.answer)

    # Show retrieval + diagram quality summary
    if st.session_state.retrieved_items:
        with st.expander("🔎 Retrieval details (incl. diagram quality)", expanded=False):
            for i, it in enumerate(st.session_state.retrieved_items, start=1):
                st.write(
                    f"{i}. Page {it.get('page')} | sim={it.get('similarity', 0):.3f} "
                    f"| img_score={it.get('img_avg_score', 0.5):.2f} (votes={it.get('img_votes',0)}) "
                    f"| final={it.get('final_score', it.get('similarity',0)):.3f}"
                )

    # Diagrams
    if st.session_state.retrieved_items:
        st.markdown("### 📷 Related diagrams")

        pdf_name = os.path.basename(st.session_state.manual_pdf_path)
        stats = get_image_feedback_stats()
        hide_bad = bool(st.session_state.get("hide_bad_images", True))
        hide_thr = float(st.session_state.get("hide_bad_images_threshold", 0.25))

        for it in st.session_state.retrieved_items:
            chunk_id = str(it.get("chunk_id"))
            images = it.get("images") or []
            page_fallback = it.get("page", "?")

            if not images:
                continue

            # sort images: higher helpfulness first (unknown = 0.5)
            def img_sort_key(idx_img: Tuple[int, Dict[str, Any]]) -> float:
                idx, _img = idx_img
                d = stats.get((chunk_id, idx))
                return float(d.get("score", 0.5)) if d else 0.5

            images_sorted = list(enumerate(images))
            images_sorted.sort(key=img_sort_key, reverse=True)

            for idx, img in images_sorted:
                img_bytes = b64_to_bytes(img["b64"])
                page = img.get("page", page_fallback)

                fb = stats.get((chunk_id, idx))
                score = float(fb.get("score", 0.5)) if fb else 0.5
                votes = int(fb.get("n", 0)) if fb else 0

                # optional hide
                if hide_bad and votes >= 3 and score <= hide_thr:
                    st.caption(
                        f"🚫 Hidden down-voted diagram (score={score:.2f}, votes={votes}) "
                        f"— Document: {pdf_name}, Page {page}"
                    )
                    continue

                st.image(img_bytes, use_container_width=True)

                st.caption(
                    f"**Document:** {pdf_name}  \n"
                    f"**Recommended page:** Page {page}  \n"
                    f"**Diagram helpfulness:** {score:.2f} (votes: {votes})"
                )

                img_key = (chunk_id, idx)
                if img_key not in st.session_state.image_feedback_given:
                    choice = st.radio(
                        "Was this image helpful?",
                        ["👍 Yes", "👎 No"],
                        index=None,
                        key=f"imgfb_{chunk_id}_{idx}",
                        horizontal=True,
                    )
                    if choice:
                        save_image_feedback(chunk_id, idx, choice.startswith("👍"))
                        st.session_state.image_feedback_given.add(img_key)
                        st.success("Image feedback saved")

# =========================
# Main
# =========================
def main():
    init_state()
    with st.sidebar:
        sidebar_ui()
    chat_ui()

if __name__ == "__main__":
    main()
