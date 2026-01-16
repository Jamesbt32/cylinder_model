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
    try:
        return os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return os.getenv("OPENAI_API_KEY")

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
        "manual_pdf_name": None,
        "manual_pdf_b64": None,

        "kb_ready": False,

        "question_text": "",
        "question_id": None,

        "retrieved_items": [],
        "answer": None,

        # per-question: prevent “same images from last question”
        "question_diagram_cache": {},  # qid -> list[diagram dict]

        # vote UX (don’t re-ask once voted)
        "image_feedback_given": set(),  # {(chunk_id, img_index)}

        # rebuild toast
        "kb_rebuild_complete": False,
        "kb_last_stats": None,

        # UI controls
        "hide_bad_images": True,
        "hide_bad_images_threshold": 0.25,
        "hide_bad_min_votes": 3,
        "force_diversity_pages": True,
        "max_images_to_show": 12,
        "show_why_overlay": True,
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

def compute_question_id(q: str) -> str:
    qn = re.sub(r"\s+", " ", (q or "").strip().lower())
    return stable_hash(qn or str(time.time()))

def clear_view():
    # clear answer + diagrams for “current view”
    st.session_state.answer = None
    st.session_state.retrieved_items = []
    # keep feedback_given set (so we don’t re-ask for already voted images in same session)
    # remove per-image widget states for radios
    for k in list(st.session_state.keys()):
        if k.startswith("imgfb_"):
            del st.session_state[k]

def on_question_edit():
    # Auto-clear diagrams when user edits question text
    clear_view()

# =========================
# PDF deep-linking
# =========================
def pdf_page_link(page: int) -> Optional[str]:
    b64 = st.session_state.get("manual_pdf_b64")
    if not b64:
        path = st.session_state.get("manual_pdf_path")
        if not path or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            b64 = bytes_to_b64(f.read())
        st.session_state.manual_pdf_b64 = b64
    return f"data:application/pdf;base64,{b64}#page={page}"

# =========================
# Image extraction
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
                if b and not looks_like_blank(b):
                    out.append({"b64": bytes_to_b64(b), "page": page_num, "kind": "embedded"})
            except Exception:
                pass

        # render fallback
        if not out:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                b = pix.tobytes("png")
                if not looks_like_blank(b):
                    out.append({"b64": bytes_to_b64(b), "page": page_num, "kind": "render"})
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
    vecs = []
    for i in range(0, len(texts), 64):
        r = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts[i:i + 64],
        )
        for d in r.data:
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
# Feedback aggregation (global + per-question)
# =========================
def _read_feedback_rows() -> List[Dict[str, Any]]:
    if not os.path.exists(IMAGE_FEEDBACK_FILE):
        return []
    rows = []
    with open(IMAGE_FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def get_image_feedback_stats() -> Tuple[
    Dict[Tuple[str, int], Dict[str, Any]],
    Dict[Tuple[str, int, str], Dict[str, Any]]
]:
    """
    Returns:
      global_stats[(chunk_id, img_idx)] = {pos, neg, votes, score}
      per_q_stats[(chunk_id, img_idx, qid)] = {pos, neg, votes, score}
    Score uses Laplace smoothing: (pos+1)/(votes+2)
    """
    global_stats: Dict[Tuple[str, int], Dict[str, Any]] = {}
    per_q_stats: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    for r in _read_feedback_rows():
        cid = str(r.get("chunk_id", ""))
        idx = r.get("image_index", None)
        helpful = r.get("helpful", None)
        qid = str(r.get("question_id", ""))

        if not cid or idx is None or helpful is None:
            continue

        idx = int(idx)

        gkey = (cid, idx)
        global_stats.setdefault(gkey, {"pos": 0, "neg": 0})
        if helpful:
            global_stats[gkey]["pos"] += 1
        else:
            global_stats[gkey]["neg"] += 1

        if qid:
            qkey = (cid, idx, qid)
            per_q_stats.setdefault(qkey, {"pos": 0, "neg": 0})
            if helpful:
                per_q_stats[qkey]["pos"] += 1
            else:
                per_q_stats[qkey]["neg"] += 1

    def finalize(d: Dict[str, Any]) -> Dict[str, Any]:
        pos = int(d.get("pos", 0))
        neg = int(d.get("neg", 0))
        votes = pos + neg
        score = (pos + 1) / (votes + 2)
        d["votes"] = votes
        d["score"] = float(score)
        return d

    for k in list(global_stats.keys()):
        global_stats[k] = finalize(global_stats[k])

    for k in list(per_q_stats.keys()):
        per_q_stats[k] = finalize(per_q_stats[k])

    return global_stats, per_q_stats

def confidence_weight(votes: int) -> float:
    # ramp confidence by ~8 votes
    return min(1.0, votes / 8.0)

def combined_helpfulness(
    global_score: float, global_votes: int,
    perq_score: float, perq_votes: int
) -> float:
    # Blend per-question score (if present) + global score
    wq = confidence_weight(perq_votes)
    wg = confidence_weight(global_votes)
    if wq + wg == 0:
        return 0.5
    return (perq_score * wq + global_score * wg) / (wq + wg)

# =========================
# Image feedback persistence
# =========================
def save_image_feedback(chunk_id: str, img_index: int, helpful: bool, question_id: Optional[str]):
    entry = {
        "ts": time.time(),
        "chunk_id": chunk_id,
        "image_index": int(img_index),
        "helpful": bool(helpful),
        "question_id": question_id,
    }
    with open(IMAGE_FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

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
        st.error("faiss-cpu not installed.")
        return

    client = get_client()
    reader = PdfReader(pdf)

    items: List[Dict[str, Any]] = []
    img_pages = 0
    img_total = 0

    prog = st.progress(0)
    status = st.empty()

    n_pages = len(reader.pages)
    for i, page in enumerate(reader.pages, start=1):
        status.write(f"Processing page {i}/{n_pages} ...")
        text = (page.extract_text() or "").strip()
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

# =========================
# Retrieval
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

    scores, ids = index.search(qv, k)
    out = []
    for s, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(meta):
            continue
        it = dict(meta[idx])
        it["similarity"] = float(s)
        out.append(it)
    return out

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
        st.session_state.manual_pdf_name = pdf.name
        st.session_state.manual_pdf_b64 = None
        st.success(f"Loaded {pdf.name}")
        # new doc => clear view
        clear_view()

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
    st.subheader("🖼️ Diagram ranking controls")
    st.checkbox("Auto-hide low-score diagrams", key="hide_bad_images")
    st.slider("Hide threshold", 0.0, 1.0, 0.25, 0.05, key="hide_bad_images_threshold")
    st.slider("Min votes before hiding", 1, 10, 3, 1, key="hide_bad_min_votes")
    st.checkbox("Force diversity (no duplicate pages)", key="force_diversity_pages")
    st.checkbox("Explain why each image was shown", key="show_why_overlay")
    st.slider("Max images to show", 1, 50, 12, 1, key="max_images_to_show")

# =========================
# Build diagrams list for current question (sorted by usefulness)
# =========================
def build_diagrams_for_question(qid: str, retrieved_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    global_stats, perq_stats = get_image_feedback_stats()

    diagrams: List[Dict[str, Any]] = []
    for it in retrieved_items:
        cid = str(it.get("chunk_id", ""))
        page_fallback = int(it.get("page", 0) or 0)
        sim = float(it.get("similarity", 0.0))
        images = it.get("images") or []

        for idx, img in enumerate(images):
            page = int(img.get("page", page_fallback) or page_fallback)

            g = global_stats.get((cid, idx), {"score": 0.5, "votes": 0})
            pq = perq_stats.get((cid, idx, qid), {"score": 0.5, "votes": 0})

            gscore, gv = float(g.get("score", 0.5)), int(g.get("votes", 0))
            pqscore, pqv = float(pq.get("score", 0.5)), int(pq.get("votes", 0))

            comb = combined_helpfulness(gscore, gv, pqscore, pqv)
            conf = (confidence_weight(gv) + confidence_weight(pqv)) / 2.0
            # Confidence-weighted ranking: usefulness dominates, similarity tiebreaker
            final_rank = comb * (0.7 + 0.3 * conf) + 0.08 * sim

            diagrams.append({
                "chunk_id": cid,
                "img_index": idx,
                "page": page,
                "img_b64": img.get("b64"),
                "similarity": sim,
                "global_score": gscore,
                "global_votes": gv,
                "perq_score": pqscore,
                "perq_votes": pqv,
                "combined_score": comb,
                "confidence": conf,
                "rank": final_rank,
            })

    diagrams.sort(key=lambda d: float(d.get("rank", 0.0)), reverse=True)
    return diagrams

# =========================
# Chat UI
# =========================
def chat_ui():
    st.title("💬 The Vaillant Brain")

    if not st.session_state.kb_ready:
        st.info("Upload a PDF and rebuild the knowledge base to begin.")
        return

    q = st.text_input("Ask The Vaillant Brain a question", key="question_text", on_change=on_question_edit)

    if st.button("Ask", type="primary") and q:
        if not require_openai_key():
            return

        qid = compute_question_id(q)
        st.session_state.question_id = qid

        ctx = retrieve(q)
        st.session_state.retrieved_items = ctx

        # Build + cache diagrams for this question to prevent “stale diagrams”
        diagrams = build_diagrams_for_question(qid, ctx)
        st.session_state.question_diagram_cache[qid] = diagrams

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

    # =========================
    # Diagrams (sorted by usefulness)
    # =========================
    qid = st.session_state.get("question_id")
    if not qid:
        return

    diagrams = st.session_state.question_diagram_cache.get(qid, [])
    if not diagrams:
        return

    st.markdown("### 📷 Related diagrams (highest usefulness first)")

    pdf_name = st.session_state.get("manual_pdf_name") or (
        os.path.basename(st.session_state.get("manual_pdf_path") or "manual.pdf")
    )

    hide_bad = bool(st.session_state.get("hide_bad_images", True))
    hide_thr = float(st.session_state.get("hide_bad_images_threshold", 0.25))
    hide_min_votes = int(st.session_state.get("hide_bad_min_votes", 3))
    force_div = bool(st.session_state.get("force_diversity_pages", True))
    show_why = bool(st.session_state.get("show_why_overlay", True))
    max_show = int(st.session_state.get("max_images_to_show", 12))

    shown = 0
    seen_pages = set()

    for d in diagrams:
        if shown >= max_show:
            break

        page = int(d.get("page", 0) or 0)
        if force_div and page in seen_pages:
            continue

        combined = float(d.get("combined_score", 0.5))
        gv = int(d.get("global_votes", 0))
        pqv = int(d.get("perq_votes", 0))
        total_votes = gv + pqv

        if hide_bad and total_votes >= hide_min_votes and combined < hide_thr:
            continue

        img_b64 = d.get("img_b64")
        if not img_b64:
            continue

        seen_pages.add(page)
        shown += 1

        img_bytes = b64_to_bytes(img_b64)
        st.image(img_bytes, use_container_width=True)

        # Required label
        st.caption(
            f"**Document:** {pdf_name}  \n"
            f"**Recommended page:** Page {page}  \n"
            f"**Diagram helpfulness:** {combined:.2f} (votes: {total_votes})"
        )

        link = pdf_page_link(page)
        if link:
            st.markdown(f"[Open PDF at page {page}]({link})")

        if show_why:
            with st.expander("Why this image was shown", expanded=False):
                st.write(
                    f"- Retrieval similarity: **{float(d.get('similarity', 0.0)):.3f}**\n"
                    f"- Global score: **{float(d.get('global_score', 0.5)):.2f}** (votes: {gv})\n"
                    f"- Per-question score: **{float(d.get('perq_score', 0.5)):.2f}** (votes: {pqv})\n"
                    f"- Combined helpfulness: **{combined:.2f}**\n"
                    f"- Confidence weight: **{float(d.get('confidence', 0.0)):.2f}**\n"
                    f"- Final rank: **{float(d.get('rank', 0.0)):.3f}**"
                )

        # ✅ usefulness radio button on EVERY image
        img_key = (str(d["chunk_id"]), int(d["img_index"]))
        widget_key = f"imgfb_{d['chunk_id']}_{d['img_index']}_{qid}"  # include qid so it doesn't get “stuck”

        # If already voted this session, show what they chose (read-only)
        if img_key in st.session_state.image_feedback_given:
            st.caption("✅ Feedback recorded for this diagram.")
        else:
            choice = st.radio(
                "Was this image helpful?",
                ["👍 Yes", "👎 No"],
                index=None,
                key=widget_key,
                horizontal=True,
            )
            if choice:
                save_image_feedback(
                    chunk_id=str(d["chunk_id"]),
                    img_index=int(d["img_index"]),
                    helpful=choice.startswith("👍"),
                    question_id=qid,
                )
                st.session_state.image_feedback_given.add(img_key)

                # update the cached scores live by rebuilding diagrams for this qid
                ctx = st.session_state.get("retrieved_items", [])
                st.session_state.question_diagram_cache[qid] = build_diagrams_for_question(qid, ctx)

                st.success("Image feedback saved")
                st.rerun()

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
