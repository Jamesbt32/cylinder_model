# chatbot.py
# Run with: streamlit run chatbot.py

import os
import re
import json
import time
import base64
import hashlib
from typing import List, Dict, Any

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
st.set_page_config(page_title="AI Simulation Assistant", layout="wide")

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
# Session state
# =========================
def init_state():
    defaults = {
        "manual_pdf_path": None,
        "kb_ready": False,
        "retrieved_items": [],
        "answer": None,
        "recommended_page": None,
        "image_feedback_given": set(),  # (chunk_id, img_index)
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

    out = []
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)

        for img in page.get_images(full=True):
            base = doc.extract_image(img[0])
            b = base.get("image")
            if not b or looks_like_blank(b):
                continue
            out.append({
                "b64": bytes_to_b64(b),
                "kind": "embedded",
                "page": page_num,
            })

        if not out:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            b = pix.tobytes("png")
            if not looks_like_blank(b):
                out.append({
                    "b64": bytes_to_b64(b),
                    "kind": "render",
                    "page": page_num,
                })

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
        batch = texts[i:i + 64]
        r = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch,
        )
        for d in r.data:
            vecs.append(np.array(d.embedding, dtype="float32"))
    return np.vstack(vecs)

# =========================
# FAISS
# =========================
@st.cache_resource
def load_faiss():
    if faiss is None or not os.path.exists(KB_INDEX):
        return None, None
    return (
        faiss.read_index(KB_INDEX),
        json.load(open(KB_META, "r", encoding="utf-8")),
    )

# =========================
# KB rebuild
# =========================
def rebuild_kb():
    pdf = st.session_state.manual_pdf_path
    if not pdf:
        st.error("Upload a PDF first.")
        return
    if faiss is None:
        st.error("faiss-cpu not installed")
        return
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY missing")
        return

    client = OpenAI()
    reader = PdfReader(pdf)

    items = []
    prog = st.progress(0)

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < 120:
            continue

        images = extract_page_images(pdf, i)
        for ch in re.split(r"\n{2,}", text):
            items.append({
                "page": i,
                "chunk_id": f"p{i}-{stable_hash(ch)}",
                "text": ch,
                "images": images,
            })

        prog.progress(int(i / len(reader.pages) * 80))

    emb = embed_texts(client, [x["text"] for x in items])
    faiss.normalize_L2(emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, KB_INDEX)
    json.dump(items, open(KB_META, "w", encoding="utf-8"), indent=2)

    st.cache_resource.clear()
    st.session_state.kb_ready = True
    st.success("Knowledge base rebuilt")

# =========================
# Retrieval
# =========================
def retrieve(query: str, k: int = 6):
    index, meta = load_faiss()
    if not index:
        return []

    client = OpenAI()
    r = client.embeddings.create(
        model="text-embedding-3-large",
        input=query,
    )
    q = np.array(r.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)

    scores, ids = index.search(q, k)
    out = []
    for s, i in zip(scores[0], ids[0]):
        if i >= 0:
            m = dict(meta[i])
            m["similarity"] = float(s)
            out.append(m)
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
        st.success(f"Loaded {pdf.name}")

    if st.button("🔁 Rebuild Knowledge Base"):
        rebuild_kb()

# =========================
# Image feedback persistence
# =========================
def save_image_feedback(chunk_id: str, img_index: int, helpful: bool):
    entry = {
        "ts": time.time(),
        "chunk_id": chunk_id,
        "image_index": img_index,
        "helpful": helpful,
    }
    with open(IMAGE_FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

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
        st.session_state.recommended_page = ctx[0]["page"] if ctx else None

        manual_text = "\n\n".join(
            f"[Page {c['page']}]\n{c['text']}" for c in ctx
        )

        client = OpenAI()
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

        st.markdown("### 📷 Related diagrams")

        pdf_name = os.path.basename(st.session_state.manual_pdf_path)

        for it in st.session_state.retrieved_items:
            chunk_id = it["chunk_id"]
            for idx, img in enumerate(it["images"]):
                img_bytes = b64_to_bytes(img["b64"])
                page = img.get("page", it["page"])

                st.image(img_bytes, use_container_width=True)

                # 👇 REQUIRED LABEL
                st.caption(
                    f"**Document:** {pdf_name}  \n"
                    f"**Recommended page:** Page {page}"
                )

                img_key = (chunk_id, idx)
                if img_key not in st.session_state.image_feedback_given:
                    fb = st.radio(
                        "Was this image helpful?",
                        ["👍 Yes", "👎 No"],
                        index=None,
                        key=f"imgfb_{chunk_id}_{idx}",
                        horizontal=True,
                    )
                    if fb:
                        save_image_feedback(chunk_id, idx, fb.startswith("👍"))
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
