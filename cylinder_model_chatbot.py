# --- cylinder_model_multilayer_150L.py ---
# Vaillant 150 L stratified cylinder + modulating HP model
# Geometry-aware, with graphs, 3D PyVista viz, and optional AI assistant
# ENHANCED: Spinning wheel on button click + Typewriter effect for AI responses

import os
import json
import base64
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from typing import List, Dict, Any
from PyPDF2 import PdfReader
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except Exception as e:
    FITZ_AVAILABLE = False
    print("⚠️ PyMuPDF not available:", e)
import faiss
from openai import OpenAI
from difflib import SequenceMatcher
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import time

alt.data_transformers.disable_max_rows()


st.set_page_config(
    page_title="Vaillant 150 L Cylinder Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================
# 📘 KNOWLEDGE BASE (AUTO-REBUILD) — FULL DROP-IN CODE
# Adds:
# ⚡ Batch embedding (faster)
# 🧠 Chunk overlap + heading detection
# 📊 Per-page rebuild progress bar
# 🔍 Explain which PDF pages were used
# 🧪 Self-test after rebuild
# ==============================================================

import os, re, json, time
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple, Optional, Any

from PyPDF2 import PdfReader
import faiss
from openai import OpenAI

# If you have fitz available already, keep your FITZ_AVAILABLE logic.
# This block assumes you already created `client` earlier as OpenAI(api_key=API_KEY).
# If not, it will create a local client inside the rebuild function from env/secrets.

# -----------------------------------------------------
# ✅ Session-state bootstrap (keep near top of file)
# -----------------------------------------------------
if "kb_version" not in st.session_state:
    st.session_state.kb_version = 0
if "kb_checked" not in st.session_state:
    st.session_state.kb_checked = False
if "pdf_mtime" not in st.session_state:
    st.session_state.pdf_mtime = None
if "kb_last_build_report" not in st.session_state:
    st.session_state.kb_last_build_report = {}
if "kb_last_selftest" not in st.session_state:
    st.session_state.kb_last_selftest = {}


# -----------------------------------------------------
# 🧠 Chunking helpers: overlap + heading detection
# -----------------------------------------------------
_HEADING_RE = re.compile(
    r"^\s*(?:[A-Z0-9][A-Z0-9 \-–—/]{6,}|[0-9]+\.[0-9.]*\s+\S+.+|.+:)\s*$"
)

def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def detect_headings(text: str) -> List[Tuple[int, str]]:
    """
    Returns list of (line_index, heading_text) for lines that look like headings.
    Simple heuristic: ALL CAPS / numbered headings / lines ending with ':'.
    """
    headings = []
    lines = [ln.strip() for ln in text.splitlines()]
    for i, ln in enumerate(lines):
        if not ln:
            continue
        if _HEADING_RE.match(ln):
            headings.append((i, ln[:120]))
    return headings

def chunk_text_with_overlap(
    text: str,
    max_words: int = 450,
    overlap_words: int = 80,
) -> List[str]:
    """
    Word-based chunking with overlap.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    n = len(words)

    while start < n:
        end = min(start + max_words, n)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap_words)

    return chunks

def chunk_page_text(
    page_text: str,
    max_words: int = 450,
    overlap_words: int = 80
) -> List[Dict[str, Any]]:
    """
    Produces chunk dicts including detected heading context.
    """
    page_text = _clean_text(page_text)
    if len(page_text) < 40:
        return []

    lines = [ln.strip() for ln in page_text.splitlines()]
    headings = detect_headings(page_text)

    # Build a "heading context" per line: last heading above that line
    last_heading_for_line = {}
    last = None
    h_idx = 0
    for i in range(len(lines)):
        while h_idx < len(headings) and headings[h_idx][0] == i:
            last = headings[h_idx][1]
            h_idx += 1
        last_heading_for_line[i] = last

    # Re-join with line breaks for approximate line grouping
    joined = "\n".join(lines)

    # Chunk at word-level
    raw_chunks = chunk_text_with_overlap(joined, max_words=max_words, overlap_words=overlap_words)

    # Assign a best-effort heading context: find the first line that appears in chunk
    chunk_dicts = []
    for ch in raw_chunks:
        # naive: use first non-empty line in chunk for heading lookup
        ch_lines = [l.strip() for l in ch.splitlines() if l.strip()]
        heading_ctx = None
        if ch_lines:
            first_line = ch_lines[0]
            try:
                # Find its index in lines
                li = lines.index(first_line)
                heading_ctx = last_heading_for_line.get(li)
            except ValueError:
                heading_ctx = None

        chunk_dicts.append({
            "text": ch,
            "heading": heading_ctx
        })

    return chunk_dicts


# -----------------------------------------------------
# ⚡ Batched embeddings helper
# -----------------------------------------------------
def embed_texts_batched(
    client_local: OpenAI,
    texts: List[str],
    model: str = "text-embedding-3-large",
    batch_size: int = 96,
    sleep_s: float = 0.0
) -> np.ndarray:
    """
    Returns embeddings matrix float32 [N, D].
    Uses OpenAI embeddings endpoint with batched input.
    """
    vectors: List[np.ndarray] = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        resp = client_local.embeddings.create(model=model, input=batch)
        # Keep order stable
        embs = [np.array(d.embedding, dtype="float32") for d in resp.data]
        vectors.extend(embs)
        if sleep_s > 0:
            time.sleep(sleep_s)
    mat = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype="float32")
    return mat


# -----------------------------------------------------
# 📂 FAISS loader (versioned cache)
# -----------------------------------------------------
@st.cache_resource
def load_faiss_index(kb_version: int) -> Tuple[Optional[faiss.Index], Optional[List[Dict[str, Any]]]]:
    kb_dir = os.path.join(os.path.dirname(__file__), "kb")
    idx_path = os.path.join(kb_dir, "vaillant_joint_faiss.index")
    meta_path = os.path.join(kb_dir, "vaillant_joint_meta.json")

    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        return None, None

    try:
        index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    except Exception as e:
        st.error(f"❌ Failed to load FAISS: {e}")
        return None, None


# -----------------------------------------------------
# 🔍 Retrieval (uses versioned loader)
# -----------------------------------------------------
def retrieve_faiss_context(query: str, top_k: int = 3, show_debug: bool = False):
    index, meta = load_faiss_index(st.session_state.kb_version)
    if not index or not meta:
        if show_debug:
            st.warning("⚠️ FAISS index not loaded.")
        return []

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        if show_debug:
            st.error("⚠️ Missing OPENAI_API_KEY")
        return []

    try:
        client_local = OpenAI(api_key=api_key)
        emb_resp = client_local.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        emb_vec = np.array(emb_resp.data[0].embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(emb_vec)

        scores, ids = index.search(emb_vec, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if 0 <= idx < len(meta):
                item = dict(meta[idx])
                item["similarity"] = float(score)
                results.append(item)

        if show_debug:
            st.write(f"Top-{len(results)} results:")
            for r in results:
                st.write(
                    f"- p{r.get('page')} sim={r.get('similarity'):.3f} "
                    f"heading={r.get('heading') or '—'}"
                )

        return results
    except Exception as e:
        if show_debug:
            st.error(f"⚠️ Retrieval error: {e}")
        return []


# -----------------------------------------------------
# 📊 🧪 Self-test after rebuild
# -----------------------------------------------------
def kb_self_test(sample_queries: List[str] = None, top_k: int = 3) -> Dict[str, Any]:
    if sample_queries is None:
        sample_queries = [
            "heat pump operation",
            "legionella cycle",
            "installation diagram",
            "temperature sensor",
        ]

    report: Dict[str, Any] = {
        "ok": False,
        "checks": [],
        "samples": []
    }

    index, meta = load_faiss_index(st.session_state.kb_version)
    if not index or not meta:
        report["checks"].append({"name": "index_loaded", "ok": False, "detail": "Index/meta missing"})
        return report

    report["checks"].append({"name": "index_loaded", "ok": True, "detail": f"ntotal={index.ntotal}, dim={index.d}"})
    report["checks"].append({"name": "meta_nonempty", "ok": len(meta) > 0, "detail": f"chunks={len(meta)}"})

    # run a few queries and verify results exist
    any_hits = False
    for q in sample_queries:
        res = retrieve_faiss_context(q, top_k=top_k, show_debug=False)
        ok = len(res) > 0
        any_hits = any_hits or ok
        report["samples"].append({
            "query": q,
            "ok": ok,
            "top_pages": [r.get("page") for r in res[:top_k]],
            "top_headings": [r.get("heading") for r in res[:top_k]],
        })

    report["checks"].append({"name": "retrieval_hits", "ok": any_hits, "detail": "At least one query returned results"})
    report["ok"] = all(c["ok"] for c in report["checks"])
    return report


# -----------------------------------------------------
# 🔁 Rebuild Knowledge Base (PDF → chunks → embeddings → FAISS)
# SAFE VERSION: accepts pdf_filename OR pdf_name
# -----------------------------------------------------
def rebuild_knowledge_base(
    pdf_filename: str = None,
    pdf_name: str = None,
    max_words: int = 450,
    overlap_words: int = 80,
    embedding_model: str = "text-embedding-3-large",
    embedding_batch_size: int = 96,
    min_text_chars: int = 80,
) -> None:
    """
    Rebuilds the FAISS knowledge base from a PDF.
    SAFETY: accepts both pdf_filename and pdf_name to avoid caller mismatch bugs.
    """

    # -------------------------------
    # ✅ SAFETY RESOLUTION
    # -------------------------------
    pdf_filename = pdf_filename or pdf_name or "8000014609_03.pdf"

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error("❌ Missing OPENAI_API_KEY. Cannot rebuild KB.")
        return

    client_local = OpenAI(api_key=api_key)

    base_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(base_dir, pdf_filename)

    if not os.path.exists(pdf_path):
        st.error(f"❌ PDF not found: {pdf_path}")
        return

    kb_dir = os.path.join(base_dir, "kb")
    os.makedirs(kb_dir, exist_ok=True)

    idx_out = os.path.join(kb_dir, "vaillant_joint_faiss.index")
    meta_out = os.path.join(kb_dir, "vaillant_joint_meta.json")

    # -------------------------------
    # 📘 Open PDF
    # -------------------------------
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        st.error(f"❌ Could not open PDF: {e}")
        return

    n_pages = len(reader.pages)
    items = []
    pages_used = []

    progress = st.progress(0)
    status = st.empty()

    # -------------------------------
    # 🧠 Extract + chunk text
    # -------------------------------
    for pnum in range(1, n_pages + 1):
        status.write(f"📝 Processing page {pnum}/{n_pages}...")
        try:
            page = reader.pages[pnum - 1]
            text = page.extract_text() or ""
            text = _clean_text(text)

            if len(text) < min_text_chars:
                progress.progress(int((pnum / n_pages) * 40))
                continue

            pages_used.append(pnum)

            chunk_dicts = chunk_page_text(
                text,
                max_words=max_words,
                overlap_words=overlap_words,
            )

            for ch in chunk_dicts:
                items.append({
                    "page": pnum,
                    "heading": ch.get("heading"),
                    "text": ch["text"],
                    "image_paths": [],        # images added later
                    "figure_captions": []     # ready for caption support
                })

        except Exception as e:
            st.warning(f"⚠️ Page {pnum} failed: {e}")

        progress.progress(int((pnum / n_pages) * 40))

    if not items:
        st.error("❌ No usable text extracted. KB rebuild aborted.")
        return
    
    

    # -------------------------------
    # ⚡ Embeddings (batched)
    # -------------------------------
    status.write(f"🔢 Embedding {len(items)} chunks...")
    texts = []

    for it in items:
        if it.get("heading"):
            texts.append(f"Heading: {it['heading']}\n\n{it['text']}")
        else:
            texts.append(it["text"])

    try:
        mat = embed_texts_batched(
            client_local=client_local,
            texts=texts,
            model=embedding_model,
            batch_size=embedding_batch_size,
        )
    except Exception as e:
        st.error(f"❌ Embedding failed: {e}")
        return

    if mat.size == 0:
        st.error("❌ Embedding matrix empty.")
        return

    faiss.normalize_L2(mat)

    # -------------------------------
    # 📦 Build FAISS index
    # -------------------------------
    status.write("📦 Building FAISS index...")
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    progress.progress(95)

    faiss.write_index(index, idx_out)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    progress.progress(100)

    # -------------------------------
    # 📊 Build report
    # -------------------------------
    st.session_state.kb_last_build_report = {
        "pdf": pdf_filename,
        "pages_total": n_pages,
        "pages_used": pages_used,
        "chunks_total": len(items),
        "embedding_model": embedding_model,
        "embedding_dim": int(mat.shape[1]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    st.success("✅ Knowledge base rebuild complete.")



# -----------------------------------------------------
# 🔁 Automatic rebuild on page open (safe)
# -----------------------------------------------------
def auto_rebuild_kb_on_open(
    pdf_filename: str = "8000014609_03.pdf",
) -> None:
    base_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(base_dir, pdf_filename)

    kb_dir = os.path.join(base_dir, "kb")
    idx_path = os.path.join(kb_dir, "vaillant_joint_faiss.index")
    meta_path = os.path.join(kb_dir, "vaillant_joint_meta.json")

    if not os.path.exists(pdf_path):
        st.warning("⚠️ Manual PDF not found – skipping KB rebuild.")
        return

    pdf_mtime = os.path.getmtime(pdf_path)
    kb_exists = os.path.exists(idx_path) and os.path.exists(meta_path)

    needs_rebuild = (not kb_exists) or (st.session_state.pdf_mtime != pdf_mtime)

    if needs_rebuild:
        with st.spinner("📘 Building knowledge base (auto)..."):
            rebuild_knowledge_base(pdf_filename=pdf_filename)

        st.session_state.pdf_mtime = pdf_mtime
        st.session_state.kb_version += 1
        st.cache_resource.clear()

        st.success("✅ Knowledge base rebuilt automatically.")
        st.rerun()



# -----------------------------------------------------
# 🔍 UI helper: show build report + self-test
# (Call this anywhere you want; sidebar is a good place.)
# -----------------------------------------------------
def show_kb_diagnostics_ui():
    report = st.session_state.get("kb_last_build_report", {})
    test = st.session_state.get("kb_last_selftest", {})

    with st.expander("📘 Knowledge Base Diagnostics", expanded=False):
        st.write(f"KB Version: **{st.session_state.kb_version}**")

        if report:
            st.markdown("### 🔍 Pages used")
            pages_used = report.get("pages_used", [])
            st.write(f"- PDF pages total: **{report.get('pdf_pages_total')}**")
            st.write(f"- Pages used (text extracted): **{len(pages_used)}**")
            st.write(f"- Page list: {pages_used[:80]}{' ...' if len(pages_used) > 80 else ''}")

            st.markdown("### 🧠 Chunking")
            ch = report.get("chunking", {})
            st.write(f"- max_words: **{ch.get('max_words')}**")
            st.write(f"- overlap_words: **{ch.get('overlap_words')}**")
            st.write(f"- chunks_total: **{report.get('chunks_total')}**")

            st.markdown("### ⚡ Embeddings")
            st.write(f"- model: **{report.get('embedding_model')}**")
            st.write(f"- dim: **{report.get('embedding_dim')}**")
            st.write(f"- built: **{report.get('timestamp')}**")
        else:
            st.info("No build report yet (KB may already exist and not have rebuilt this session).")

        st.markdown("### 🧪 Self-test")
        if test:
            st.write(f"Overall: {'✅ PASS' if test.get('ok') else '❌ FAIL'}")
            for c in test.get("checks", []):
                st.write(f"- {c['name']}: {'✅' if c['ok'] else '❌'} — {c.get('detail','')}")
            st.markdown("#### Sample queries")
            for s in test.get("samples", []):
                st.write(
                    f"- **{s['query']}**: {'✅' if s['ok'] else '❌'} "
                    f"(pages: {s.get('top_pages')})"
                )
        else:
            st.info("Self-test not run yet.")





# --- Logo ---
try:
    with open("vaillant_logo.png", "rb") as file:
        data = base64.b64encode(file.read()).decode("utf-8")
    st.markdown(
        f"""
        <div style="text-align: center; padding-top: 10px; padding-bottom: 10px;">
            <img src="data:image/png;base64,{data}" width="200">
            <h1 style="margin-top: 10px;">Vaillant 150 L Cylinder Model Simulation</h1>
        </div>
        <hr style="border:1px solid #ccc;">
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.title("Vaillant 150 L Cylinder Model Simulation")

# ---------- PHYSICAL CONSTANTS ----------
RHO = 1000.0
C_P = 4186.0
P_EL_MAX = 5000.0

# Layer geometry
BASE_LAYER_PROPERTIES: Dict[str, Dict[str, float]] = {
    "Bottom Layer": {
        "Volume_L": 31.54,
        "Water_Layer_Surface_mm2": 86529,
        "Water_External_Surface_mm2": 476155,
    },
    "Lower-Mid Layer": {
        "Volume_L": 37.82,
        "Water_Layer_Surface_mm2": 86529,
        "Water_External_Surface_mm2": 428639,
    },
    "Upper-Mid Layer": {
        "Volume_L": 40.66,
        "Water_Layer_Surface_mm2": 112221,
        "Water_External_Surface_mm2": 430239,
    },
    "Top Layer": {
        "Volume_L": 37.36,
        "Water_Layer_Surface_mm2": 112221,
        "Water_External_Surface_mm2": 476155,
    },
}

# ---------- HEAT PUMP MODEL ----------
def solve_hp(T_tank: float, mod: float, Pmax_W: float, Pmin_W: float, Pmin_ratio: float):
    """Simple modulating HP model with COP depending on tank temperature."""
    if mod < Pmin_ratio:
        Pel = 0.0
    else:
        frac = (mod - Pmin_ratio) / (1 - Pmin_ratio)
        Pel = Pmin_W + (Pmax_W - Pmin_W) * frac
    COP = max(4.0 - 0.1 * (T_tank - 45.0), 1.0)
    Qhp = Pel * COP
    return Qhp, Pel, COP

# ---------- SIMULATION ----------
print(">>> run_sim started")

buffer_layers = None
buffer_history = []

@st.cache_data
def run_sim(

    
    dt_min: float,
    sim_hrs: float,
    Utank: float,
    Tamb: float,
    setp: float,
    hyst: float,
    Pmin: float,
    Pmax: float,
    Tsrc: float,
    tap: Dict[str, List[float]],
    initial_tank_temp: float,
    legionella_enabled: bool = False,
    legionella_temp: float = 65.0,
    legionella_duration: float = 30.0,
    legionella_frequency: float = 7.0,
    legionella_start_hour: int = 2,
    senso_enabled: bool = False,
    comfort_mode: str = "Comfort",
    eco_temp: float = 45.0,
    comfort_temp: float = 55.0,
    enable_time_program: bool = False,
    morning_start: int = 6,
    morning_end: int = 9,
    evening_start: int = 17,
    evening_end: int = 22,
    boost_mode: bool = False,
    holiday_mode: bool = False,
    holiday_temp: float = 40.0,
    adaptive_timestep: bool = True,
    buffer_enabled: bool = False,
 

):
    """4-layer stratified tank simulation with adaptive time stepping, improved tapping, and enhanced mixing."""
    # --------------------------
    # Validate tap dict
    # --------------------------
    for key in ["time", "volume", "rate_lpm"]:
        if key not in tap:
            raise ValueError(f"tap dict missing '{key}'")

    tap_time = np.array(tap["time"])
    tap_vol = np.array(tap["volume"])
    tap_rate = np.array(tap["rate_lpm"])

    N_layers = 4

    # ===============================
    # BUFFER STRATIFICATION SETUP ✅
    # ===============================
    N_buf = 4
    BUFFER_RHO = 1000.0
    BUFFER_CP = 4186.0

    buffer_enabled_flag = "On" if buffer_enabled else "Off"
    buffer_volume_L = st.session_state.get("buffer_volume_L", 0)
    buffer_temp_init = st.session_state.get("buffer_temp_init", 45)
    immersion_enabled = st.session_state.get("immersion_enabled", False)
    immersion_power_kw = st.session_state.get("immersion_power_kw", 0.0)
    boost_cylinder = st.session_state.get("boost_cylinder", False)
    pump_flow_lpm = st.session_state.get("pump_flow_lpm", 0)

    buffer_layers = None
    buffer_history: List[np.ndarray] = []

    # ===============================
    # ADAPTIVE TIME STEP SETUP ✅
    # ===============================
    if adaptive_timestep:
        dt_base = dt_min * 60.0        # [s]
        dt_tap = max(dt_base / 3, 60.0)
    else:
        dt_base = dt_min * 60.0
        dt_tap = dt_base

    # Pre-allocate safe max step count
    max_steps = int(sim_hrs * 3600.0 / dt_min) * 2

    if buffer_enabled_flag == "On":
        buffer_layers = np.zeros((max_steps, N_buf), dtype=float)
        buffer_layers[0, :] = buffer_temp_init

    # ===============================
    # DYNAMIC ARRAY INITIALIZATION ✅
    # ===============================
    time_array = []
    T_array = []
    Qhp_array = []
    Pel_array = []
    COPs_array = []
    HP_on_array = []
    Qloss_total_array = []
    Tap_flow_array = []
    Tap_temp_array = []
    Mod_frac_array = []
    Legionella_active_array = []
    Senso_mode_array = []
    Senso_setpoint_array = []
    Tap_active_array = []

    # ===============================
    # GEOMETRY SETUP ✅
    # ===============================
    def mm2_to_m2(x):
        return x / 1e6

    layer_order = ["Bottom Layer", "Lower-Mid Layer", "Upper-Mid Layer", "Top Layer"]

    V_layers_m3 = np.array(
        [BASE_LAYER_PROPERTIES[n]["Volume_L"] / 1000.0 for n in layer_order]
    )
    A_ext = np.array(
        [mm2_to_m2(BASE_LAYER_PROPERTIES[n]["Water_External_Surface_mm2"]) for n in layer_order]
    )
    A_int = np.array(
        [mm2_to_m2(BASE_LAYER_PROPERTIES[n]["Water_Layer_Surface_mm2"]) for n in layer_order]
    )

    m_layer = RHO * V_layers_m3
    UA_loss_layer = Utank * A_ext
    U_int = 50.0
    UA_int_layer = U_int * A_int

    Pmax_W = P_EL_MAX * Pmax
    Pmin_W = Pmax_W * Pmin
    coil_bottom_idx = 2

    # Initial temperatures
    T_prev = np.full(N_layers, initial_tank_temp, dtype=float)

    # Legionella cycle tracking
    legionella_cycle_start_time = None
    legionella_in_progress = False

    # Simulation time tracking
    tnow_s = 0.0
    sim_end_s = sim_hrs * 3600.0
    step_count = 0

    # ============================================================
    # MAIN SIMULATION LOOP WITH ADAPTIVE TIME STEPPING
    # ============================================================
    while tnow_s < sim_end_s and step_count < max_steps:
        tnow_min = tnow_s / 60.0
        tnow_hrs = tnow_s / 3600.0

        # Check for tapping in next base time step
        tap_check_end = tnow_min + (dt_base / 60.0)
        idx_check = np.where((tap_time >= tnow_min) & (tap_time < tap_check_end))[0]
        is_tapping = len(idx_check) > 0

        # Select adaptive time step
        if adaptive_timestep and is_tapping:
            dt_s = dt_tap
        else:
            dt_s = dt_base

        dt_current_min = dt_s / 60.0

        T_new = T_prev.copy()
        T_top = T_prev[-1]
        T_bottom = T_prev[0]

        # ============================================================
        # 🎛️ SENSOCOMFORT CONTROL LOGIC
        # ============================================================
        active_setpoint = setp
        current_mode = "Standard"

        if senso_enabled:
            hour_of_day = tnow_hrs % 24.0

            if holiday_mode:
                active_setpoint = holiday_temp
                current_mode = "Holiday"
            elif boost_mode:
                active_setpoint = min(comfort_temp + 5, 65)
                current_mode = "Boost"
            elif enable_time_program:
                in_morning = morning_start <= hour_of_day < morning_end
                in_evening = evening_start <= hour_of_day < evening_end

                if in_morning or in_evening:
                    active_setpoint = comfort_temp
                    current_mode = "Comfort"
                else:
                    active_setpoint = eco_temp
                    current_mode = "ECO"
            elif comfort_mode == "ECO":
                active_setpoint = eco_temp
                current_mode = "ECO"
            elif comfort_mode == "Comfort":
                active_setpoint = comfort_temp
                current_mode = "Comfort"
            elif comfort_mode == "Auto":
                in_morning = morning_start <= hour_of_day < morning_end
                in_evening = evening_start <= hour_of_day < evening_end

                if in_morning or in_evening:
                    active_setpoint = comfort_temp
                    current_mode = "Auto-Comfort"
                else:
                    active_setpoint = eco_temp
                    current_mode = "Auto-ECO"

        # ============================================================
        # 🦠 LEGIONELLA CYCLE LOGIC (overrides SensoComfort)
        # ============================================================
        legionella_override = False

        if legionella_enabled:
            current_day = int(tnow_hrs / 24.0)
            hour_of_day = tnow_hrs % 24.0

            if (
                current_day % legionella_frequency == 0
                and legionella_start_hour <= hour_of_day < legionella_start_hour + 0.1
                and not legionella_in_progress
            ):
                legionella_cycle_start_time = tnow_min
                legionella_in_progress = True

            if legionella_in_progress:
                elapsed = tnow_min - legionella_cycle_start_time
                if elapsed < legionella_duration:
                    legionella_override = True
                else:
                    legionella_in_progress = False

        # ============================================================
        # HEAT PUMP CONTROL
        # ============================================================
        mod = 0.0
        on = False

        target_temp = legionella_temp if legionella_override else active_setpoint
        target_hyst = 5.0 if legionella_override else hyst
        Ton = target_temp - target_hyst

        boost_factor = 1.5 if (boost_mode and senso_enabled and not legionella_override) else 1.0

        if legionella_override:
            if T_top < target_temp:
                on = True
                if T_top <= target_temp - 5.0:
                    mod = 1.0
                else:
                    raw = (target_temp - T_top) / 5.0
                    mod = max(Pmin, np.clip(raw, 0.0, 1.0))
        else:
            if T_top < target_temp:
                on = True
                if T_top <= Ton:
                    mod = min(1.0, 1.0 * boost_factor)
                else:
                    raw = (target_temp - T_top) / target_hyst
                    mod = Pmin + (1 - Pmin) * np.clip(raw, 0.0, 1.0)
                    if boost_factor > 1.0:
                        mod = min(1.0, mod * boost_factor)

        if mod < Pmin:
            on = False
            mod = 0.0

        Q = Pe = COP = 0.0
        if on:
            Q, Pe, COP = solve_hp(T_bottom, mod, Pmax_W, Pmin_W, Pmin)

        # ============================================================
        # IMPROVED TAPPING WITH INSTANTANEOUS FLOW RATE
        # ============================================================
        idx = np.where((tap_time >= tnow_min) & (tap_time < tnow_min + dt_current_min))[0]
        Vtap_L = np.sum(tap_vol[idx]) if idx.size > 0 else 0.0
        tap_rate_current = np.sum(tap_rate[idx]) if idx.size > 0 else 0.0

        tap_duration_s = 0.0
        if len(idx) > 0:
            for i_idx in idx:
                duration = (tap_vol[i_idx] / tap_rate[i_idx]) * 60.0
                tap_duration_s += duration

        if tap_duration_s > 0:
            mdot_tap = (Vtap_L / 1000.0) * RHO / tap_duration_s
        else:
            mdot_tap = 0.0

        tap_active = tap_duration_s > 0

        # ============================================================
        # ENERGY BALANCE FOR EACH LAYER
        # ============================================================
        Q_loss_total = 0.0

        for i in range(N_layers):
            Q_net = 0.0

            if i > 0:
                Q_net += UA_int_layer[i] * (T_prev[i - 1] - T_prev[i])
            if i < N_layers - 1:
                Q_net += UA_int_layer[i] * (T_prev[i + 1] - T_prev[i])

            Q_loss = UA_loss_layer[i] * (T_prev[i] - Tamb)
            Q_net -= Q_loss
            Q_loss_total += Q_loss

            if i < coil_bottom_idx and Q > 0:
                Q_net += Q / coil_bottom_idx

            if i == N_layers - 1 and tap_duration_s > 0:
                Q_tap_out = mdot_tap * C_P * T_prev[i]
                tap_fraction = min(tap_duration_s / dt_s, 1.0)
                Q_net -= Q_tap_out * tap_fraction
            elif i == 0 and tap_duration_s > 0:
                Q_tap_in = mdot_tap * C_P * Tamb
                tap_fraction = min(tap_duration_s / dt_s, 1.0)
                Q_net += Q_tap_in * tap_fraction

            dT = Q_net * dt_s / (m_layer[i] * C_P)
            T_new[i] = np.clip(T_prev[i] + dT, 0.0, 100.0)

        # ============================================================
        # ✅ IMPROVED ANTI-DESTRATIFICATION (GRADUAL MIXING 80/20)
        # ============================================================
        for i in range(N_layers - 1):
            if T_new[i] > T_new[i + 1]:
                T_upper = T_new[i + 1]
                T_lower = T_new[i]
                T_new[i] = 0.8 * T_lower + 0.2 * T_upper
                T_new[i + 1] = 0.2 * T_lower + 0.8 * T_upper



                # ============================
        # STORE STEP RESULTS
        # ============================
        time_array.append(tnow_hrs)
        T_array.append(T_new.copy())
        Qhp_array.append(Q)
        Pel_array.append(Pe)
        COPs_array.append(COP)
        HP_on_array.append(on)
        Qloss_total_array.append(Q_loss_total)
        Tap_flow_array.append(tap_rate_current)
        Tap_temp_array.append(T_new[-1])
        Mod_frac_array.append(mod)
        Legionella_active_array.append(legionella_override)
        Senso_mode_array.append(current_mode)
        Senso_setpoint_array.append(active_setpoint)
        Tap_active_array.append(tap_active)

# ===============================
# ✅ BUFFER UPDATE - ONE PER STEP
# ===============================
        if buffer_enabled_flag == "On" and buffer_layers is not None:
            if step_count == 0:
        # Save initial state
             buffer_history.append(buffer_layers[0].copy())
            elif step_count > 0:
                buffer_mass = (buffer_volume_L / 1000.0) * BUFFER_RHO

                for l in range(N_buf):
                    Qnet_buf = 0.0

            # Conduction between layers - read from PREVIOUS timestep
                    if l > 0:
                        Qnet_buf += 500 * (buffer_layers[step_count-1, l-1] - buffer_layers[step_count-1, l])
                    if l < N_buf - 1:
                        Qnet_buf += 500 * (buffer_layers[step_count-1, l+1] - buffer_layers[step_count-1, l])

            # Immersion heater in bottom layer
                    if l == 0 and immersion_enabled:
                        Qnet_buf += immersion_power_kw * 1000

            # Boost pump extraction from top layer
                    if boost_cylinder and l == N_buf - 1:
                        mdot = (pump_flow_lpm / 60) / 1000 * BUFFER_RHO
                        Qnet_buf -= mdot * BUFFER_CP * (buffer_layers[step_count-1, -1] - T_prev[0])

            # Calculate new temperature FROM PREVIOUS TIMESTEP
                    new_temp = buffer_layers[step_count-1, l] + (
                        Qnet_buf * dt_s / ((buffer_mass / N_buf) * BUFFER_CP)
            )

            # Store in current timestep
                    buffer_layers[step_count, l] = np.clip(new_temp, 10, 90)

                buffer_history.append(buffer_layers[step_count].copy())

# Advance time
        T_prev = T_new
        tnow_s += dt_s
        step_count += 1

 
    # ============================================================
    # CONVERT DYNAMIC ARRAYS TO DATAFRAME
    # ============================================================
    time_h = np.array(time_array)
    T_matrix = np.array(T_array)

    df = pd.DataFrame(
        {
            "Time (h)": time_h,
            "HP Power (W)": Pel_array,
            "HP Heat (W)": Qhp_array,
            "COP": COPs_array,
            "HP_On": HP_on_array,
            "Modulation": Mod_frac_array,
            "Tap Flow (L/min)": Tap_flow_array,
            "Tap Temp (°C)": Tap_temp_array,
            "Q_Loss (W)": Qloss_total_array,
            "Legionella_Active": Legionella_active_array,
            "Senso_Mode": Senso_mode_array,
            "Senso_Setpoint (°C)": Senso_setpoint_array,
            "Tap_Active": Tap_active_array,
        }
    )

    if len(T_matrix) > 0:
        df["T_Bottom Layer (°C)"] = T_matrix[:, 0]
        df["T_Lower-Mid Layer (°C)"] = T_matrix[:, 1]
        df["T_Upper-Mid Layer (°C)"] = T_matrix[:, 2]
        df["T_Top Layer (°C)"] = T_matrix[:, 3]
        df["T_Avg (°C)"] = T_matrix.mean(axis=1)
    else:
        df["T_Bottom Layer (°C)"] = []
        df["T_Lower-Mid Layer (°C)"] = []
        df["T_Upper-Mid Layer (°C)"] = []
        df["T_Top Layer (°C)"] = []
        df["T_Avg (°C)"] = []

            # ------------------------------------------
    # ✅ BUFFER RESULTS → DATAFRAME (if enabled)
    # ------------------------------------------
    if buffer_enabled_flag == "On" and buffer_layers is not None and len(time_array) > 0:
        buf_steps = len(time_array)  # same number of time points as main tank
        buf_slice = buffer_layers[:buf_steps, :]  # [steps, N_buf]

        # 4 buffer layer temperature series
        for i in range(N_buf):
            df[f"Buffer Layer {i+1} (°C)"] = buf_slice[:, i]

        # Average buffer temperature
        buffer_mass = (buffer_volume_L / 1000.0) * BUFFER_RHO  # [kg]
        buf_avg = buf_slice.mean(axis=1)  # [°C]
        df["Buffer Avg Temp (°C)"] = buf_avg

        # Instantaneous buffer energy content vs 0 °C [kWh]
        df["Buffer Energy (kWh vs 0°C)"] = (
            buffer_mass * BUFFER_CP * buf_avg / 3.6e6
        )

        # Buffer pump flow (constant if boost pump enabled)
        if boost_cylinder and pump_flow_lpm > 0:
            df["Buffer Pump Flow (L/min)"] = float(pump_flow_lpm)
        else:
            df["Buffer Pump Flow (L/min)"] = 0.0


 

    # ===============================
    # ENERGY INTEGRATION
    # ===============================
    if len(df) > 1:
        df["dt_hours"] = df["Time (h)"].diff().fillna(
            df["Time (h)"].iloc[1] - df["Time (h)"].iloc[0]
        )
    else:
        df["dt_hours"] = 0.0

    total_heat_kWh = (df["HP Heat (W)"] * df["dt_hours"]).sum() / 1000.0
    total_power_kWh = (df["HP Power (W)"] * df["dt_hours"]).sum() / 1000.0
    total_losses_kWh = (df["Q_Loss (W)"] * df["dt_hours"]).sum() / 1000.0
    hp_runtime_min = (df["HP_On"].astype(float) * df["dt_hours"] * 60.0).sum()
    avg_cop = total_heat_kWh / total_power_kWh if total_power_kWh > 0 else 0.0
    legionella_runtime_min = (
        (df["Legionella_Active"].astype(float) * df["dt_hours"] * 60.0).sum()
        if legionella_enabled
        else 0.0
    )

    summary = {
        "Simulation Hours": sim_hrs,
        "Tank Volume (L)": 150.0,
        "Total HP Electrical Energy (kWh)": total_power_kWh,
        "Total Heat Delivered by HP (kWh)": total_heat_kWh,
        "Total Losses (kWh)": total_losses_kWh,
        "HP Run Time (minutes)": hp_runtime_min,
        "Average COP": avg_cop,
        "Legionella Cycle Runtime (minutes)": legionella_runtime_min,
    }

        # Optional: buffer energy change over simulation
    if buffer_enabled_flag == "On" and buffer_layers is not None and len(time_array) > 1:
        buf_steps = len(time_array)
        buf_slice = buffer_layers[:buf_steps, :]
        buffer_mass = (buffer_volume_L / 1000.0) * BUFFER_RHO
        buf_avg = buf_slice.mean(axis=1)
        buf_energy = buffer_mass * BUFFER_CP * buf_avg / 3.6e6  # [kWh vs 0°C]

        buffer_energy_change = buf_energy[-1] - buf_energy[0]
        summary["Buffer Energy Change (kWh)"] = buffer_energy_change
        summary["Buffer Start Temp (°C)"] = float(buf_avg[0])
        summary["Buffer End Temp (°C)"] = float(buf_avg[-1])


    return df, summary, BASE_LAYER_PROPERTIES



# ---------- CHARTS ----------

# GLOBAL shared selection for synced hover across all charts
global_hover = alt.selection_point(
    fields=["Time (h)"],
    nearest=True,
    on="mousemove",
    empty=False
)

def add_hover_summary(base_chart, df, fields):
    """
    Adds:
    - Synced hover rule across all charts (using global_hover)
    - Multi-field tooltip
    - Cursor point
    """
    # Invisible selector that binds the hover to this chart
    selectors = (
        alt.Chart(df)
        .mark_point(opacity=0)
        .encode(x="Time (h):Q")
        .add_params(global_hover)
    )

    # Synced vertical rule across charts
    rule = (
        alt.Chart(df)
        .mark_rule(color="gray")
        .encode(x="Time (h):Q")
        .transform_filter(global_hover)
    )

    # Tooltip+dot
    tooltip_fields = [
        alt.Tooltip(col, title=title, format=".2f")
        for col, title in fields
    ]

    points = (
        alt.Chart(df)
        .mark_circle(size=60, color="black")
        .encode(
            x="Time (h):Q",
            y=fields[0][0] + ":Q",
            tooltip=tooltip_fields,
        )
        .transform_filter(global_hover)
    )

    # Add shading behind base chart
    shading = add_state_shading(df)

    return shading + base_chart + selectors + rule + points


def add_state_shading(df):
    """
    Creates layered background shading for:
    - Tap active
    - Legionella active
    - Comfort/ECO/Holiday modes
    Returns an Altair chart layer that can be added behind any chart.
    """
    layers = []

    # --- 1. TAPPING (orange) ---
    tap_df = df[df["Tap_Active"] == True]
    if len(tap_df) > 0:
        tap_layer = (
            alt.Chart(tap_df)
            .mark_rect(opacity=0.15, color="#f59e0b")
            .encode(
                x="Time (h):Q",
                x2="Time (h):Q"
            )
        )
        layers.append(tap_layer)

    # --- 2. LEGIONELLA (purple) ---
    leg_df = df[df["Legionella_Active"] == True]
    if len(leg_df) > 0:
        leg_layer = (
            alt.Chart(leg_df)
            .mark_rect(opacity=0.12, color="#8b5cf6")
            .encode(
                x="Time (h):Q",
                x2="Time (h):Q"
            )
        )
        layers.append(leg_layer)

    # --- 3. SensoComfort Modes (blue, green, pink) ---
    mode_colors = {
        "Comfort": "#60a5fa",
        "Auto-Comfort": "#60a5fa",
        "ECO": "#4ade80",
        "Auto-ECO": "#4ade80",
        "Holiday": "#f9a8d4",
        "Boost": "#f87171"
    }

    for mode, color in mode_colors.items():
        mode_df = df[df["Senso_Mode"] == mode]
        if len(mode_df) > 0:
            rect = (
                alt.Chart(mode_df)
                .mark_rect(opacity=0.10, color=color)
                .encode(
                    x="Time (h):Q",
                    x2="Time (h):Q"
                )
            )
            layers.append(rect)

    if layers:
        return alt.layer(*layers)
    else:
        return alt.Chart()

def chart_stratification(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith("T_") and "Avg" not in c]
    d = df.melt("Time (h)", value_vars=cols, var_name="Layer", value_name="Temp (°C)")

    label_map = {
        "T_Bottom Layer (°C)": "Bottom Layer",
        "T_Lower-Mid Layer (°C)": "Lower-Mid Layer",
        "T_Upper-Mid Layer (°C)": "Upper-Mid Layer",
        "T_Top Layer (°C)": "Top Layer",
    }
    d["Layer"] = d["Layer"].map(label_map)

    color_scale = alt.Scale(
        domain=["Top Layer", "Upper-Mid Layer", "Lower-Mid Layer", "Bottom Layer"],
        range=["#dc2626", "#f97316", "#60a5fa", "#1e3a8a"],
    )

    chart = (
        alt.Chart(d)
        .mark_line()
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)", axis=alt.Axis(grid=True)),
            y=alt.Y("Temp (°C):Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Layer:N", scale=color_scale, legend=alt.Legend(title="Tank Layer")),
        )
        .properties(height=400, title="Tank Stratification (4 Layers)")
        .interactive()
    )
    
    # Add day markers for 7-day view with labels
    max_time = d["Time (h)"].max()
    if max_time > 48:  # If more than 2 days, add day separators
        day_data = pd.DataFrame({
            'x': [24 * i for i in range(0, int(max_time / 24) + 1)],
            'label': [f'Day {i+1}' for i in range(int(max_time / 24) + 1)]
        })
        
        # Vertical day lines
        rule = alt.Chart(day_data).mark_rule(strokeDash=[4, 4], color='gray', opacity=0.5).encode(
            x='x:Q'
        )
        
        # Day labels at top
        text = alt.Chart(day_data[day_data['x'] < max_time]).mark_text(
            align='left', dx=5, dy=-180, fontSize=10, color='gray'
        ).encode(
            x='x:Q',
            text='label:N'
        )
        
        chart = chart + rule + text
    
    return chart

def chart_power(df: pd.DataFrame):
    chart = (
        alt.Chart(df)
        .mark_line(color="#ef4444")
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)"),
            y="HP Power (W)"
        )
        .properties(height=300, title="Heat Pump Electrical Power (W)")
    )
    
    # Add day markers for 7-day view
    max_time = df["Time (h)"].max()
    if max_time > 48:
        day_lines = pd.DataFrame({
            'x': [24 * i for i in range(1, int(max_time / 24) + 1)]
        })
        rule = alt.Chart(day_lines).mark_rule(strokeDash=[4, 4], color='gray', opacity=0.5).encode(
            x='x:Q'
        )
        chart = chart + rule
    
    return chart

def chart_heat(df: pd.DataFrame):
    chart = (
        alt.Chart(df)
        .mark_line(color="#1d4ed8")
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)"),
            y="HP Heat (W)"
        )
        .properties(height=300, title="Heat Pump Heat Output (W)")
    )
    
    # Add day markers for 7-day view
    max_time = df["Time (h)"].max()
    if max_time > 48:
        day_lines = pd.DataFrame({
            'x': [24 * i for i in range(1, int(max_time / 24) + 1)]
        })
        rule = alt.Chart(day_lines).mark_rule(strokeDash=[4, 4], color='gray', opacity=0.5).encode(
            x='x:Q'
        )
        chart = chart + rule
    
    return chart

def chart_tank_losses(df: pd.DataFrame):
    """
    Plot total tank losses (W), with synced hover and shading.
    """
    base = (
        alt.Chart(df)
        .mark_line(color="#9333ea", strokeWidth=2)
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)", axis=alt.Axis(grid=True)),
            y=alt.Y("Q_Loss (W):Q", title="Tank Heat Loss (W)", scale=alt.Scale(zero=False))
        )
        .properties(height=300, title="Tank Heat Losses (W)")
    )

    fields = [
        ("Q_Loss (W)", "Tank Loss (W)"),
        ("T_Top Layer (°C)", "Top Temp (°C)"),
        ("T_Avg (°C)", "Tank Avg Temp (°C)"),
        ("HP Power (W)", "HP Power (W)"),
        ("HP Heat (W)", "HP Heat (W)")
    ]

    chart = add_hover_summary(base, df, fields)

    # Day markers for 2+ days
    max_time = df["Time (h)"].max()
    if max_time > 48:
        day_lines = pd.DataFrame({
            'x': [24 * i for i in range(1, int(max_time / 24) + 1)]
        })
        rule = (
            alt.Chart(day_lines)
            .mark_rule(strokeDash=[4, 4], color='gray', opacity=0.5)
            .encode(x='x:Q')
        )
        chart = chart + rule

    return chart


def chart_cop(df: pd.DataFrame):
    chart = (
        alt.Chart(df)
        .mark_line(color="green")
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)"),
            y="COP"
        )
        .properties(height=300, title="Coefficient of Performance (COP)")
    )
    
    # Add day markers for 7-day view
    max_time = df["Time (h)"].max()
    if max_time > 48:
        day_lines = pd.DataFrame({
            'x': [24 * i for i in range(1, int(max_time / 24) + 1)]
        })
        rule = alt.Chart(day_lines).mark_rule(strokeDash=[4, 4], color='gray', opacity=0.5).encode(
            x='x:Q'
        )
        chart = chart + rule
    
    return chart

def chart_modulation(df: pd.DataFrame):
    base = alt.Chart(df).encode(x=alt.X("Time (h):Q", title="Time (hours)"))
    mod = base.mark_area(opacity=0.6, color="#2563eb").encode(y="Modulation:Q")
    hp_on = base.mark_line(color="orange").encode(y="HP_On:Q")
    
    chart = (mod + hp_on).properties(height=300, title="HP Modulation & On/Off State")
    
    # Add day markers for 7-day view
    max_time = df["Time (h)"].max()
    if max_time > 48:
        day_lines = pd.DataFrame({
            'x': [24 * i for i in range(1, int(max_time / 24) + 1)]
        })
        rule = alt.Chart(day_lines).mark_rule(strokeDash=[4, 4], color='gray', opacity=0.5).encode(
            x='x:Q'
        )
        chart = chart + rule
    
    return chart

def chart_tap(df: pd.DataFrame):
    chart = (
        alt.Chart(df)
        .mark_bar(color="#f59e0b")
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)"),
            y="Tap Flow (L/min)"
        )
        .properties(height=250, title="Tap Flow (L/min)")
    )
    
    # Add day markers for 7-day view
    max_time = df["Time (h)"].max()
    if max_time > 48:
        day_lines = pd.DataFrame({
            'x': [24 * i for i in range(1, int(max_time / 24) + 1)]
        })
        rule = alt.Chart(day_lines).mark_rule(strokeDash=[4, 4], color='gray', opacity=0.5).encode(
            x='x:Q'
        )
        chart = chart + rule
    
    return chart

def chart_buffer_strat(df):
    """Chart showing buffer tank stratification across 4 layers."""
    cols = [c for c in df.columns if c.startswith("Buffer Layer")]

    if not cols:
        return alt.Chart()

    d = df.melt("Time (h)", value_vars=cols, var_name="Layer", value_name="Temp (°C)")

    color_scale = alt.Scale(
        domain=[f"Buffer Layer {i+1} (°C)" for i in range(4)],
        range=["#dc2626", "#f97316", "#fbbf24", "#60a5fa"]
    )

    chart = (
        alt.Chart(d)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)"),
            y=alt.Y("Temp (°C):Q", title="Buffer Temperature (°C)", scale=alt.Scale(zero=False)),
            color=alt.Color("Layer:N", scale=color_scale)
        )
        .properties(height=400, title="Buffer Tank Stratification")
    )

    return chart



def chart_legionella(df: pd.DataFrame):
    """Chart showing Legionella cycle activity and top layer temperature."""
    base = alt.Chart(df).encode(x=alt.X("Time (h):Q", title="Time (hours)"))
    
    # Temperature line
    temp_line = base.mark_line(color="#dc2626", strokeWidth=2).encode(
        y=alt.Y("T_Top Layer (°C):Q", title="Temperature (°C)")
    )
    
    # Legionella active indicator (as area)
    legionella_area = base.mark_area(opacity=0.3, color="#8b5cf6").encode(
        y=alt.Y("Legionella_Active:Q", title="Legionella Active", scale=alt.Scale(domain=[0, 1]))
    )
    
    chart = (temp_line + legionella_area).properties(
        height=300, 
        title="Legionella Cycle Activity & Top Layer Temperature"
    )
    
    # Add day markers for 7-day view
    max_time = df["Time (h)"].max()
    if max_time > 48:
        day_lines = pd.DataFrame({
            'x': [24 * i for i in range(1, int(max_time / 24) + 1)],
            'day_label': [f'Day {i+1}' for i in range(int(max_time / 24))]
        })
        rule = alt.Chart(day_lines).mark_rule(strokeDash=[4, 4], color='gray', opacity=0.5).encode(
            x='x:Q'
        )
        chart = chart + rule
    
    return chart

def chart_sensocomfort(df: pd.DataFrame):
    """Chart showing SensoComfort mode changes and setpoint over time."""
    base = alt.Chart(df).encode(x=alt.X("Time (h):Q", title="Time (hours)"))
    
    # Top layer temperature
    temp_line = base.mark_line(color="#2563eb", strokeWidth=2).encode(
        y=alt.Y("T_Top Layer (°C):Q", title="Temperature (°C)")
    )
    
    # SensoComfort setpoint
    setpoint_line = base.mark_line(color="#f59e0b", strokeWidth=2, strokeDash=[5, 5]).encode(
        y=alt.Y("Senso_Setpoint (°C):Q")
    )
    
    chart = (temp_line + setpoint_line).properties(
        height=300,
        title="SensoComfort Control: Temperature vs Setpoint"
    )
    
    # Add day markers for 7-day view
    max_time = df["Time (h)"].max()
    if max_time > 48:
        day_lines = pd.DataFrame({
            'x': [24 * i for i in range(1, int(max_time / 24) + 1)]
        })
        rule = alt.Chart(day_lines).mark_rule(strokeDash=[4, 4], color='gray', opacity=0.5).encode(
            x='x:Q'
        )
        chart = chart + rule
    
    return chart

def chart_tapping_detail(df: pd.DataFrame):
    """Detailed chart showing temperature response during tapping events."""
    base = alt.Chart(df).encode(x=alt.X("Time (h):Q", title="Time (hours)"))
    
    # Temperature line
    temp_line = base.mark_line(color="#dc2626", strokeWidth=2).encode(
        y=alt.Y("T_Top Layer (°C):Q", title="Temperature (°C)", scale=alt.Scale(zero=False))
    )
    
    # Tap active indicator (shaded regions)
    tap_active_df = df[df["Tap_Active"] == True].copy()
    if len(tap_active_df) > 0:
        tap_bars = alt.Chart(tap_active_df).mark_rect(
            opacity=0.2,
            color="orange"
        ).encode(
            x="Time (h):Q",
            x2=alt.X2("Time (h):Q"),
            y=alt.value(0),
            y2=alt.value(400)
        )
        chart = temp_line + tap_bars
    else:
        chart = temp_line
    
    chart = chart.properties(
        height=300,
        title="Top Layer Temperature During Tapping (Orange = Tap Active)"
    )
    
    # Add day markers for 7-day view
    max_time = df["Time (h)"].max()
    if max_time > 48:
        day_lines = pd.DataFrame({
            'x': [24 * i for i in range(1, int(max_time / 24) + 1)]
        })
        rule = alt.Chart(day_lines).mark_rule(strokeDash=[4, 4], color='gray', opacity=0.5).encode(
            x='x:Q'
        )
        chart = chart + rule
    


    
    return chart

def chart_buffer_pump(df: pd.DataFrame):
    """Chart showing buffer pump flow over time."""

    if "Buffer Pump Flow (L/min)" not in df.columns:
        return alt.Chart()

    chart = (
        alt.Chart(df)
        .mark_bar(color="#ef4444")
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)"),
            y=alt.Y("Buffer Pump Flow (L/min):Q", title="Buffer Pump Flow (L/min)")
        )
        .properties(height=250, title="Buffer Pump Flow (L/min)")
    )

    # Add day markers
    max_time = df["Time (h)"].max()
    if max_time > 48:
        day_lines = pd.DataFrame({
            "x": [24 * i for i in range(1, int(max_time / 24) + 1)]
        })
        rule = (
            alt.Chart(day_lines)
            .mark_rule(strokeDash=[4, 4], opacity=0.5)
            .encode(x="x:Q")
        )
        chart = chart + rule

    return chart



# ==============================================================
# MAIN APPLICATION
# ==============================================================
def main():
    # ==============================================================
    # 🧩 SIDEBAR — Simulation Parameters
    # ==============================================================
    with st.sidebar:
        st.header("⚙️ Simulation Parameters")
        initial_tank_temp = st.slider(
    "Initial Tank Temperature (°C)",
    10, 80, 45, 1, key="param_initial_temp"
)
        
        if not st.session_state.kb_checked:
            auto_rebuild_kb_on_open(pdf_filename="8000014609_03.pdf")
            st.session_state.kb_checked = True

        # Continue with Knowledge Base section...


        st.markdown("### Tank and Environment")
        st.write("Tank Volume (L): **150 L (fixed)**")

        # Simulation Period Selector
        sim_period = st.radio(
            "Simulation Period",
            ["24 Hours", "7 Days"],
            index=0,
            horizontal=True,
            key="sim_period"
        )
        
        # Set simulation hours based on selection
        if sim_period == "24 Hours":
            simh = 24
            st.info("📊 Viewing: 24-hour detailed analysis")
        else:
            simh = 168  # 7 days = 168 hours
            st.info("📊 Viewing: 7-day weekly pattern")

        Utank = st.slider("Tank U-value (W/m²K)", 0.1, 2.0, 0.4, 0.1, key="param_utank")
        Tamb = st.slider("Ambient Temp (°C)", 0, 30, 15, key="param_tamb")
        dtm = st.slider("Time Step (min)", 1, 10, 5, key="param_dtm")

        st.markdown("### Heat Pump Control")
        setp = st.slider("Setpoint (°C)", 40, 90, 50, 1, key="param_setp")
        hyst = st.slider("Hysteresis (°C)", 1.0, 10.0, 5.0, 0.5, key="param_hyst")
        Pmax = st.slider("Max HP Power Ratio", 0.5, 1.0, 1.0, 0.05, key="param_pmax")
        Pmin = st.slider("Min Modulation Ratio", 0.1, 0.5, 0.2, 0.05, key="param_pmin")

        st.markdown("---")
        st.markdown("### 🎛️ SensoComfort Controls")
        
        senso_enabled = st.radio(
            "Enable SensoComfort Smart Control",
            ["Off", "On"],
            index=0,
            horizontal=True,
            key="senso_toggle"
        )
        
        if senso_enabled == "On":
            st.markdown("#### Comfort Settings")
            comfort_mode = st.selectbox(
                "Comfort Mode",
                ["ECO", "Comfort", "Auto"],
                index=1,
                key="comfort_mode"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                eco_temp = st.number_input("ECO Temperature (°C)", 40, 60, 45, 1, key="eco_temp")
            with col2:
                comfort_temp = st.number_input("Comfort Temperature (°C)", 45, 65, 55, 1, key="comfort_temp")
            
            st.markdown("#### Time Program")
            enable_time_program = st.checkbox("Enable Time Program", value=True, key="enable_time_program")
            
            if enable_time_program:
                st.markdown("**Morning Comfort Period**")
                col1, col2 = st.columns(2)
                with col1:
                    morning_start = st.slider("Start Hour", 0, 23, 6, 1, key="morning_start")
                with col2:
                    morning_end = st.slider("End Hour", 0, 23, 9, 1, key="morning_end")
                
                st.markdown("**Evening Comfort Period**")
                col1, col2 = st.columns(2)
                with col1:
                    evening_start = st.slider("Start Hour", 0, 23, 17, 1, key="evening_start")
                with col2:
                    evening_end = st.slider("End Hour", 0, 23, 22, 1, key="evening_end")
            else:
                morning_start, morning_end = 6, 9
                evening_start, evening_end = 17, 22
            
            st.markdown("#### Advanced Features")
            boost_mode = st.checkbox("Quick Heat-up Mode", value=False, key="boost_mode")
            holiday_mode = st.checkbox("Holiday Mode (Reduced Temp)", value=False, key="holiday_mode")
            
            if holiday_mode:
                holiday_temp = st.slider("Holiday Temperature (°C)", 35, 50, 40, 1, key="holiday_temp")
            else:
                holiday_temp = 40
                
        else:
            comfort_mode = "Comfort"
            eco_temp = 45
            comfort_temp = 55
            enable_time_program = False
            morning_start, morning_end = 6, 9
            evening_start, evening_end = 17, 22
            boost_mode = False
            holiday_mode = False
            holiday_temp = 40

        st.markdown("---")
        st.markdown("### 🦠 Legionella Protection")
        legionella_enabled = st.radio(
            "Enable Legionella Cycle",
            ["Off", "On"],
            index=0,
            horizontal=True,
            key="legionella_toggle"
        )
        
        if legionella_enabled == "On":
            legionella_temp = st.slider("Legionella Target Temp (°C)", 60, 75, 65, 1, key="legionella_temp")
            legionella_duration = st.slider("Hold Duration (minutes)", 10, 60, 30, 5, key="legionella_duration")
            legionella_frequency = st.slider("Frequency (days)", 1, 14, 7, 1, key="legionella_freq")
            legionella_start_hour = st.slider("Start Hour (24h)", 0, 23, 2, 1, key="legionella_hour")
        else:
            legionella_temp = 65
            legionella_duration = 30
            legionella_frequency = 7
            legionella_start_hour = 2

            st.markdown("---")
        st.markdown("### 🔥 Buffer Tank")
        
        buffer_enabled = st.radio(
            "Enable Buffer",
            ["Off", "On"],
            horizontal=True,
            key="buffer_enabled"
        )
        
        if buffer_enabled == "On":
            buffer_volume_L = st.slider("Buffer Volume (L)", 50, 500, 150, 10, key="buffer_volume_L")
            buffer_temp_init = st.slider("Initial Buffer Temp (°C)", 20, 80, 45, key="buffer_temp_init")
            immersion_enabled = st.checkbox("Enable Immersion Heater", key="immersion_enabled")
            immersion_power_kw = st.slider("Immersion Power (kW)", 0.0, 10.0, 3.0, key="immersion_power_kw")
            boost_cylinder = st.checkbox("Boost Cylinder via Buffer Pump", key="boost_cylinder")
            pump_flow_lpm = st.slider("Pump Flow (L/min)", 2, 30, 12, key="pump_flow_lpm")

        st.markdown("---")
        st.markdown("### 📘 Knowledge Base")
        
        # Test knowledge base button
        if st.button("🧪 Test Knowledge Base", key="btn_test_kb"):
            index, meta = load_faiss_index()
            if index and meta:
                st.success(f"✅ Knowledge base is working!")
                st.write(f"- **Total chunks**: {len(meta)}")
                st.write(f"- **Index dimension**: {index.d}")
                st.write(f"- **Total vectors**: {index.ntotal}")
                
                # Show a sample chunk
                if len(meta) > 0:
                    sample = meta[0]
                    st.write("**Sample chunk:**")
                    st.write(f"- Page: {sample.get('page', 'N/A')}")
                    st.write(f"- Text preview: {sample.get('text', '')[:150]}...")
                    st.write(f"- Images: {len(sample.get('image_paths', []))}")
                    
                # Test retrieval with debug output
                st.markdown("**Test Retrieval:**")
                test_results = retrieve_faiss_context("heat pump operation", top_k=2, show_debug=True)
            else:
                st.error("❌ Knowledge base not found or failed to load")
        
        if st.button("🔁 Rebuild Knowledge Base (PDF → FAISS)", key="btn_rebuild_kb"):
            with st.spinner("🔄 Rebuilding knowledge base..."):
                rebuild_knowledge_base()
            st.success("✅ Knowledge base successfully rebuilt!")
            # Clear the cache to reload
            st.cache_resource.clear()



    # ==============================================================
    # 🚿 Domestic Hot Water Tapping Schedule
    # ==============================================================
    # Base daily tapping pattern
    daily_taps = [
        {"Hour": 7, "Minute": 0, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 7, "Minute": 5, "Energy_kWh": 1.4, "Flow_lpm": 6, "Duration_sec": 300, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 7, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 8, "Minute": 1, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 8, "Minute": 15, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 8, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 8, "Minute": 45, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 9, "Minute": 0, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 9, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 10, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 11, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 11, "Minute": 45, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 12, "Minute": 45, "Energy_kWh": 0.315, "Flow_lpm": 4, "Duration_sec": 101.25, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 14, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 15, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 16, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 18, "Minute": 0, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 18, "Minute": 15, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 18, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 19, "Minute": 0, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 20, "Minute": 30, "Energy_kWh": 0.735, "Flow_lpm": 4, "Duration_sec": 236.25, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 21, "Minute": 15, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 21, "Minute": 30, "Energy_kWh": 1.4, "Flow_lpm": 6, "Duration_sec": 300, "T_inlet": 10, "T_outlet": 50},
    ]
    
    # Expand tapping schedule for 7 days
    tap = {"time": [], "volume": [], "rate_lpm": []}
    num_days = 7 if sim_period == "7 Days" else 1
    
    for day in range(num_days):
        for e in daily_taps:
            tmin = day * 24 * 60 + e["Hour"] * 60 + e["Minute"]
            vol_L = e["Flow_lpm"] * e["Duration_sec"] / 60.0
            tap["time"].append(tmin)
            tap["volume"].append(vol_L)
            tap["rate_lpm"].append(e["Flow_lpm"])


    # ==============================================================
    # ⚙️ Simulation Auto-Run Logic
    # ==============================================================
    params = {
        "sim_period": sim_period,
        "dtm": dtm,
        "simh": simh,
        "Utank": Utank,
        "Tamb": Tamb,
        "setp": setp,
        "hyst": hyst,
        "Pmin": Pmin,
        "Pmax": Pmax,
        "legionella_enabled": legionella_enabled == "On",
        "legionella_temp": legionella_temp,
        "legionella_duration": legionella_duration,
        "legionella_frequency": legionella_frequency,
        "legionella_start_hour": legionella_start_hour,
        "senso_enabled": senso_enabled == "On",
        "comfort_mode": comfort_mode,
        "eco_temp": eco_temp,
        "comfort_temp": comfort_temp,
        "enable_time_program": enable_time_program,
        "morning_start": morning_start,
        "morning_end": morning_end,
        "evening_start": evening_start,
        "evening_end": evening_end,
        "boost_mode": boost_mode,
        "holiday_mode": holiday_mode,
        "holiday_temp": holiday_temp,
        "adaptive_timestep": True,  # Enable adaptive time stepping
        "buffer_enabled": buffer_enabled == "On",
        "buffer_volume_L": st.session_state.get("buffer_volume_L", 0),
        "buffer_temp_init": st.session_state.get("buffer_temp_init", 45),
        "immersion_enabled": st.session_state.get("immersion_enabled", False),
        "immersion_power_kw": st.session_state.get("immersion_power_kw", 0.0),
        "boost_cylinder": st.session_state.get("boost_cylinder", False),
        "pump_flow_lpm": st.session_state.get("pump_flow_lpm", 0),
    }
    param_key = tuple(params.values())

    # Detect parameter changes
    param_changed = (
        "last_params" not in st.session_state
        or st.session_state["last_params"] != param_key
    )

    # --- Run simulation if parameters changed ---
    if param_changed:
        with st.spinner("🔄 Running simulation..."):
            df, summary, layer_properties = run_sim(
                # ✅ NEW: buffer system
                buffer_enabled = st.session_state.get("buffer_enabled") == "On",
                dt_min=params["dtm"],
                sim_hrs=params["simh"],
                Utank=params["Utank"],
                Tamb=params["Tamb"],
                setp=params["setp"],
                hyst=params["hyst"],
                Pmin=params["Pmin"],
                Pmax=params["Pmax"],
                Tsrc=5.0,
                tap=tap,


                ## Legionella
                legionella_enabled=params["legionella_enabled"],
                legionella_temp=params["legionella_temp"],
                legionella_duration=params["legionella_duration"],
                legionella_frequency=params["legionella_frequency"],
                legionella_start_hour=params["legionella_start_hour"],

                ## SensoComfort
                senso_enabled=params["senso_enabled"],
                comfort_mode=params["comfort_mode"],
                eco_temp=params["eco_temp"],
                comfort_temp=params["comfort_temp"],
                enable_time_program=params["enable_time_program"],
                morning_start=params["morning_start"],
                morning_end=params["morning_end"],
                evening_start=params["evening_start"],
                evening_end=params["evening_end"],
                boost_mode=params["boost_mode"],
                holiday_mode=params["holiday_mode"],
                holiday_temp=params["holiday_temp"],
                adaptive_timestep=params["adaptive_timestep"],
                initial_tank_temp=initial_tank_temp,
                

)


            
            st.session_state["df"] = df
            st.session_state["summary"] = summary
            st.session_state["layer_properties"] = layer_properties
            st.session_state["last_params"] = param_key
            st.session_state["ai_paused"] = True

        st.success("✅ Simulation updated! AI assistant paused until parameters stabilize.")
    else:
        st.session_state["ai_paused"] = False

    # ==============================================================
    # 📊 Simulation Results + Graphs
    # ==============================================================
    if "summary" in st.session_state:
        df = st.session_state["df"]
        summary = st.session_state["summary"]
        layer_properties = st.session_state["layer_properties"]

        st.markdown("---")
        st.subheader("📊 Simulation Summary")
        
        # Display period info
        if sim_period == "7 Days":
            st.info("📅 **7-Day Summary** - Weekly energy consumption and performance metrics")
        else:
            st.info("📅 **24-Hour Summary** - Daily energy consumption and performance metrics")
        
        # Main summary metrics
        col1, col2 = st.columns(2)
        with col1:
            for key, val in list(summary.items())[:4]:
                st.write(f"**{key}:** {val:.2f}" if isinstance(val, float) else f"**{key}:** {val}")
        with col2:
            for key, val in list(summary.items())[4:]:
                st.write(f"**{key}:** {val:.2f}" if isinstance(val, float) else f"**{key}:** {val}")
        
        # Add daily breakdown for 7-day period
        if sim_period == "7 Days":
            st.markdown("---")
            st.subheader("📆 Daily Energy & Performance Breakdown")
            
            # Calculate daily statistics
            df['Day'] = (df['Time (h)'] // 24).astype(int)
            daily_stats = []
            
            for day in range(7):
                day_df = df[df['Day'] == day]
                if len(day_df) > 0:
                    daily_energy = day_df["HP Power (W)"].sum() * (params["dtm"] * 60) / 3.6e6
                    daily_heat = day_df["HP Heat (W)"].sum() * (params["dtm"] * 60) / 3.6e6
                    daily_losses = day_df["Q_Loss (W)"].sum() * (params["dtm"] * 60) / 3.6e6
                    daily_runtime = int(day_df["HP_On"].sum() * params["dtm"])
                    daily_cop = daily_heat / daily_energy if daily_energy > 0 else 0
                    daily_legionella = int(day_df["Legionella_Active"].sum() * params["dtm"])
                    
                    # Calculate tap events and volume for the day
                    daily_tap_volume = day_df["Tap Flow (L/min)"].sum() * params["dtm"]
                    daily_tap_events = (day_df["Tap Flow (L/min)"] > 0).sum()
                    
                    daily_stats.append({
                        "Day": f"Day {day + 1}",
                        "Energy (kWh)": f"{daily_energy:.3f}",
                        "Heat Output (kWh)": f"{daily_heat:.3f}",
                        "Losses (kWh)": f"{daily_losses:.3f}",
                        "HP Runtime (min)": daily_runtime,
                        "Avg COP": f"{daily_cop:.2f}",
                        "Tap Volume (L)": f"{daily_tap_volume:.1f}",
                        "Tap Events": daily_tap_events,
                        "Legionella (min)": daily_legionella
                    })
            
            # Add weekly totals row
            total_energy = sum([float(d["Energy (kWh)"]) for d in daily_stats])
            total_heat = sum([float(d["Heat Output (kWh)"]) for d in daily_stats])
            total_losses = sum([float(d["Losses (kWh)"]) for d in daily_stats])
            total_runtime = sum([d["HP Runtime (min)"] for d in daily_stats])
            avg_cop_week = total_heat / total_energy if total_energy > 0 else 0
            total_tap_volume = sum([float(d["Tap Volume (L)"]) for d in daily_stats])
            total_tap_events = sum([d["Tap Events"] for d in daily_stats])
            total_legionella = sum([d["Legionella (min)"] for d in daily_stats])
            
            daily_stats.append({
                "Day": "📊 WEEKLY TOTAL",
                "Energy (kWh)": f"{total_energy:.3f}",
                "Heat Output (kWh)": f"{total_heat:.3f}",
                "Losses (kWh)": f"{total_losses:.3f}",
                "HP Runtime (min)": total_runtime,
                "Avg COP": f"{avg_cop_week:.2f}",
                "Tap Volume (L)": f"{total_tap_volume:.1f}",
                "Tap Events": total_tap_events,
                "Legionella (min)": total_legionella
            })
            
            daily_df_display = pd.DataFrame(daily_stats)
            
            # Style the dataframe to highlight the totals row
            st.dataframe(
                daily_df_display, 
                use_container_width=True, 
                hide_index=True,
                height=350
            )
            
            # Add key insights
            st.markdown("### 📈 Weekly Insights")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Energy", f"{total_energy:.2f} kWh", f"Avg: {total_energy/7:.2f} kWh/day")
            with col2:
                st.metric("Total Hot Water", f"{total_tap_volume:.1f} L", f"Avg: {total_tap_volume/7:.1f} L/day")
            with col3:
                st.metric("Weekly COP", f"{avg_cop_week:.2f}", f"Efficiency: {(avg_cop_week/4)*100:.0f}%")
            with col4:
                if total_legionella > 0:
                    st.metric("Legionella Cycles", f"{total_legionella} min", f"{total_legionella/60:.1f} hours")
                else:
                    st.metric("Legionella Cycles", "Disabled", "0 hours")

        st.markdown("---")
        st.subheader("📈 Graph Viewer")
        
        # Add period indicator
        period_label = "7-Day" if sim_period == "7 Days" else "24-Hour"
        st.caption(f"Viewing: **{period_label}** simulation results")
        
        choice = st.selectbox(
        "Select a graph:",
    [
        "Coefficient of Performance (COP)",
        "Tank Stratification (Multi-Layer)",
        "Heat Pump Power (W)",
        "Heat Pump Heat Output (W)",
        "HP Modulation & On/Off State",
        "Tap Flow (L/min)",
        "🦠 Legionella Cycle Activity",
        "🔥 Tank Heat Losses (W)",
        "🎛️ SensoComfort Control",
        "🚿 Tapping Detail (Temperature Response)",
        "Buffer Stratification (4-Layer)",
        "Buffer Pump Flow (L/min)",

        




    ],
    key="graph_choice"
)

        

        if choice == "Coefficient of Performance (COP)":
            st.altair_chart(chart_cop(df), use_container_width=True)
        elif choice == "Tank Stratification (Multi-Layer)":
            st.altair_chart(chart_stratification(df), use_container_width=True)
        elif choice == "Heat Pump Power (W)":
            st.altair_chart(chart_power(df), use_container_width=True)
        elif choice == "Heat Pump Heat Output (W)":
            st.altair_chart(chart_heat(df), use_container_width=True)
        elif choice == "🔥 Tank Heat Losses (W)":
            st.altair_chart(chart_tank_losses(df), use_container_width=True)
        elif choice == "HP Modulation & On/Off State":
            st.altair_chart(chart_modulation(df), use_container_width=True)
        elif choice == "Tap Flow (L/min)":
            st.altair_chart(chart_tap(df), use_container_width=True)
        elif choice == "🦠 Legionella Cycle Activity":
            st.altair_chart(chart_legionella(df), use_container_width=True)
        elif choice == "🎛️ SensoComfort Control":
            st.altair_chart(chart_sensocomfort(df), use_container_width=True)
            
            # Show mode distribution
            if senso_enabled == "On":
                st.markdown("---")
                st.markdown("#### SensoComfort Mode Distribution")
                mode_counts = df['Senso_Mode'].value_counts()
                mode_df = pd.DataFrame({
                    'Mode': mode_counts.index,
                    'Time (hours)': (mode_counts.values * params["dtm"]) / 60
                })
                
                mode_chart = alt.Chart(mode_df).mark_bar().encode(
                    x=alt.X('Mode:N', title='Operating Mode'),
                    y=alt.Y('Time (hours):Q', title='Time (hours)'),
                    color=alt.Color('Mode:N', legend=None)
                ).properties(height=250)
                
                st.altair_chart(mode_chart, use_container_width=True)
        elif choice == "🚿 Tapping Detail (Temperature Response)":
            st.altair_chart(chart_tapping_detail(df), use_container_width=True)
            st.info("🔍 **Adaptive Time Stepping Enabled**: Orange shaded areas show when tapping occurs. "
                   "Notice the rapid temperature drops during tap events - this improved model captures transients "
                   "that were averaged out in the original 5-minute timestep approach!")
            
        elif choice == "Buffer Stratification (4-Layer)":
            # Check if buffer columns exist in dataframe
            buffer_cols = [c for c in df.columns if c.startswith("Buffer Layer")]
            if buffer_cols:
                st.altair_chart(chart_buffer_strat(df), use_container_width=True)
                # Show buffer configuration info
                st.info(
                    f"📊 Buffer Volume: {st.session_state.get('buffer_volume_L', 0)} L | "
                    f"Initial Temp: {st.session_state.get('buffer_temp_init', 45)}°C | "
                    f"Immersion: {'ON' if st.session_state.get('immersion_enabled', False) else 'OFF'} "
                    f"({st.session_state.get('immersion_power_kw', 0)} kW) | "
                    f"Boost Pump: {'ON' if st.session_state.get('boost_cylinder', False) else 'OFF'} "
                    f"({st.session_state.get('pump_flow_lpm', 0)} L/min)"
                )
            else:
                st.warning("⚠️ Buffer tank is not enabled. Please enable buffer tank in the sidebar to view this graph.")
                st.info("💡 **To enable:** Sidebar → 🔥 Buffer Tank → Enable Buffer: **On**")

        elif choice == "Buffer Pump Flow (L/min)":
                if "Buffer Pump Flow (L/min)" in df.columns:
                    st.altair_chart(chart_buffer_pump(df), use_container_width=True)
                    st.info(
                    f"🧯 Buffer pump is modeled as a constant {st.session_state.get('pump_flow_lpm', 0)} L/min "
                    f"whenever 'Boost Cylinder via Buffer Pump' is enabled."
                )
                else:
                    st.warning("⚠️ Buffer pump is not enabled or no buffer data is available.")
                    st.info("💡 Enable buffer tank and boost pump in the sidebar to see this graph.")


            
     

# ============================
# CONSTANTS
# ============================
RHO = 1000
CP = 4186
BUFFER_RHO = 1000
BUFFER_CP = 4186
P_EL_MAX = 5000

FEEDBACK_FILE = "chat_feedback.csv"
MODEL_FILE = "ml_feedback_model.pkl"
VEC_FILE = "ml_feedback_vectorizer.pkl"
RETRAIN_THRESHOLD = 5


# -----------------------------------------------------
# 🎬 Typewriter Effect
# -----------------------------------------------------
def typewriter_effect(text: str, speed: float = 0.02):
    """Display text with a typewriter effect."""
    placeholder = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(speed)
    return placeholder


# -----------------------------------------------------
# 🤖 Learning Chatbot Feedback Functions
# -----------------------------------------------------
def train_feedback_model(df: pd.DataFrame):
    if len(df) < 5:
        return None, None
    X = [f"{q} {r}" for q, r in zip(df["question"], df["response"])]
    y = (df["helpful"] == "👍 Yes").astype(int)
    vec = TfidfVectorizer(max_features=3000)
    Xv = vec.fit_transform(X)
    model = LogisticRegression(max_iter=1000).fit(Xv, y)
    joblib.dump(vec, VEC_FILE)
    joblib.dump(model, MODEL_FILE)
    return vec, model


@st.cache_data(ttl=60)
def load_feedback_data():
    """Load feedback data from CSV."""
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    else:
        return pd.DataFrame(columns=["timestamp", "question", "response", "helpful", "type"])


# --------------------------------------------------------
# 🔧 Environment setup
# --------------------------------------------------------
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_PANEL"] = "false"


# ==============================================================
# 🔍 ENHANCED IMAGE FILTERING HELPERS (CORRECT ORDER)
# ==============================================================

def is_blank_page(text: str, min_chars: int = 50, min_words: int = 10) -> bool:
    """
    Check if page is essentially blank or contains only headers/footers.
    Returns True if blank (should be FILTERED OUT).
    """
    if not text:
        return True
    
    cleaned = text.strip()
    
    if len(cleaned) < min_chars:
        return True
    
    words = cleaned.split()
    if len(words) < min_words:
        return True
    
    if all(word.replace('.', '').replace(',', '').isdigit() for word in words):
        return True
    
    unique_words = set(words)
    if len(unique_words) < len(words) * 0.5:
        return True
    
    return False


def detect_qr_code_patterns(ocr_text: str) -> bool:
    """
    Detect if text contains QR code patterns or URLs typical in QR codes.
    Returns True if QR code is detected (should be FILTERED OUT).
    """
    if not ocr_text:
        return False
    
    text_lower = ocr_text.lower().strip()
    
    # QR codes often have very little text or specific patterns
    qr_indicators = [
        'http://' in text_lower,
        'https://' in text_lower,
        'www.' in text_lower,
        'scan' in text_lower,
        'qr code' in text_lower,
        'barcode' in text_lower,
        # Very short text indicates QR code
        len(text_lower.split()) < 5 and len(text_lower) < 50,
        # QR codes often have random alphanumeric strings
        any(len(word) > 20 and word.isalnum() for word in text_lower.split()),
    ]
    
    return any(qr_indicators)


def detect_watermark_text(text: str) -> bool:
    """
    Detect common watermark/branding text that should be filtered.
    Returns True if watermark detected (should be FILTERED OUT).
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Expanded watermark patterns - VERY AGGRESSIVE
    watermark_patterns = [
        # Benchmark-specific
        'benchmark',
        'collective mark',
        'mark of quality',
        'compliance and best practice',
        'commissioning, servicing and maintenance',
        'domestic heating',
        'water solutions',
        
        # Common watermarks
        'for reference only',
        'confidential',
        'draft',
        'not for distribution',
        'copyright',
        '©',
        'all rights reserved',
        'proprietary',
        'internal use only',
        'do not copy',
        'sample',
        'preview',
        'watermark',
        
        # Vaillant branding
        'vaillant group',
        'vaillant gmbh',
        'installer',
        'commissioning checklist',
        'service record',
        'best practice',
        
        # Document headers/footers
        'page ',
        'issue date',
        'document number',
        'version',
        'revision',
        'date:',
        'ref:',
    ]
    
    # If text is SHORT and contains ANY watermark pattern, it's likely a watermark
    if len(text.split()) < 20:
        if any(pattern in text_lower for pattern in watermark_patterns):
            return True
    
    # If >50% of text is watermark phrases, reject
    watermark_count = sum(1 for pattern in watermark_patterns if pattern in text_lower)
    words = text_lower.split()
    
    if len(words) > 0:
        # Very aggressive: if 2+ watermark patterns found, likely watermark page
        if watermark_count >= 2:
            return True
        
        # If watermark words make up >20% of text, reject
        if watermark_count > len(set(words)) * 0.2:
            return True
    
    return False


def is_relevant_diagram(base_img_data: Dict, min_width: int = 200, min_height: int = 200) -> bool:
    """
    Filter out QR codes, logos, and other non-technical diagrams.
    Returns True if image should be KEPT.
    """
    width = base_img_data.get("width", 0)
    height = base_img_data.get("height", 0)
    
    if width < min_width or height < min_height:
        return False
    
    aspect_ratio = width / height if height > 0 else 0
    if 0.95 < aspect_ratio < 1.05:
        if width < 400:
            return False
    
    if width > 3000 or height > 3000:
        return False
        
    return True


def has_sufficient_content(ocr_text: str, min_words: int = 5) -> bool:
    """
    Check if image contains meaningful technical content via OCR.
    Returns True if content is relevant (should be KEPT).
    STRICTER version to filter out more noise.
    """
    if not ocr_text or len(ocr_text.strip()) < 15:
        return False
    
    # IMMEDIATE rejection checks
    if detect_qr_code_patterns(ocr_text):
        return False
    
    if detect_watermark_text(ocr_text):
        return False
    
    if is_blank_page(ocr_text):
        return False
    
    # Count words - STRICTER
    words = ocr_text.split()
    if len(words) < min_words:
        return False
    
    # Check for technical keywords
    technical_keywords = [
        'cylinder', 'pump', 'valve', 'sensor', 'temperature', 
        'pressure', 'flow', 'heating', 'water', 'installation',
        'wiring', 'diagram', 'circuit', 'tank', 'coil', 'pipe',
        'control', 'thermostat', 'outlet', 'inlet', 'connector',
        'boiler', 'heat exchanger', 'expansion', 'safety',
        'electrical', 'connection', 'terminal', 'switch',
        'specification', 'dimension', 'mounting', 'bracket',
        'gas', 'flue', 'combustion', 'burner', 'ignition',
        'zone', 'radiator', 'pipework', 'fitting', 'joint',
        'condensate', 'drain', 'vent', 'air', 'exhaust',
        'manifold', 'header', 'mixer', 'actuator', 'relay',
        'mm', 'cm', 'inch', 'bar', 'psi', 'kwh', 'kw'
    ]
    
    text_lower = ocr_text.lower()
    
    technical_count = sum(1 for keyword in technical_keywords if keyword in text_lower)
    
    # Require MORE technical keywords
    if technical_count >= 3:
        return True
    
    # EXPANDED branding keywords to reject
    branding_keywords = [
        'benchmark', 'collective mark', 'quality', 'compliance',
        'best practice', 'commissioning', 'servicing', 'maintenance',
        'mark of quality', 'installer', 'competent person',
        'building regulations', 'certificate', 'declaration',
        'warranty', 'guarantee', 'registration',
        'domestic heating', 'water solutions',
        'vaillant group', 'gmbh'
    ]
    
    branding_count = sum(1 for k in branding_keywords if k in text_lower)
    
    # If MOSTLY branding text, reject
    if branding_count > technical_count:
        return False
    
    # Require at least SOME technical content
    return technical_count >= 1


def clean_ocr_text(ocr_text: str) -> str:
    """
    Remove watermarks, headers, footers, and noise from OCR text.
    Returns cleaned text.
    """
    if not ocr_text:
        return ""
    
    lines = ocr_text.split('\n')
    cleaned_lines = []
    
    watermark_patterns = [
        'for reference only', 'confidential', 'draft',
        'not for distribution', 'copyright', '©',
        'all rights reserved', 'proprietary',
        'vaillant group', 'benchmark', 'mark of quality',
        'page ', 'issue date', 'document number',
        'version', 'revision'
    ]
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if not line_lower:
            continue
        
        if any(pattern in line_lower for pattern in watermark_patterns):
            continue
        
        if line.strip().replace('.', '').replace(',', '').isdigit():
            continue
        
        if len(line.strip()) < 5:
            continue
        
        cleaned_lines.append(line.strip())
    
    return '\n'.join(cleaned_lines)


def extract_text_from_image(image_path: str) -> str:
    """Extract text from image using OCR with cleaning."""
    try:
        import pytesseract
        from PIL import Image
        
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        
        cleaned_text = clean_ocr_text(text)
        
        return cleaned_text.strip()
    except ImportError:
        return "[OCR not available - install pytesseract]"
    except Exception as e:
        return f"[OCR error: {e}]"


def highlight_relevant_text(ocr_text: str, query: str, max_snippets: int = 3) -> List[str]:
    """Find text snippets from OCR that are relevant to the query."""
    if not ocr_text or len(ocr_text) < 10:
        return []
    
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
    query_words = set(query.lower().split())
    scored_lines = []
    
    for line in lines:
        if detect_watermark_text(line):
            continue
        
        line_words = set(line.lower().split())
        overlap = len(query_words & line_words)
        if overlap > 0:
            scored_lines.append((overlap, line))
    
    scored_lines.sort(reverse=True, key=lambda x: x[0])
    return [line for _, line in scored_lines[:max_snippets]]


def resolve_images_for_item(item: Dict) -> List[str]:
    """Resolve images for a manual page."""
    images = []
    base_dir = os.path.dirname(__file__)
    
    for rel_path in item.get("image_paths", []):
        if os.path.isabs(rel_path):
            abs_path = rel_path
        else:
            abs_path = os.path.join(base_dir, rel_path)
        
        if os.path.exists(abs_path):
            images.append(abs_path)
    
    return images


def filter_retrieved_items(items: List[Dict], query: str) -> List[Dict]:
    """
    Filter out low-relevance items from retrieval results.
    BALANCED: Removes obvious noise while keeping technical diagrams.
    """
    filtered = []
    
    for item in items:
        # RELAXED similarity threshold (was 0.30, now 0.25)
        if item.get("similarity", 0) < 0.25:
            continue
        
        item_text = item.get("text", "")
        
        # Check if text content is blank/watermark (keep this)
        if is_blank_page(item_text) or detect_watermark_text(item_text):
            continue
        
        # Filter images - MORE LENIENT
        images = resolve_images_for_item(item)
        relevant_images = []
        
        for img_path in images:
            try:
                # Quick filename check - ONLY obvious noise
                filename_lower = os.path.basename(img_path).lower()
                if any(keyword in filename_lower for keyword in [
                    'benchmark', 'logo', 'qr', 'barcode'
                ]):
                    continue
                
                # Get image dimensions
                try:
                    from PIL import Image
                    img = Image.open(img_path)
                    width, height = img.size
                    
                    # RELAXED size filtering (was 250, now 150)
                    if width < 150 or height < 150:
                        continue
                    
                    # RELAXED aspect ratio filtering
                    aspect_ratio = width / height
                    
                    # Only reject VERY square + VERY small (QR codes)
                    if 0.95 < aspect_ratio < 1.05:  # Nearly perfect square
                        if width < 300:  # And small
                            continue
                    
                    # Keep most aspect ratios (was 4.0/0.25, now 6.0/0.15)
                    if aspect_ratio > 6.0 or aspect_ratio < 0.15:
                        continue
                        
                except Exception:
                    # On dimension error, keep the image
                    pass
                
                # OCR check - MORE FORGIVING
                ocr_text = extract_text_from_image(img_path)
                
                if not ocr_text.startswith("["):  # OCR succeeded
                    # IMMEDIATE rejection only for OBVIOUS QR codes
                    if detect_qr_code_patterns(ocr_text):
                        continue
                    
                    # IMMEDIATE rejection only for PURE watermarks
                    if detect_watermark_text(ocr_text):
                        # Double-check: if has technical words, keep it anyway
                        if not any(word in ocr_text.lower() for word in [
                            'cylinder', 'pump', 'valve', 'temperature', 
                            'pressure', 'flow', 'heating', 'installation',
                            'wiring', 'diagram', 'mm', 'bar', 'kw'
                        ]):
                            continue
                    
                    # RELAXED content requirements (was 5 words, now 3)
                    if has_sufficient_content(ocr_text, min_words=3):
                        relevant_images.append(img_path)
                    else:
                        # Even if insufficient text, keep if it's a large image
                        # (likely a technical diagram with minimal text)
                        if width >= 400 and height >= 400:
                            relevant_images.append(img_path)
                else:
                    # OCR not available - KEEP BY DEFAULT if size OK
                    if width >= 200 and height >= 200:
                        # Only reject if nearly square AND small
                        aspect_ratio = width / height
                        if not (0.95 < aspect_ratio < 1.05 and width < 300):
                            relevant_images.append(img_path)
                    
            except Exception as e:
                # On any error, KEEP the image (was reject, now keep)
                relevant_images.append(img_path)
        
        # Update item with filtered images
        item["image_paths"] = relevant_images
        
        # Keep items with text OR images (more lenient)
        if item.get("text") or relevant_images:
            filtered.append(item)
    
    return filtered


def save_feedback(question: str, response: str, helpful: bool, feedback_type: str = "text"):
    """Save user feedback."""
    try:
        df = load_feedback_data()
        
        new_row = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response": response[:500],
            "helpful": "👍 Yes" if helpful else "👎 No",
            "type": feedback_type
        }])
        
        if "type" not in df.columns:
            df["type"] = "text"
        
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(FEEDBACK_FILE, index=False)
        
        st.success(f"✅ Thank you for your feedback!")
        
    except Exception as e:
        st.error(f"❌ Failed to save feedback: {e}")


# ==============================================================
# 🤖 CHATBOT UI
# ==============================================================

# ==============================================================
# ✅ COMPLETE CHATBOT REPLACEMENT (PATH-FREE IMAGES via Base64)
# Drop-in replacement for your current "render_chatbot" section
# PLUS the KB rebuild changes needed to store images in metadata.
#
# What this fixes:
# - Images show locally but not on Streamlit Cloud  ✅ fixed
# - No filesystem image paths needed at runtime     ✅ fixed
# - load_faiss_index signature mismatch             ✅ fixed
#
# Assumptions:
# - You have PyMuPDF (fitz) available for image extraction.
#   If not, KB will still build text-only.
# - Your PDF is located beside this script: 8000014609_03.pdf
# ==============================================================

import os
import io
import re
import json
import time
import base64
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple

import faiss
from openai import OpenAI
from PyPDF2 import PdfReader

# -----------------------------------------------------
# ✅ Fix: load_faiss_index requires kb_version argument
# -----------------------------------------------------
@st.cache_resource
def load_faiss_index(kb_version: int) -> Tuple[Optional[faiss.Index], Optional[List[Dict[str, Any]]]]:
    kb_dir = os.path.join(os.path.dirname(__file__), "kb")
    idx_path = os.path.join(kb_dir, "vaillant_joint_faiss.index")
    meta_path = os.path.join(kb_dir, "vaillant_joint_meta.json")

    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        return None, None

    try:
        index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    except Exception as e:
        st.error(f"❌ Failed to load FAISS: {e}")
        return None, None


# -----------------------------------------------------
# ✅ Base64 helpers (image bytes <-> base64 strings)
# -----------------------------------------------------
def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64.encode("utf-8"))

def sniff_image_mime(data: bytes) -> str:
    # Basic signatures
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


# -----------------------------------------------------
# ✅ Optional OCR (kept from your logic, but safe)
# -----------------------------------------------------
def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """OCR from image bytes (optional). Returns bracketed message if unavailable."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return (text or "").strip()
    except ImportError:
        return "[OCR not available - install pytesseract + pillow]"
    except Exception as e:
        return f"[OCR error: {e}]"


# -----------------------------------------------------
# ✅ Simple noise filters (keep it conservative)
# -----------------------------------------------------
def detect_qr_code_patterns(ocr_text: str) -> bool:
    if not ocr_text:
        return False
    t = ocr_text.lower().strip()
    return any([
        "http://" in t,
        "https://" in t,
        "www." in t,
        "qr" in t and "code" in t,
        "scan" in t and len(t) < 80,
        len(t.split()) < 5 and len(t) < 50,
        any(len(w) > 20 and w.isalnum() for w in t.split()),
    ])

def detect_watermark_text(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    pats = [
        "benchmark", "collective mark", "mark of quality",
        "commissioning", "servicing", "maintenance",
        "for reference only", "confidential", "draft",
        "copyright", "all rights reserved", "proprietary",
        "vaillant group", "vaillant gmbh",
        "page ", "issue date", "document number", "version", "revision",
    ]
    # Only treat as watermark if short-ish AND contains watermark keywords
    if len(t.split()) < 30 and any(p in t for p in pats):
        return True
    return False

def highlight_relevant_text(ocr_text: str, query: str, max_snippets: int = 3) -> List[str]:
    if not ocr_text or len(ocr_text) < 10:
        return []
    q_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    scored = []
    for ln in lines:
        if detect_watermark_text(ln):
            continue
        ln_words = set(re.findall(r"[a-z0-9]+", ln.lower()))
        overlap = len(q_words & ln_words)
        if overlap > 0:
            scored.append((overlap, ln))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ln for _, ln in scored[:max_snippets]]


# -----------------------------------------------------
# ✅ Embedding helper (batched)
# -----------------------------------------------------
def embed_texts_batched(
    client_local: OpenAI,
    texts: List[str],
    model: str = "text-embedding-3-large",
    batch_size: int = 96,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client_local.embeddings.create(model=model, input=batch)
        embs = [np.array(d.embedding, dtype="float32") for d in resp.data]
        vectors.extend(embs)
    return np.vstack(vectors) if vectors else np.zeros((0, 0), dtype="float32")


# -----------------------------------------------------
# ✅ PDF image extraction (per-page) using PyMuPDF (fitz)
# Stores image bytes as base64.
# -----------------------------------------------------
def extract_page_images_b64(pdf_path: str, page_number_1based: int, max_images: int = 12) -> List[Dict[str, Any]]:
    """
    Returns list of dicts: { "b64": "...", "mime": "image/png", "width": int, "height": int }
    Safe: if fitz missing, returns [].
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number_1based - 1)

        # Get images on page
        img_list = page.get_images(full=True) or []
        for img_i, img in enumerate(img_list[:max_images]):
            xref = img[0]
            pix = None
            try:
                base = doc.extract_image(xref)
                img_bytes = base.get("image", b"")
                if not img_bytes:
                    continue
                mime = base.get("ext", "").lower()
                # ext sometimes returns 'png', 'jpeg'. Convert to MIME
                if mime == "png":
                    mime_s = "image/png"
                elif mime in ("jpg", "jpeg"):
                    mime_s = "image/jpeg"
                elif mime == "webp":
                    mime_s = "image/webp"
                else:
                    mime_s = sniff_image_mime(img_bytes)

                # best-effort width/height from pixmap
                try:
                    pix = fitz.Pixmap(doc, xref)
                    w, h = pix.width, pix.height
                except Exception:
                    w, h = 0, 0
                finally:
                    if pix is not None:
                        try:
                            pix = None
                        except Exception:
                            pass

                out.append({
                    "b64": bytes_to_b64(img_bytes),
                    "mime": mime_s,
                    "width": int(w),
                    "height": int(h),
                })
            except Exception:
                continue

        doc.close()
    except Exception:
        return []
    return out


# -----------------------------------------------------
# ✅ IMPORTANT: Update your rebuild_knowledge_base to store images_b64
# Replace your existing rebuild_knowledge_base with this version.
# (Keeps your chunking/heading logic if you already have it.)
# -----------------------------------------------------
def rebuild_knowledge_base(
    pdf_filename: str = None,
    pdf_name: str = None,
    max_words: int = 450,
    overlap_words: int = 80,
    embedding_model: str = "text-embedding-3-large",
    embedding_batch_size: int = 96,
    min_text_chars: int = 80,
) -> None:
    """
    Rebuild FAISS KB from PDF.
    Stores:
      - text chunks
      - heading
      - page number
      - images_b64 (list of image dicts with base64, mime, width, height)
    """

    # --------- resolve filename ----------
    pdf_filename = pdf_filename or pdf_name or "8000014609_03.pdf"

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error("❌ Missing OPENAI_API_KEY. Cannot rebuild KB.")
        return
    client_local = OpenAI(api_key=api_key)

    base_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(base_dir, pdf_filename)
    if not os.path.exists(pdf_path):
        st.error(f"❌ PDF not found: {pdf_path}")
        return

    kb_dir = os.path.join(base_dir, "kb")
    os.makedirs(kb_dir, exist_ok=True)
    idx_out = os.path.join(kb_dir, "vaillant_joint_faiss.index")
    meta_out = os.path.join(kb_dir, "vaillant_joint_meta.json")

    # --------- open PDF ----------
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        st.error(f"❌ Could not open PDF: {e}")
        return

    n_pages = len(reader.pages)
    items: List[Dict[str, Any]] = []
    pages_used: List[int] = []

    progress = st.progress(0)
    status = st.empty()

    # You already have these in your file; if not, add simple versions:
    def _clean_text(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk_text_with_overlap(text: str, max_words: int, overlap_words: int) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        n = len(words)
        while start < n:
            end = min(start + max_words, n)
            ch = " ".join(words[start:end]).strip()
            if ch:
                chunks.append(ch)
            if end == n:
                break
            start = max(0, end - overlap_words)
        return chunks

    # --------- extract, chunk, attach images ----------
    for pnum in range(1, n_pages + 1):
        status.write(f"📝 Processing page {pnum}/{n_pages}...")
        try:
            page = reader.pages[pnum - 1]
            text = _clean_text(page.extract_text() or "")
            if len(text) < min_text_chars:
                progress.progress(int((pnum / n_pages) * 45))
                continue

            pages_used.append(pnum)

            # Extract images ONCE per page (then attach to each chunk)
            page_images = extract_page_images_b64(pdf_path, pnum)

            raw_chunks = chunk_text_with_overlap(text, max_words=max_words, overlap_words=overlap_words)
            for ch in raw_chunks:
                items.append({
                    "page": pnum,
                    "heading": None,
                    "text": ch,
                    "images_b64": page_images,  # <- base64 images embedded here
                })

        except Exception as e:
            st.warning(f"⚠️ Page {pnum} failed: {e}")

        progress.progress(int((pnum / n_pages) * 45))

    if not items:
        st.error("❌ No usable text extracted. KB rebuild aborted.")
        return

    # --------- embeddings ----------
    status.write(f"🔢 Embedding {len(items)} chunks...")
    texts = [it["text"] for it in items]
    try:
        mat = embed_texts_batched(
            client_local=client_local,
            texts=texts,
            model=embedding_model,
            batch_size=embedding_batch_size,
        )
    except Exception as e:
        st.error(f"❌ Embedding failed: {e}")
        return

    if mat.size == 0:
        st.error("❌ Embedding matrix empty.")
        return

    faiss.normalize_L2(mat)

    # --------- build FAISS ----------
    status.write("📦 Building FAISS index...")
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    progress.progress(95)
    faiss.write_index(index, idx_out)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    progress.progress(100)

    st.session_state.kb_last_build_report = {
        "pdf": pdf_filename,
        "pages_total": n_pages,
        "pages_used": pages_used,
        "chunks_total": len(items),
        "embedding_model": embedding_model,
        "embedding_dim": int(mat.shape[1]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    st.success("✅ Knowledge base rebuild complete.")


# -----------------------------------------------------
# ✅ Retrieval (unchanged logic, but uses correct loader call)
# -----------------------------------------------------
def retrieve_faiss_context(query: str, top_k: int = 3, show_debug: bool = False) -> List[Dict[str, Any]]:
    index, meta = load_faiss_index(st.session_state.kb_version)
    if not index or not meta:
        if show_debug:
            st.warning("⚠️ FAISS index not loaded.")
        return []

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        if show_debug:
            st.error("⚠️ Missing OPENAI_API_KEY")
        return []

    try:
        client_local = OpenAI(api_key=api_key)
        emb_resp = client_local.embeddings.create(model="text-embedding-3-large", input=query)
        emb_vec = np.array(emb_resp.data[0].embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(emb_vec)

        scores, ids = index.search(emb_vec, top_k)
        results: List[Dict[str, Any]] = []

        for score, idx in zip(scores[0], ids[0]):
            if 0 <= idx < len(meta):
                item = dict(meta[idx])
                item["similarity"] = float(score)
                results.append(item)

        if show_debug:
            st.write(f"Top-{len(results)} results:")
            for r in results:
                st.write(f"- p{r.get('page')} sim={r.get('similarity'):.3f}")

        return results
    except Exception as e:
        if show_debug:
            st.error(f"⚠️ Retrieval error: {e}")
        return []


# -----------------------------------------------------
# ✅ OPTIONAL: filter retrieved items lightly (do NOT drop images aggressively)
# -----------------------------------------------------
def filter_retrieved_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for it in items:
        if it.get("similarity", 0) < 0.25:
            continue
        txt = it.get("text", "") or ""
        if detect_watermark_text(txt):
            continue
        filtered.append(it)
    return filtered


# ==============================================================
# ✅ COMPLETE CHATBOT REPLACEMENT
# Replace your entire render_chatbot() with this.
# ==============================================================

def render_chatbot():
    st.subheader("💬 AI Simulation Assistant")
    st.caption("Ask about cylinder specifications, installation, or simulation behavior (manual RAG + diagrams)")

    # Session bootstrap
    if "chatbot_response" not in st.session_state:
        st.session_state.chatbot_response = None
    if "chatbot_query" not in st.session_state:
        st.session_state.chatbot_query = None
    if "retrieved_items" not in st.session_state:
        st.session_state.retrieved_items = []
    if "current_response_id" not in st.session_state:
        st.session_state.current_response_id = None

    # UI controls
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Question",
            key="user_question",
            placeholder="e.g., What is the cylinder volume? How does legionella protection work?",
            label_visibility="collapsed",
        )
    with col2:
        st.write("")
        st.write("")
        ask_button = st.button("🔍 Ask", key="btn_ask", use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🧭 Data Source")
        data_mode = st.radio(
            "Choose response mode:",
            ["Manual Data (RAG from Vaillant PDF)", "OpenAI General Knowledge"],
            index=0,
            key="radio_data_mode",
        )
    with c2:
        st.markdown("### 🔧 Debug Options")
        show_debug = st.checkbox("Show retrieval details", value=False, key="show_debug")
        show_diag_debug = st.checkbox("Show diagram debug", value=False, key="show_diag_debug")

    # Ask action
    if ask_button and query:
        api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
        if not api_key:
            st.warning("⚠️ Please set your OpenAI API key.")
            return

        with st.spinner("🤔 Thinking..."):
            client_chat = OpenAI(api_key=api_key)

            retrieved_items: List[Dict[str, Any]] = []
            kb_context = ""

            if data_mode.startswith("Manual"):
                top_items = retrieve_faiss_context(query, top_k=6, show_debug=show_debug)
                top_items = filter_retrieved_items(top_items)

                if not top_items:
                    st.warning("No relevant manual content found. Try rephrasing your question.")
                    return

                retrieved_items = top_items
                kb_context = "\n\n".join(
                    f"[Manual Page {it.get('page','?')}, Similarity {it.get('similarity',0):.3f}]\n{it.get('text','')}"
                    for it in top_items
                )
            else:
                kb_context = "Use general HVAC and heat pump knowledge."

            # Optional simulation context (kept from your app)
            layer_properties = st.session_state.get("layer_properties", {})
            geometry_context = "\n".join(
                f"{n}: Volume={p['Volume_L']:.1f} L"
                for n, p in layer_properties.items()
            )
            summary = st.session_state.get("summary", {})
            summary_text = "\n".join(f"{k}: {v}" for k, v in summary.items())

            if data_mode.startswith("Manual"):
                system_msg = (
                    "You are a technical assistant for Vaillant heating systems. "
                    "Answer ONLY using the provided Manual Context. "
                    "If the answer is not in the manual, say: "
                    "'This information is not available in the manual.' "
                    "Be concise and technical."
                )
            else:
                system_msg = "You are an HVAC expert assistant."

            prompt = f"""
## Manual Context
{kb_context}

## Simulation Summary
{summary_text}

## Geometry
{geometry_context}

## Question
{query}

Provide a clear, technical answer based on the context above.
""".strip()

            try:
                resp = client_chat.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3
                )
                answer = resp.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"❌ OpenAI error: {e}")
                return

        st.session_state.chatbot_response = answer
        st.session_state.chatbot_query = query
        st.session_state.current_response_id = time.strftime("%Y%m%d-%H%M%S")
        st.session_state.retrieved_items = retrieved_items

    # Render answer
    if st.session_state.chatbot_response and st.session_state.chatbot_query:
        st.markdown("---")
        st.markdown("### 💬 Answer")
        st.markdown(st.session_state.chatbot_response)

        # Diagrams from base64 (NO PATHS)
        if data_mode.startswith("Manual") and st.session_state.retrieved_items:
            st.markdown("---")
            st.markdown("### 📷 Related Technical Diagrams (embedded)")

            total_shown = 0

            if show_diag_debug:
                with st.expander("🐛 DEBUG: Retrieved items + image counts", expanded=True):
                    st.write(f"Items: {len(st.session_state.retrieved_items)}")
                    for i, it in enumerate(st.session_state.retrieved_items, start=1):
                        imgs = it.get("images_b64", []) or []
                        st.write(f"- Item {i}: page={it.get('page')} sim={it.get('similarity',0):.3f} images={len(imgs)}")

            # Show images grouped by page
            for it in st.session_state.retrieved_items:
                page_num = it.get("page", "?")
                images = it.get("images_b64", []) or []
                if not images:
                    continue

                st.markdown(f"#### 📄 Manual Page {page_num}")

                for img_idx, img_obj in enumerate(images, start=1):
                    try:
                        b64 = img_obj.get("b64", "")
                        if not b64:
                            continue
                        img_bytes = b64_to_bytes(b64)
                        mime = img_obj.get("mime") or sniff_image_mime(img_bytes)

                        # Optional QR/watermark filter based on OCR (light touch)
                        ocr_text = ""
                        if mime.startswith("image/"):
                            ocr_text = extract_text_from_image_bytes(img_bytes)
                            if ocr_text and not ocr_text.startswith("[") and detect_qr_code_patterns(ocr_text):
                                continue

                        st.image(
                            img_bytes,
                            caption=f"Page {page_num} - Diagram {img_idx}",
                            use_container_width=True
                        )
                        total_shown += 1

                        # OCR snippets (optional)
                        if ocr_text and not ocr_text.startswith("["):
                            snippets = highlight_relevant_text(ocr_text, st.session_state.chatbot_query)
                            if snippets:
                                with st.expander("📝 Key Text from Diagram", expanded=False):
                                    for s in snippets:
                                        st.markdown(f"• `{s}`")
                            with st.expander("📄 Full OCR Text", expanded=False):
                                st.text(ocr_text)

                    except Exception:
                        continue

            if total_shown == 0:
                st.info("No diagrams were available for these retrieved manual pages (or they were filtered).")

        st.markdown("---")
        st.caption("Tip: If diagrams still don’t appear, click **🔁 Rebuild Knowledge Base** so images get embedded.")


# ==============================================================
# ✅ OPTIONAL: safer auto-rebuild that also clears cache
# (call this from your sidebar startup, same as you did)
# ==============================================================

def auto_rebuild_kb_on_open(pdf_filename: str = "8000014609_03.pdf") -> None:
    base_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(base_dir, pdf_filename)

    kb_dir = os.path.join(base_dir, "kb")
    idx_path = os.path.join(kb_dir, "vaillant_joint_faiss.index")
    meta_path = os.path.join(kb_dir, "vaillant_joint_meta.json")

    if not os.path.exists(pdf_path):
        st.warning("⚠️ Manual PDF not found – skipping KB rebuild.")
        return

    pdf_mtime = os.path.getmtime(pdf_path)
    kb_exists = os.path.exists(idx_path) and os.path.exists(meta_path)

    if "pdf_mtime" not in st.session_state:
        st.session_state.pdf_mtime = None
    if "kb_version" not in st.session_state:
        st.session_state.kb_version = 0

    needs_rebuild = (not kb_exists) or (st.session_state.pdf_mtime != pdf_mtime)

    if needs_rebuild:
        with st.spinner("📘 Building knowledge base (auto)..."):
            rebuild_knowledge_base(pdf_filename=pdf_filename)

        st.session_state.pdf_mtime = pdf_mtime
        st.session_state.kb_version += 1
        st.cache_resource.clear()
        st.success("✅ Knowledge base rebuilt automatically.")
        st.rerun()


# ==============================================================
# 🔄 ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    main()
    render_chatbot()