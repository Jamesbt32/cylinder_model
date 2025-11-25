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
import fitz
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


st.markdown("""
<style>

/* Fixed collapsible chat shell */
.chat-shell {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: #0f172a;
    border-top: 2px solid #334155;
    z-index: 999;
    box-shadow: 0 -4px 10px rgba(0,0,0,0.3);
}

/* Header bar */
.chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    cursor: pointer;
    background: #020617;
    color: white;
    font-weight: 600;
}

/* Expandable body */
.chat-body {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.35s ease-in-out;
    padding: 0 1rem;
}

/* When expanded */
.chat-shell.expanded .chat-body {
    max-height: 400px;
    padding-bottom: 1rem;
}

/* Prevent graphs from being hidden */
.block-container {
    padding-bottom: 160px !important;
}

</style>
""", unsafe_allow_html=True)

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


summary = st.session_state.get("summary", {})

# -----------------------------------------------------
# 🤖 Learning Chatbot Feedback + Retraining
# -----------------------------------------------------
FEEDBACK_FILE = "chat_feedback.csv"
MODEL_FILE = "ml_feedback_model.pkl"
VEC_FILE = "ml_feedback_vectorizer.pkl"
RETRAIN_THRESHOLD = 5

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

@st.cache_data(ttl=60)  # Cache for 60 seconds to balance freshness and speed
def load_feedback_data():
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    else:
        return pd.DataFrame(columns=["timestamp", "question", "response", "helpful"])
    
    

# --------------------------------------------------------
# 🔧 Environment setup
# --------------------------------------------------------
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_PANEL"] = "false"

# --------------------------------------------------------
# 🔑 OpenAI client
# --------------------------------------------------------
API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)

if not API_KEY:
    st.warning("⚠️ OPENAI_API_KEY not found. Please add it to Streamlit secrets or environment.")
    client = None
else:
    client = OpenAI(api_key=API_KEY)

# --------------------------------------------------------
# 📂 Load FAISS index + metadata
# --------------------------------------------------------
@st.cache_resource
def load_faiss_index():
    """
    Load the FAISS index and associated metadata (manual chunks + images).
    Returns (index, metadata) or (None, None) if files are missing.
    """
    # Try multiple possible paths
    possible_paths = [
        # Absolute path (original)
        (r"C:\Users\james\OneDrive\Documents\Python\kb\vaillant_joint_faiss.index",
         r"C:\Users\james\OneDrive\Documents\Python\kb\vaillant_joint_meta.json"),
        # Relative to script
        (os.path.join(os.path.dirname(__file__), "kb", "vaillant_joint_faiss.index"),
         os.path.join(os.path.dirname(__file__), "kb", "vaillant_joint_meta.json")),
        # Current directory
        (os.path.join("kb", "vaillant_joint_faiss.index"),
         os.path.join("kb", "vaillant_joint_meta.json")),
    ]
    
    INDEX_PATH = None
    META_PATH = None
    
    # Find first existing path
    for idx_path, meta_path in possible_paths:
        if os.path.exists(idx_path) and os.path.exists(meta_path):
            INDEX_PATH = idx_path
            META_PATH = meta_path
            break
    
    if not INDEX_PATH or not META_PATH:
        st.warning("⚠️ Knowledge base not found. Please rebuild it using the sidebar button.")
        return None, None

    try:
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    except Exception as e:
        st.error(f"❌ Failed to load FAISS: {e}")
        return None, None

# --------------------------------------------------------
# 🔍 Retrieve relevant manual chunks via FAISS
# --------------------------------------------------------
def retrieve_faiss_context(query: str, top_k: int = 3, show_debug: bool = False):
    """
    Retrieve the top_k most relevant manual chunks from the FAISS index
    given a user query. Uses OpenAI embeddings for similarity search.
    
    Args:
        query: The search query
        top_k: Number of results to return
        show_debug: If True, display debug information (for testing only)
    """
    index, meta = load_faiss_index()
    if not index or not meta:
        if show_debug:
            st.warning("⚠️ FAISS index not loaded. Cannot retrieve manual context.")
        return []

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        if show_debug:
            st.error("⚠️ Missing OPENAI_API_KEY")
        return []

    try:
        client = OpenAI(api_key=api_key)
        
        emb_resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=query,
        )
        emb_vec = np.array(emb_resp.data[0].embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(emb_vec)

        scores, ids = index.search(emb_vec, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if 0 <= idx < len(meta):
                item = meta[idx]
                item["similarity"] = float(score)
                results.append(item)
                
        # Only display debug info if explicitly requested
        if show_debug and results:
            st.success(f"✅ Found {len(results)} relevant manual sections")
            with st.expander("📖 Retrieved Context (click to view)", expanded=False):
                for i, item in enumerate(results, 1):
                    st.write(f"**Chunk {i}** (Page {item.get('page', '?')}, Similarity: {item.get('similarity', 0):.3f})")
                    st.write(item.get('text', '')[:200] + "...")
                    if item.get('image_paths'):
                        st.write(f"  📷 {len(item.get('image_paths', []))} images available")
        elif show_debug and not results:
            st.warning("⚠️ No relevant context found in knowledge base")
            
        return results

    except Exception as e:
        if show_debug:
            st.error(f"⚠️ Error retrieving FAISS context: {e}")
            import traceback
            st.code(traceback.format_exc())
        return []

def rebuild_knowledge_base():
    """
    Extracts text + diagrams from the Vaillant PDF, generates embeddings,
    and builds FAISS + metadata.
    """
    st.info("📘 Starting knowledge base rebuild…")

    if not API_KEY or not client:
        st.error("❌ Missing OpenAI API key.")
        return

    pdf_path = os.path.join(os.path.dirname(__file__), "8000014609_03.pdf")
    if not os.path.exists(pdf_path):
        st.error(f"❌ PDF not found: {pdf_path}")
        return

    try:
        reader = PdfReader(pdf_path)
        doc = fitz.open(pdf_path)
    except Exception as e:
        st.error(f"Could not open PDF: {e}")
        return

    items, page_texts, page_images = [], {}, {}

    def chunk_text(text, max_len=700):
        sents = text.split(". ")
        chunks, buf, ln = [], [], 0
        for s in sents:
            buf.append(s)
            ln += len(s.split())
            if ln > max_len:
                chunks.append(". ".join(buf))
                buf, ln = [], 0
        if buf:
            chunks.append(". ".join(buf))
        return chunks

    # Extract text
    st.info("📝 Extracting text from PDF…")
    for pnum, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
            if text and len(text.strip()) > 20:
                page_texts[pnum] = chunk_text(text.strip())
        except Exception as e:
            st.warning(f"⚠️ Text extract error on page {pnum}: {e}")

    # Extract images
    st.info("🖼️ Extracting diagrams…")
    base_dir = os.path.dirname(__file__)
    kb_dir = os.path.join(base_dir, "kb")
    img_dir = os.path.join(kb_dir, "diagrams")
    os.makedirs(img_dir, exist_ok=True)

    for pnum, page in enumerate(doc, start=1):
        imgs = page.get_images(full=True)
        if imgs:
            for i, img in enumerate(imgs):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    w, h = base_image.get("width", 0), base_image.get("height", 0)
                    if w < 50 or h < 50:
                        continue
                    fname = f"page_{pnum}_img_{i+1}.{base_image['ext']}"
                    path = os.path.join(img_dir, fname)
                    with open(path, "wb") as f:
                        f.write(base_image["image"])
                    page_images.setdefault(pnum, []).append(path)
                except Exception as e:
                    st.warning(f"⚠️ Image extract error on page {pnum}: {e}")
        else:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                path = os.path.join(img_dir, f"page_{pnum}.png")
                pix.save(path)
                page_images.setdefault(pnum, []).append(path)
            except Exception as e:
                st.warning(f"⚠️ Rasterization failed on page {pnum}: {e}")

    # Combine text + images
    for pnum, chunks in page_texts.items():
        imgs = page_images.get(pnum, [])
        for c in chunks:
            items.append({"page": pnum, "text": c, "image_paths": imgs})

    st.success(f"✅ Extracted {len(items)} text chunks.")

    # Generate embeddings
    st.info("🔢 Generating OpenAI embeddings…")
    embeddings = []
    for i, it in enumerate(items, 1):
        try:
            emb = client.embeddings.create(model="text-embedding-3-large", input=it["text"])
            embeddings.append(np.array(emb.data[0].embedding, dtype="float32"))
        except Exception as e:
            st.warning(f"Embedding error at chunk {i}: {e}")
            embeddings.append(np.zeros(3072, dtype="float32"))
        if i % 25 == 0:
            st.write(f"Progress: {i}/{len(items)} chunks")

    emb_matrix = np.vstack(embeddings)
    faiss.normalize_L2(emb_matrix)

    # Build FAISS index
    index = faiss.IndexFlatIP(emb_matrix.shape[1])
    index.add(emb_matrix)

    # Save
    os.makedirs(kb_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(kb_dir, "vaillant_joint_faiss.index"))
    with open(os.path.join(kb_dir, "vaillant_joint_meta.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    st.success("🎉 Knowledge base successfully rebuilt!")

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







        # ==============================================================
        # 💬 AI CHATBOT SECTION (BELOW GRAPHS)
        # ==============================================================
st.markdown("---")
st.subheader("💬 AI Simulation Assistant")
st.caption("Ask things like: *Why does the COP drop during tapping?* or *Explain stratification losses.*")

        # --- User input widgets ---
query = st.text_input("Your question:", key="user_question")

st.markdown("### 🧭 Data Source")
data_mode = st.radio(
            "Choose how the assistant should respond:",
            ["Manual Data (RAG from Vaillant PDF)", "OpenAI General Knowledge"],
index=0,
horizontal=True,
key="radio_data_mode",
        )

        # -----------------------------------------------------
        # 💬 Learning Chatbot Integration with Spinner & Typewriter
        # -----------------------------------------------------
        
        # Initialize session state for storing responses
if 'chatbot_response' not in st.session_state:
            st.session_state['chatbot_response'] = None
if 'chatbot_query' not in st.session_state:
            st.session_state['chatbot_query'] = None
if 'retrieved_items' not in st.session_state:
            st.session_state['retrieved_items'] = []
        
        # Only run chatbot if query changed
if query and query != st.session_state['chatbot_query']:
            api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
            if not api_key:
                st.warning("⚠️ Please set your OpenAI API key.")
            else:
                # Show spinner while processing
                with st.spinner("🔄 Thinking..."):
                    client_chat = OpenAI(api_key=api_key)

                    # === Generate multiple candidate responses ===
                    responses = []
                    retrieved_items = []  # Store retrieved context
                    
                    for _ in range(3):
                        kb_context = ""
                        if data_mode.startswith("Manual"):
                            top_items = retrieve_faiss_context(query, top_k=3)
                            retrieved_items = top_items  # Save for display
                            
                            if top_items:
                                kb_context = "\n\n".join(
                                    [f"[Manual Page {it.get('page','?')}, Similarity: {it.get('similarity', 0):.3f}]\n{it.get('text','')}" 
                                     for it in top_items]
                                )
                            else:
                                kb_context = "No specific manual context found for this query."

                        layer_properties = st.session_state.get("layer_properties", {})

                        geometry_context = "\n".join(
    [f"{n}: Vol={p['Volume_L']:.1f} L" for n, p in layer_properties.items()]
)

                        
                        summary_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])

                        prompt = f"""
You are an HVAC expert analyzing a Vaillant 150 L stratified cylinder with a modulating heat pump.

## Manual Context
{kb_context}

## Simulation Summary
{summary_text}

## Geometry
{geometry_context}

## Question
{query}

Please provide a detailed, technical answer based on the information provided above.
"""
                        try:
                            r = client_chat.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": prompt}]
                            )
                            responses.append(r.choices[0].message.content.strip())
                        except Exception as e:
                            st.error(f"OpenAI error: {e}")
                            break

                    if responses:
                        # === Load or train ML feedback model ===
                        feedback_df = load_feedback_data()
                        if os.path.exists(MODEL_FILE) and os.path.exists(VEC_FILE):
                            vec = joblib.load(VEC_FILE)
                            model = joblib.load(MODEL_FILE)
                        else:
                            vec, model = train_feedback_model(feedback_df)

                        # === Rank responses ===
                        if model and vec:
                            Xtest = [query + " " + r for r in responses]
                            probs = model.predict_proba(vec.transform(Xtest))[:, 1]
                            best_idx = int(np.argmax(probs))
                        else:
                            best_idx = 0

                        best_response = responses[best_idx]
                        
                        # Store in session state
                        st.session_state['chatbot_response'] = best_response
                        st.session_state['chatbot_query'] = query
                        st.session_state['retrieved_items'] = retrieved_items
        
        # Display stored response if available
if st.session_state['chatbot_response'] and query:
            best_response = st.session_state['chatbot_response']
            retrieved_items = st.session_state['retrieved_items']
            
            # === Show final answer with typewriter effect (only on first display) ===
            st.markdown("### 💬 Chatbot Response")
            if query != st.session_state.get('last_displayed_query'):
                typewriter_effect(best_response, speed=0.01)
                st.session_state['last_displayed_query'] = query
            else:
                st.markdown(best_response)
            
            # === Display retrieved images if using manual data ===
            pages_with_images = []
            if data_mode.startswith("Manual") and retrieved_items:
                st.markdown("---")
                st.markdown("### 📷 Related Diagrams from Manual")
                
                for idx, item in enumerate(retrieved_items, 1):
                    if item.get('image_paths'):
                        page_num = item.get('page', '?')
                        pages_with_images.append(page_num)
                        
                        st.markdown(f"#### 📄 Page {page_num}")
                        for img_path in item.get('image_paths', []):
                            if os.path.exists(img_path):
                                st.image(img_path, caption=f"Page {page_num}", use_column_width=True)
                            else:
                                st.warning(f"Image not found: {img_path}")
                        
                        # Feedback for this specific page using form for faster submission
                        with st.form(key=f"form_page_{page_num}_{hash(query)}"):
                            st.markdown(f"**Was Page {page_num} helpful for your question?**")
                            feedback_page = st.radio(
                                f"Page {page_num} helpfulness:",
                                ["👍 Yes", "👎 No"],
                                horizontal=True,
                                key=f"radio_page_{page_num}_{hash(query)}"
                            )
                            submitted_page = st.form_submit_button(f"💾 Save Feedback for Page {page_num}")
                            
                            if submitted_page:
                                # Clear cache to get fresh data
                                load_feedback_data.clear()
                                feedback_df = load_feedback_data()
                                new_row = {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "question": f"{query} (Page {page_num})",
                                    "response": f"Page {page_num} content",
                                    "helpful": feedback_page
                                }
                                feedback_df = pd.concat([feedback_df, pd.DataFrame([new_row])], ignore_index=True)
                                feedback_df.to_csv(FEEDBACK_FILE, index=False)
                                load_feedback_data.clear()  # Clear again after save
                                st.success(f"✅ Feedback saved for Page {page_num}!")
                        
                        st.markdown("---")
            
            # === General feedback using form for faster submission ===
            with st.form(key=f"form_overall_{hash(query)}"):
                st.markdown("### 📝 Overall Response Feedback")
                feedback = st.radio(
                    "Was this overall answer helpful?",
                    ["👍 Yes", "👎 No"],
                    horizontal=True,
                    key=f"radio_overall_{hash(query)}"
                )
                submitted_overall = st.form_submit_button("💾 Save Overall Feedback")
                
                if submitted_overall:
                    # Clear cache to get fresh data
                    load_feedback_data.clear()
                    feedback_df = load_feedback_data()
                    new_row = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": query,
                        "response": best_response,
                        "helpful": feedback
                    }
                    feedback_df = pd.concat([feedback_df, pd.DataFrame([new_row])], ignore_index=True)
                    feedback_df.to_csv(FEEDBACK_FILE, index=False)
                    load_feedback_data.clear()  # Clear again after save
                    st.success("✅ Feedback saved!")

                    # Retrain if threshold met
                    if len(feedback_df) % RETRAIN_THRESHOLD == 0:
                        vec, model = train_feedback_model(feedback_df)
                        st.success("🎯 Model retrained with latest feedback!")

                    # === Dashboard ===
                    with st.expander("📊 Feedback Dashboard", expanded=False):
                        if feedback_df.empty:
                            st.info("No feedback data yet.")
                        else:
                            total = len(feedback_df)
                            helpful = (feedback_df["helpful"] == "👍 Yes").sum()
                            ratio = helpful / total * 100
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Total", total)
                            c2.metric("Helpful", helpful)
                            c3.metric("Helpful %", f"{ratio:.1f}%")

                            feedback_df["date"] = pd.to_datetime(feedback_df["timestamp"]).dt.date
                            daily = (
                                feedback_df.groupby("date")["helpful"]
                                .apply(lambda x: (x == "👍 Yes").mean() * 100)
                                .reset_index(name="Helpful %")
                            )
                            chart = (
                                alt.Chart(daily)
                                .mark_line(point=True)
                                .encode(x="date:T", y="Helpful %:Q")
                                .properties(height=250, title="Helpful Feedback Trend")
                            )
                            st.altair_chart(chart, use_container_width=True)
                            st.dataframe(feedback_df.sort_values("timestamp", ascending=False))


                            

# --- Entry point ---
if __name__ == "__main__":
    main()