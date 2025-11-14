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

st.set_page_config(
    page_title="Vaillant 150 L Cylinder Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    legionella_enabled: bool = False,
    legionella_temp: float = 65.0,
    legionella_duration: float = 30.0,
    legionella_frequency: float = 7.0,
    legionella_start_hour: int = 2,
):
    """4-layer stratified tank simulation with optional Legionella cycle."""
    for key in ["time", "volume", "rate_lpm"]:
        if key not in tap:
            raise ValueError(f"tap dict missing '{key}'")

    tap_time = np.array(tap["time"])
    tap_vol = np.array(tap["volume"])
    tap_rate = np.array(tap["rate_lpm"])

    N_layers = 4
    dt_s = dt_min * 60.0
    steps = int(sim_hrs * 60.0 / dt_min)
    time_h = np.arange(0, steps * dt_min, dt_min) / 60.0

    def mm2_to_m2(x): return x / 1e6
    layer_order = ["Bottom Layer", "Lower-Mid Layer", "Upper-Mid Layer", "Top Layer"]

    V_layers_m3 = np.array([BASE_LAYER_PROPERTIES[n]["Volume_L"] / 1000.0 for n in layer_order])
    A_ext = np.array([mm2_to_m2(BASE_LAYER_PROPERTIES[n]["Water_External_Surface_mm2"]) for n in layer_order])
    A_int = np.array([mm2_to_m2(BASE_LAYER_PROPERTIES[n]["Water_Layer_Surface_mm2"]) for n in layer_order])

    m_layer = RHO * V_layers_m3
    UA_loss_layer = Utank * A_ext
    U_int = 50.0
    UA_int_layer = U_int * A_int

    Pmax_W = P_EL_MAX * Pmax
    Pmin_W = Pmax_W * Pmin
    Ton = setp - hyst
    coil_bottom_idx = 2

    T = np.zeros((steps, N_layers))
    Qhp = np.zeros(steps)
    Pel = np.zeros(steps)
    COPs = np.zeros(steps)
    HP_on = np.zeros(steps, bool)
    Qloss_total = np.zeros(steps)
    Tap_flow = np.zeros(steps)
    Tap_temp = np.zeros(steps)
    Mod_frac = np.zeros(steps)
    Legionella_active = np.zeros(steps, bool)

    T_bottom_init = setp - 15.0
    T_top_init = setp
    T[0, :] = np.linspace(T_bottom_init, T_top_init, N_layers)

    # Legionella cycle tracking
    legionella_cycle_start_time = None
    legionella_in_progress = False

    for k in range(1, steps):
        T_prev = T[k - 1, :].copy()
        T_new = T_prev.copy()
        T_top = T_prev[-1]
        T_bottom = T_prev[0]
        tnow_min = k * dt_min
        tnow_hrs = tnow_min / 60.0

        # ============================================================
        # 🦠 LEGIONELLA CYCLE LOGIC
        # ============================================================
        legionella_override = False
        
        if legionella_enabled:
            current_day = int(tnow_hrs / 24.0)
            hour_of_day = tnow_hrs % 24.0
            
            # Check if it's time to start a new cycle
            if (current_day % legionella_frequency == 0 and 
                legionella_start_hour <= hour_of_day < legionella_start_hour + 0.1 and
                not legionella_in_progress):
                legionella_cycle_start_time = tnow_min
                legionella_in_progress = True
            
            # Check if cycle is in progress
            if legionella_in_progress:
                elapsed = tnow_min - legionella_cycle_start_time
                if elapsed < legionella_duration:
                    legionella_override = True
                    Legionella_active[k] = True
                else:
                    legionella_in_progress = False
        
        # ============================================================
        # HEAT PUMP CONTROL (with Legionella override)
        # ============================================================
        mod = 0.0
        on = False
        
        if legionella_override:
            # During Legionella cycle: heat to legionella_temp
            if T_top < legionella_temp:
                on = True
                if T_top <= legionella_temp - 5.0:
                    mod = 1.0  # Full power when far from target
                else:
                    raw = (legionella_temp - T_top) / 5.0
                    mod = max(Pmin, np.clip(raw, 0.0, 1.0))
        else:
            # Normal operation: heat to setpoint
            if T_top < setp:
                on = True
                if T_top <= Ton:
                    mod = 1.0
                else:
                    raw = (setp - T_top) / hyst
                    mod = Pmin + (1 - Pmin) * np.clip(raw, 0.0, 1.0)
        if mod < Pmin:
            on = False
            mod = 0.0
        Mod_frac[k] = mod

        Q = Pe = COP = 0.0
        if on:
            Q, Pe, COP = solve_hp(T_bottom, mod, Pmax_W, Pmin_W, Pmin)
        HP_on[k] = on
        Qhp[k] = Q
        Pel[k] = Pe
        COPs[k] = COP

        idx = np.where((tap_time > tnow_min - dt_min) & (tap_time <= tnow_min))[0]
        Vtap_L = np.sum(tap_vol[idx]) if idx.size > 0 else 0.0
        mdot_tap = (Vtap_L / 1000.0) * RHO / dt_s
        Tap_flow[k] = np.sum(tap_rate[idx]) if idx.size > 0 else 0.0
        Tap_temp[k] = T_top

        for i in range(N_layers):
            Q_net = 0.0

            if i > 0:
                Q_net += UA_int_layer[i] * (T_prev[i - 1] - T_prev[i])
            if i < N_layers - 1:
                Q_net += UA_int_layer[i] * (T_prev[i + 1] - T_prev[i])

            Q_loss = UA_loss_layer[i] * (T_prev[i] - Tamb)
            Q_net -= Q_loss
            Qloss_total[k] += Q_loss

            if i < coil_bottom_idx and Qhp[k] > 0:
                Q_net += Qhp[k] / coil_bottom_idx

            if i == N_layers - 1:
                Q_net -= mdot_tap * C_P * T_prev[i]
            elif i == 0:
                Q_net += mdot_tap * C_P * Tamb

            dT = Q_net * dt_s / (m_layer[i] * C_P)
            T_new[i] = np.clip(T_prev[i] + dT, 0.0, 100.0)

        for i in range(N_layers - 1):
            if T_new[i] > T_new[i + 1]:
                avg = 0.5 * (T_new[i] + T_new[i + 1])
                T_new[i] = avg
                T_new[i + 1] = avg

        T[k, :] = T_new

    df = pd.DataFrame({
        "Time (h)": time_h,
        "HP Power (W)": Pel,
        "HP Heat (W)": Qhp,
        "COP": COPs,
        "HP_On": HP_on,
        "Modulation": Mod_frac,
        "Tap Flow (L/min)": Tap_flow,
        "Tap Temp (°C)": Tap_temp,
        "Q_Loss (W)": Qloss_total,
        "Legionella_Active": Legionella_active,
    })

    for i, name in enumerate(layer_order):
        df[f"T_{name} (°C)"] = T[:, i]
    df["T_Avg (°C)"] = T.mean(axis=1)

    total_heat_kWh = df["HP Heat (W)"].sum() * dt_s / 3.6e6
    total_power_kWh = df["HP Power (W)"].sum() * dt_s / 3.6e6
    total_losses_kWh = df["Q_Loss (W)"].sum() * dt_s / 3.6e6
    hp_runtime_min = int(df["HP_On"].sum() * dt_min)
    avg_cop = total_heat_kWh / total_power_kWh if total_power_kWh > 0 else 0.0
    legionella_runtime_min = int(df["Legionella_Active"].sum() * dt_min) if legionella_enabled else 0

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

    return df, summary, BASE_LAYER_PROPERTIES

# ---------- CHARTS ----------
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

# ==============================================================
# MAIN APPLICATION
# ==============================================================
def main():
    # ==============================================================
    # 🧩 SIDEBAR — Simulation Parameters
    # ==============================================================
    with st.sidebar:
        st.header("⚙️ Simulation Parameters")

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
                legionella_enabled=params["legionella_enabled"],
                legionella_temp=params["legionella_temp"],
                legionella_duration=params["legionella_duration"],
                legionella_frequency=params["legionella_frequency"],
                legionella_start_hour=params["legionella_start_hour"],
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
        elif choice == "HP Modulation & On/Off State":
            st.altair_chart(chart_modulation(df), use_container_width=True)
        elif choice == "Tap Flow (L/min)":
            st.altair_chart(chart_tap(df), use_container_width=True)
        elif choice == "🦠 Legionella Cycle Activity":
            st.altair_chart(chart_legionella(df), use_container_width=True)

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
