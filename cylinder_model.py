# --- cylinder_model_multilayer_150L.py ---
# Vaillant 150 L stratified cylinder + modulating HP model
# Geometry-aware, with graphs, 3D PyVista viz, and optional AI assistant

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

st.set_page_config(
    page_title="Vaillant 150 L Cylinder Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def load_feedback_data():
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    else:
        return pd.DataFrame(columns=["timestamp", "question", "response", "helpful"])


# then your app content:
def main():
    st.title("Vaillant 150 L Cylinder Model Simulation")
    st.write("This text now sits next to the sidebar, using the full screen width.")

# --------------------------------------------------------
# 🔧 Environment setup
# --------------------------------------------------------
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_PANEL"] = "false"

# --------------------------------------------------------
# 🔑 OpenAI client
# --------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
client = OpenAI(api_key=api_key) if api_key else None

# --------------------------------------------------------
# 📂 Load FAISS index + metadata
# --------------------------------------------------------
@st.cache_resource
@st.cache_resource
def load_faiss_index():
    """
    Load the FAISS index and associated metadata (manual chunks + images).
    Returns (index, metadata) or (None, None) if files are missing.
    """
    INDEX_PATH = "vaillant_joint_faiss.index"
    META_PATH = "vaillant_joint_meta.json"

    print(f"🔍 Looking for index: {INDEX_PATH}")
    print(f"🔍 Looking for meta: {META_PATH}")

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print("⚠️ Knowledge base index not found. Please rebuild the KB.")
        return None, None

    try:
        index = faiss.read_index(INDEX_PATH)
        print("✅ FAISS index loaded successfully.")

        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"✅ Metadata loaded successfully. Found {len(meta)} chunks.")

        return index, meta

    except Exception as e:
        print(f"❌ Failed to load FAISS index or metadata: {type(e).__name__}: {e}")
        return None, None



# --------------------------------------------------------
# 🔍  relevant manual chunks via FAISS
# --------------------------------------------------------
def _faiss_context(query: str, top_k: int = 3):
    """
    Retrieve the top_k most relevant manual chunks from the FAISS index
    given a user query. Uses OpenAI embeddings for similarity search.
    """
    index, meta = load_faiss_index()
    if not index or not meta:
        return []

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error("⚠️ Missing OPENAI_API_KEY — please set it in environment or Streamlit secrets.")
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
        return results

    except Exception as e:
        st.warning(f"⚠️ Error retrieving FAISS context: {type(e).__name__}: {e}")
        return []


# --------------------------------------------------------
# 🧱 Rebuild knowledge base from PDF
# --------------------------------------------------------
def rebuild_knowledge_base():
    """
    Extract text + diagrams from Vaillant manual and rebuild FAISS index.
    """
    st.info("📘 Reading Vaillant manual...")

    pdf_path = "8000014609_03.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"❌ PDF not found at {pdf_path}. Please upload or include it in your repo.")
        return

    # --- Load PDF ---
    try:
        reader = PdfReader(pdf_path)
        doc = fitz.open(pdf_path)
    except Exception as e:
        st.error(f"❌ Could not open PDF: {e}")
        return

    items = []
    page_texts = {}
    page_images = {}

    def chunk_text(text: str, max_len: int = 700):
        """Split text into roughly sentence-based chunks."""
        sentences = text.split(". ")
        chunks, cur, cur_len = [], [], 0
        for s in sentences:
            cur.append(s)
            cur_len += len(s.split())
            if cur_len > max_len:
                chunks.append(". ".join(cur))
                cur, cur_len = [], 0
        if cur:
            chunks.append(". ".join(cur))
        return chunks

    # --- Extract text ---
    st.info("📄 Extracting text from manual...")
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
            if text and len(text.strip()) > 20:
                page_texts[page_num] = chunk_text(text.strip())
        except Exception as e:
            st.warning(f"⚠️ Could not extract text from page {page_num}: {e}")

    # --- Extract images ---
    st.info("🖼️ Extracting diagrams...")
    for page_num, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                if base_image.get("width", 0) < 150 or base_image.get("height", 0) < 150:
                    continue
                img_dir = os.path.join("kb", "diagrams")
                os.makedirs(img_dir, exist_ok=True)
                img_filename = f"page_{page_num}_img_{img_index+1}.{ext}"
                abs_path = os.path.join(img_dir, img_filename)
                with open(abs_path, "wb") as f:
                    f.write(image_bytes)
                rel_path = os.path.join("kb", "diagrams", img_filename)
                page_images.setdefault(page_num, []).append(rel_path)
            except Exception as e:
                st.warning(f"⚠️ Could not extract image on page {page_num}: {e}")

    # --- Combine text + images ---
    st.info("🔗 Combining text + images...")
    for page_num, chunks in page_texts.items():
        imgs = page_images.get(page_num, [])
        for chunk in chunks:
            items.append({
                "page": page_num,
                "text": chunk,
                "type": "joint",
                "image_paths": imgs
            })

    # --- Generate embeddings ---
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not set — cannot generate embeddings.")
        return

    st.info(f"⚙️ Generating embeddings for {len(items)} chunks...")
    embeddings = []
    client = OpenAI(api_key=api_key)
    for i, it in enumerate(items, 1):
        try:
            emb_resp = client.embeddings.create(
                model="text-embedding-3-large",
                input=it["text"]
            )
            emb = np.array(emb_resp.data[0].embedding, dtype="float32")
            embeddings.append(emb)
        except Exception as e:
            st.warning(f"⚠️ Skipped chunk {i}: {e}")
            embeddings.append(np.zeros((3072,), dtype="float32"))
        if i % 10 == 0:
            st.write(f"→ Embedded {i}/{len(items)}")

    emb_matrix = np.vstack(embeddings)
    faiss.normalize_L2(emb_matrix)
    index = faiss.IndexFlatIP(emb_matrix.shape[1])
    index.add(emb_matrix)

    # --- Save index + metadata ---
    kb_dir = "kb"
    os.makedirs(kb_dir, exist_ok=True)
    INDEX_PATH = os.path.join(kb_dir, "vaillant_joint_faiss.index")
    META_PATH = os.path.join(kb_dir, "vaillant_joint_meta.json")

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    st.success(f"✅ Knowledge base built with {len(items)} text chunks and {sum(len(v) for v in page_images.values())} images.")



    st.title("Vaillant 150 L Cylinder Model Simulation")
    # ... rest of your simulation + charts + AI chat code ...


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

# --- Optional AI Assistant ---
try:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    client = OpenAI(api_key=api_key) if api_key else None
except Exception:
    client = None

import numpy as np

# ---------- PHYSICAL CONSTANTS ----------

RHO = 1000.0      # kg/m³
C_P = 4186.0      # J/kgK
P_EL_MAX = 5000.0 # W nominal

# 150 L, 4-layer geometry (your values)
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
    # crude COP model vs tank temp
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
):
    """
    4-layer stratified tank (150 L fixed) with defined layer geometry and HP behavior.
    """
    # --- Validate tapping input ---
    for key in ["time", "volume", "rate_lpm"]:
        if key not in tap:
            raise ValueError(f"tap dict missing '{key}'")

    tap_time = np.array(tap["time"])
    tap_vol = np.array(tap["volume"])
    tap_rate = np.array(tap["rate_lpm"])

    # --- Time discretisation ---
    N_layers = 4
    dt_s = dt_min * 60.0
    steps = int(sim_hrs * 60.0 / dt_min)
    time_h = np.arange(0, steps * dt_min, dt_min) / 60.0

    # --- Geometry setup ---
    def mm2_to_m2(x): return x / 1e6
    layer_order = ["Bottom Layer", "Lower-Mid Layer", "Upper-Mid Layer", "Top Layer"]

    V_layers_m3 = np.array([BASE_LAYER_PROPERTIES[n]["Volume_L"] / 1000.0 for n in layer_order])
    A_ext = np.array([mm2_to_m2(BASE_LAYER_PROPERTIES[n]["Water_External_Surface_mm2"]) for n in layer_order])
    A_int = np.array([mm2_to_m2(BASE_LAYER_PROPERTIES[n]["Water_Layer_Surface_mm2"]) for n in layer_order])

    m_layer = RHO * V_layers_m3
    UA_loss_layer = Utank * A_ext
    U_int = 50.0
    UA_int_layer = U_int * A_int

    # --- HP control parameters ---
    Pmax_W = P_EL_MAX * Pmax
    Pmin_W = Pmax_W * Pmin
    Ton = setp - hyst
    coil_bottom_idx = 2  # lower two layers receive HP heat

    # --- Allocate arrays ---
    T = np.zeros((steps, N_layers))
    Qhp = np.zeros(steps)
    Pel = np.zeros(steps)
    COPs = np.zeros(steps)
    HP_on = np.zeros(steps, bool)
    Qloss_total = np.zeros(steps)
    Tap_flow = np.zeros(steps)
    Tap_temp = np.zeros(steps)
    Mod_frac = np.zeros(steps)

    # --- Initial condition: temperature gradient ---
    T_bottom_init = setp - 15.0
    T_top_init = setp
    T[0, :] = np.linspace(T_bottom_init, T_top_init, N_layers)

    # --- Main simulation loop ---
    for k in range(1, steps):
        T_prev = T[k - 1, :].copy()
        T_new = T_prev.copy()
        T_top = T_prev[-1]
        T_bottom = T_prev[0]
        tnow_min = k * dt_min

        # --- HP modulation control ---
        mod = 0.0
        on = False
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

        # --- Tapping: draw from top, refill at bottom ---
        idx = np.where((tap_time > tnow_min - dt_min) & (tap_time <= tnow_min))[0]
        Vtap_L = np.sum(tap_vol[idx]) if idx.size > 0 else 0.0
        mdot_tap = (Vtap_L / 1000.0) * RHO / dt_s
        Tap_flow[k] = np.sum(tap_rate[idx]) if idx.size > 0 else 0.0
        Tap_temp[k] = T_top

        # --- Layer energy balance ---
        for i in range(N_layers):
            Q_net = 0.0

            # conduction between layers
            if i > 0:
                Q_net += UA_int_layer[i] * (T_prev[i - 1] - T_prev[i])
            if i < N_layers - 1:
                Q_net += UA_int_layer[i] * (T_prev[i + 1] - T_prev[i])

            # ambient losses
            Q_loss = UA_loss_layer[i] * (T_prev[i] - Tamb)
            Q_net -= Q_loss
            Qloss_total[k] += Q_loss

            # HP input to lower layers
            if i < coil_bottom_idx and Qhp[k] > 0:
                Q_net += Qhp[k] / coil_bottom_idx

            # tapping effect
            if i == N_layers - 1:
                Q_net -= mdot_tap * C_P * T_prev[i]
            elif i == 0:
                Q_net += mdot_tap * C_P * Tamb

            # temperature update
            dT = Q_net * dt_s / (m_layer[i] * C_P)
            T_new[i] = np.clip(T_prev[i] + dT, 0.0, 100.0)

        # enforce stratification stability
        for i in range(N_layers - 1):
            if T_new[i] > T_new[i + 1]:
                avg = 0.5 * (T_new[i] + T_new[i + 1])
                T_new[i] = avg
                T_new[i + 1] = avg

        T[k, :] = T_new

    # --- Build results dataframe ---
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
    })

    for i, name in enumerate(layer_order):
        df[f"T_{name} (°C)"] = T[:, i]
    df["T_Avg (°C)"] = T.mean(axis=1)

    # --- Energy summaries ---
    total_heat_kWh = df["HP Heat (W)"].sum() * dt_s / 3.6e6
    total_power_kWh = df["HP Power (W)"].sum() * dt_s / 3.6e6
    total_losses_kWh = df["Q_Loss (W)"].sum() * dt_s / 3.6e6
    hp_runtime_min = int(df["HP_On"].sum() * dt_min)
    avg_cop = total_heat_kWh / total_power_kWh if total_power_kWh > 0 else 0.0

    summary = {
        "Simulation Hours": sim_hrs,
        "Tank Volume (L)": 150.0,
        "Total HP Electrical Energy (kWh)": total_power_kWh,
        "Total Heat Delivered by HP (kWh)": total_heat_kWh,
        "Total Losses (kWh)": total_losses_kWh,
        "HP Run Time (minutes)": hp_runtime_min,
        "Average COP": avg_cop,
    }

    # ✅ return is correctly indented INSIDE the function
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
        range=["#dc2626", "#f97316", "#60a5fa", "#1e3a8a"],   # lower-mid is light blue
    )

    return (
        alt.Chart(d)
        .mark_line()
        .encode(
            x="Time (h)",
            y="Temp (°C)",
            color=alt.Color("Layer:N", scale=color_scale),
        )
        .properties(height=400, title="Tank Stratification (4 Layers)")
        .interactive()
    )


def chart_power(df: pd.DataFrame):
    return (
        alt.Chart(df)
        .mark_line(color="#ef4444")
        .encode(x="Time (h)", y="HP Power (W)")
        .properties(height=300, title="Heat Pump Electrical Power (W)")
    )


def chart_heat(df: pd.DataFrame):
    return (
        alt.Chart(df)
        .mark_line(color="#1d4ed8")
        .encode(x="Time (h)", y="HP Heat (W)")
        .properties(height=300, title="Heat Pump Heat Output (W)")
    )


def chart_cop(df: pd.DataFrame):
    return (
        alt.Chart(df)
        .mark_line(color="green")
        .encode(x="Time (h)", y="COP")
        .properties(height=300, title="Coefficient of Performance (COP)")
    )


def chart_modulation(df: pd.DataFrame):
    base = alt.Chart(df).encode(x="Time (h)")
    mod = base.mark_area(opacity=0.6, color="#2563eb").encode(y="Modulation:Q")
    hp_on = base.mark_line(color="orange").encode(y="HP_On:Q")
    return (mod + hp_on).properties(height=300, title="HP Modulation & On/Off State")


def chart_tap(df: pd.DataFrame):
    return (
        alt.Chart(df)
        .mark_bar(color="#f59e0b")
        .encode(x="Time (h)", y="Tap Flow (L/min)")
        .properties(height=250, title="Tap Flow (L/min)")
    )


# ---------- SIMPLE RAG HELPERS (TEXT SIMILARITY) ----------

def load_kb_items(kb_path: str = r"C:\Users\james\OneDrive\Documents\Python\kb\vaillant_manual_kb.json") -> List[Dict[str, Any]]:
    """Load KB JSON with fields: page, text, image_paths."""
    if not os.path.exists(kb_path):
        return []
    with open(kb_path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_relevant_chunks(query: str, items: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """Very simple RAG: rank by SequenceMatcher string similarity."""
    if not items:
        return []
    scores = []
    q = query.lower()
    for it in items:
        txt = it.get("text", "")
        if not txt:
            continue
        score = SequenceMatcher(None, q, txt.lower()).ratio()
        scores.append((score, it))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [it for (s, it) in scores[:top_k] if s > 0.1]


def main():
    def main():
        st.title("Vaillant 150 L Cylinder Model Simulation")

    def main():
        st.title("Vaillant 150 L Cylinder Model Simulation")

    # ==============================================================
    # 🧩 SIDEBAR — Simulation Parameters
    # ==============================================================
    with st.sidebar:
        st.header("⚙️ Simulation Parameters")

        # --- Tank & Environment ---
        st.markdown("### Tank and Environment")
        st.write("Tank Volume (L): **150 L (fixed)**")

        Utank = st.slider("Tank U-value (W/m²K)", 0.1, 2.0, 0.4, 0.1, key="param_utank")
        Tamb = st.slider("Ambient Temp (°C)", 0, 30, 15, key="param_tamb")
        simh = st.slider("Simulation Duration (h)", 1, 48, 24, key="param_simh")
        dtm = st.slider("Time Step (min)", 1, 10, 5, key="param_dtm")

        # --- Heat Pump Control ---
        st.markdown("### Heat Pump Control")
        setp = st.slider("Setpoint (°C)", 40, 90, 50, 1, key="param_setp")
        hyst = st.slider("Hysteresis (°C)", 1.0, 10.0, 5.0, 0.5, key="param_hyst")
        Pmax = st.slider("Max HP Power Ratio", 0.5, 1.0, 1.0, 0.05, key="param_pmax")
        Pmin = st.slider("Min Modulation Ratio", 0.1, 0.5, 0.2, 0.05, key="param_pmin")

        st.markdown("---")
        st.markdown("### 📘 Knowledge Base")
        if st.button("🔁 Rebuild Knowledge Base (PDF → FAISS)", key="btn_rebuild_kb"):
            with st.spinner("Rebuilding knowledge base..."):
                rebuild_knowledge_base()
            st.success("✅ Knowledge base successfully rebuilt!")

    # ==============================================================
    # 🚿 Domestic Hot Water Tapping Schedule
    # ==============================================================
    raw_taps = [
       {"Hour": 7, "Minute": 0, "Energy_kWh": 0.105,"Flow_lpm": 3,"Duration_sec": 45, "T_inlet": 10,"T_outlet": 50},
        {"Hour": 7, "Minute": 5, "Energy_kWh": 1.4, "Flow_lpm": 6, "Duration_sec": 300, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 7, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10,"T_outlet": 50},
        {"Hour": 8, "Minute": 1, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 8, "Minute": 15, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
         {"Hour": 8, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 8, "Minute": 45, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 9, "Minute": 00, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 9, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 10, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 11, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 11, "Minute": 45, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 12, "Minute": 45, "Energy_kWh": 0.315, "Flow_lpm": 4, "Duration_sec": 101.25, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 14, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 15, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 16, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 18, "Minute": 00, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 18, "Minute": 15, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 18, "Minute": 30, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 19, "Minute": 00, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 20, "Minute": 30, "Energy_kWh": 0.735, "Flow_lpm": 4, "Duration_sec": 236.25, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 21, "Minute": 15, "Energy_kWh": 0.105, "Flow_lpm": 3, "Duration_sec": 45, "T_inlet": 10, "T_outlet": 50},
        {"Hour": 21, "Minute": 30, "Energy_kWh": 1.4, "Flow_lpm": 6, "Duration_sec": 300, "T_inlet": 10, "T_outlet": 50},
    ]
    tap = {"time": [], "volume": [], "rate_lpm": []}
    for e in raw_taps:
        tmin = e["Hour"] * 60 + e["Minute"]
        vol_L = e["Flow_lpm"] * e["Duration_sec"] / 60.0
        tap["time"].append(tmin)
        tap["volume"].append(vol_L)
        tap["rate_lpm"].append(e["Flow_lpm"])

    # ==============================================================
    # ⚙️ Simulation Auto-Run Logic
    # ==============================================================
    params = {
        "dtm": dtm,
        "simh": simh,
        "Utank": Utank,
        "Tamb": Tamb,
        "setp": setp,
        "hyst": hyst,
        "Pmin": Pmin,
        "Pmax": Pmax,
    }
    param_key = tuple(params.values())

    # Detect parameter changes
    param_changed = (
        "last_params" not in st.session_state
        or st.session_state["last_params"] != param_key
    )

    # --- Run simulation if parameters changed ---
    if param_changed:
        with st.spinner("🔁 Running simulation..."):
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
        for key, val in summary.items():
            st.write(f"**{key}:** {val:.2f}" if isinstance(val, float) else f"**{key}:** {val}")

        st.markdown("---")
        st.subheader("📈 Graph Viewer")
        choice = st.selectbox(
            "Select a graph:",
            [
                "Coefficient of Performance (COP)",
                "Tank Stratification (Multi-Layer)",
                "Heat Pump Power (W)",
                "Heat Pump Heat Output (W)",
                "HP Modulation & On/Off State",
                "Tap Flow (L/min)",
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

    # ==============================================================
# 💬 AI Simulation Assistant (Self-Learning Chatbot)
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
# 💬 Learning Chatbot Integration
# -----------------------------------------------------
if query:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("⚠️ Please set your OpenAI API key.")
        st.stop()

    client = OpenAI(api_key=api_key)
    layer_properties = st.session_state["layer_properties"]
    summary = st.session_state["summary"]

    # === Generate multiple candidate responses ===
    responses = []
    for _ in range(3):
        kb_context = ""
        if data_mode.startswith("Manual"):
            top_items = retrieve_faiss_context(query, top_k=3)
       # --- Retrieve top chunks from FAISS index ---
top_items = retrieve_faiss_context(query, top_k=3)

# --- Display retrieved chunks (text + images) ---
if top_items:
    st.markdown("### 📘 Retrieved Manual Context")
    for it in top_items:
        st.markdown(f"**Page {it.get('page','?')}** — similarity: {it.get('similarity',0):.2f}")
        st.write(it.get("text", "")[:800] + "...")
        for img_path in it.get("image_paths", []):
            if os.path.exists(img_path):
                st.image(img_path, width=250)
            else:
                st.caption(f"⚠️ Image not found: {img_path}")
        st.markdown("---")

        cols = st.columns(len(it.get("image_paths", [])))
for col, img_path in zip(cols, it.get("image_paths", [])):
    if os.path.exists(img_path):
        col.image(img_path, use_column_width=True)



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
"""
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            responses.append(r.choices[0].message.content.strip())
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            st.stop()

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
        st.caption(f"🧠 ML model ranked responses (confidence {probs[best_idx]:.2f})")
    else:
        best_idx = 0
        st.caption("⚙️ No trained model yet — showing first GPT answer")

    best_response = responses[best_idx]

    # === Show final answer ===
    st.markdown("### 💬 Chatbot Response")
    st.write(best_response)

    # === Feedback section ===
    feedback = st.radio(
        "Was this answer helpful?",
        ["👍 Yes", "👎 No"],
        horizontal=True,
        key=f"feedback_{query}"
    )
    if st.button("💾 Save Feedback", key=f"save_{query}"):
        new_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": query,
            "response": best_response,
            "helpful": feedback
        }
        feedback_df = pd.concat([feedback_df, pd.DataFrame([new_row])], ignore_index=True)
        feedback_df.to_csv(FEEDBACK_FILE, index=False)
        st.success("✅ Feedback saved!")

        # Retrain if threshold met
        if len(feedback_df) % RETRAIN_THRESHOLD == 0:
            with st.spinner("🔁 Retraining ML model..."):
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























