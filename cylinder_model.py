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

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib.units import cm
import tempfile
import io
from datetime import datetime  # you already import this further down, keep only one
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except Exception as e:
    REPORTLAB_AVAILABLE = False
    print("⚠️ reportlab not available:", e)

try:
    import altair_saver  # uses vl-convert under the hood
    ALTAIR_EXPORT_AVAILABLE = True
except Exception as e:
    ALTAIR_EXPORT_AVAILABLE = False
    print("⚠️ altair_saver not available (Altair → PNG export disabled):", e)
import altair as alt
alt.data_transformers.disable_max_rows()
import altair as alt

try:
    alt.renderers.enable("vl-convert")
except:
    alt.renderers.enable("default")


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

# Path to Vaillant COP chart (must sit next to this .py file)
COP_EXCEL_PATH = "aroTHERM plus Heating Chart - Copy.xlsx"

@st.cache_resource
def load_arotherm_cop_data_from_excel(path: str) -> Dict[str, Dict[float, Dict[float, float]]]:
    """
    Build:
        { "7 kW": { ODT: { T_flow: COP, ... }, ... }, ... }

    from the official 'aroTHERM plus Heating Chart - Copy.xlsx'.

    Assumes each sheet looks like the Vaillant chart:
      - sheet names: 'aroTHERM plus 3.5kW', 'aroTHERM plus 5kW', ...
      - data block starts a few rows down
      - columns:
            col 1 : ODT (°C)
            35°C: COP in col 4
            45°C: COP in col 7
            55°C: COP in col 10
    """
    xls = pd.ExcelFile(path)
    model_sheet_map = {
        "3.5 kW": "aroTHERM plus 3.5kW",
        "5 kW": "aroTHERM plus 5kW",
        "7 kW": "aroTHERM plus 7kW",
        "10 kW": "aroTHERM plus 10kW",
        "12 kW": "aroTHERM plus 12kW",
    }

    result: Dict[str, Dict[float, Dict[float, float]]] = {}

    for model, sheet in model_sheet_map.items():
        df = pd.read_excel(xls, sheet_name=sheet)

        # From inspection: numeric rows start around index 5
        block = df.iloc[5:]

        model_table: Dict[float, Dict[float, float]] = {}

        for _, row in block.iterrows():
            odt = row.iloc[1]  # second column is ODT
            if not isinstance(odt, (int, float)) or pd.isna(odt):
                continue
            odt_val = float(odt)

            # (T_flow, column_index_for_COP)
            cop_cols = [(35.0, 4), (45.0, 7), (55.0, 10)]
            sink_map: Dict[float, float] = {}

            for Tflow, idx in cop_cols:
                if idx >= len(row):
                    continue
                val = row.iloc[idx]
                if isinstance(val, (int, float)) and not pd.isna(val):
                    sink_map[float(Tflow)] = float(val)

            if sink_map:
                model_table[odt_val] = sink_map

        # sort by ODT
        sorted_table = {k: model_table[k] for k in sorted(model_table.keys())}
        result[model] = sorted_table

    return result


# Load once and reuse
AROTHERM_COP_DATA = load_arotherm_cop_data_from_excel(COP_EXCEL_PATH)



def _interp_cop_from_table(hp_model: str, T_source: float, T_sink_flow: float):
    """
    Bilinear interpolation in (ODT, T_flow) using AROTHERM_COP_DATA.
    T_sink_flow = the FLOW temperature (°C) used for COP look-up.
    """
    model_data = AROTHERM_COP_DATA.get(hp_model)
    if not model_data:
        return 3.0

    # --- Clamp or bracket source (ODT) temperature ---
    src_points = sorted(model_data.keys())
    if T_source <= src_points[0]:
        Ts_low = Ts_high = src_points[0]
    elif T_source >= src_points[-1]:
        Ts_low = Ts_high = src_points[-1]
    else:
        Ts_low  = max(t for t in src_points if t <= T_source)
        Ts_high = min(t for t in src_points if t >= T_source)

    # Helper: interpolate COP vs flow temperature at a given ODT
    def cop_at(odt_val):
        sink_data = model_data[odt_val]          # {35:cop, 45:cop, 55:cop}
        sink_points = sorted(sink_data.keys())   # [35,45,55]

        # --- clamp or bracket flow temp ---
        if T_sink_flow <= sink_points[0]:
            Tl = Th = sink_points[0]
        elif T_sink_flow >= sink_points[-1]:
            Tl = Th = sink_points[-1]
        else:
            Tl = max(t for t in sink_points if t <= T_sink_flow)
            Th = min(t for t in sink_points if t >= T_sink_flow)

        if Tl == Th:
            return sink_data[Tl]

        # Linear interpolation between Tl and Th
        f = (T_sink_flow - Tl) / (Th - Tl)
        return sink_data[Tl] + f * (sink_data[Th] - sink_data[Tl])

    # --- Interpolate in ODT dimension ---
    COP_low = cop_at(Ts_low)

    if Ts_low == Ts_high:
        return float(np.clip(COP_low, 1.5, 7.0))

    COP_high = cop_at(Ts_high)
    f_src = (T_source - Ts_low) / (Ts_high - Ts_low)
    COP = COP_low + f_src * (COP_high - COP_low)

    return float(np.clip(COP, 1.5, 7.0))




# ---------- HEAT PUMP MODEL ----------
def solve_hp_arotherm(
    T_sink: float,
    T_source: float,
    mod: float,
    hp_model: str,
    Pmax_ratio: float,
    Pmin_ratio: float,
    quiet_mode: bool = False,
    defrost_enabled: bool = True,
):
    """
    Vaillant aroTHERM plus solver:
    - Uses real COP tables (interpolated)
    - Applies modulation limits
    - Optional quiet mode derating
    - Optional defrost penalty
    """

    # ---------- 1) Electrical input from modulation ----------
    Pel_max = P_EL_MAX * Pmax_ratio
    Pel_min = Pel_max * Pmin_ratio

    if mod <= 0 or mod < Pmin_ratio:
        return 0.0, 0.0, 0.0, {"state": "off"}

    # scale modulation between Pel_min .. Pel_max
    frac = (mod - Pmin_ratio) / (1.0 - Pmin_ratio)
    Pel = Pel_min + frac * (Pel_max - Pel_min)

    # ---------- 2) Quiet-mode derating ----------
    if quiet_mode:
        Pel *= 0.80  # ~20% derate typical for aroTHERM Q mode

    # ---------- 3) Interpolate COP from Vaillant tables ----------
    COP_raw = _interp_cop_from_table(hp_model, T_source, T_sink)

    COP = COP_raw

    # ---------- 4) Apply DEFROST penalty ----------
    if defrost_enabled and T_source < 5:
        # typical COP penalty curve for 0…5°C
        penalty = np.clip((5 - T_source) * 0.05, 0.0, 0.25)
        COP *= (1 - penalty)

    # Safety
    COP = float(np.clip(COP, 1.0, 7.0))

    # ---------- 5) Heat output ----------
    Q = Pel * COP

    meta = {
        "COP_table": COP_raw,
        "COP_final": COP,
        "Pel": Pel,
        "T_sink": T_sink,
        "T_source": T_source,
        "quiet": quiet_mode,
    }

    return Q, Pel, COP, meta



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
    buffer_volume_L: float = 0,
    buffer_temp_init: float = 45,
    immersion_enabled: bool = False,
    immersion_power_kw: float = 0.0,
    boost_cylinder: bool = False,
    pump_flow_lpm: float = 0,
    mixer_enabled: bool = False,
    flow_temp_before_mixer: float = 60,
    flow_temp_after_mixer: float = 45,
    heating_circuit_flow_lpm: float = 20,
    hp_model_name: str = "7 kW",
    quiet_mode_enabled: bool = False,
    defrost_enabled: bool = True,
    
    night_mode: str = "Eco",           # "Eco" or "Normal" (Modo noche)
    absence_mode: bool = False,        # Ausencia
    absence_temp: float = 15.0,        # Temperatura de ausencia
    frost_protection_enabled: bool = True,
    frost_protection_temp: float = 5.0,

):
    """
    Clean, working 4-layer stratified cylinder simulation with:
      - aroTHERM COP tables
      - Adaptive timestep during tapping
      - SensoComfort / Legionella logic (setpoint override)
      - Optional buffer tank (simplified but UI-compatible)
    Returns:
        df, summary, BASE_LAYER_PROPERTIES
    """

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
    # TIME STEP SETUP
    # ===============================
    if adaptive_timestep:
        dt_base = dt_min * 60.0        # [s]
        dt_tap = max(dt_base / 3.0, 60.0)
    else:
        dt_base = dt_min * 60.0
        dt_tap = dt_base

    sim_end_s = sim_hrs * 3600.0

    # ===============================
    # GEOMETRY SETUP
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

    m_layer = RHO * V_layers_m3                # kg
    UA_loss_layer = Utank * A_ext              # W/K
    U_int = 50.0                               # W/m²K (internal vertical exchange)
    UA_int_layer = U_int * A_int               # W/K

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

    m_layer = RHO * V_layers_m3                # kg
    UA_loss_layer = Utank * A_ext              # W/K
    U_int = 50.0                               # W/m²K (internal vertical exchange)
    UA_int_layer = U_int * A_int               # W/K

    # === Stratification-aware mapping of real ports to layers ===
    layer_fracs = V_layers_m3 / V_layers_m3.sum()
    boundaries = np.concatenate(([0.0], np.cumsum(layer_fracs)))  # 0..1

    def height_to_layer_index(z_mm, H_tot_mm=1000.0):
        """Return the layer index [0..N_layers-1] containing a port at height z (mm)."""
        z_frac = z_mm / H_tot_mm
        return int(np.clip(np.searchsorted(boundaries, z_frac, side="right") - 1,
                           0, N_layers - 1))

    # === Vaillant 150 L uniSTOR ports (approx, mm from bottom) ===
    Z_DHW_OUTLET   = 675.0  # port 7
    Z_COIL_TOP     = 520.0  # port 2: HP heating flow
    Z_COIL_BOTTOM  = 393.0  # port 1/5: HP return
    Z_COLD_INLET   = 154.0  # cold feed/drain region

    # Map heights to layer indices
    layer_cold_in   = height_to_layer_index(Z_COLD_INLET)
    layer_dhw_out   = height_to_layer_index(Z_DHW_OUTLET)
    layer_coil_low  = height_to_layer_index(Z_COIL_BOTTOM)
    layer_coil_high = height_to_layer_index(Z_COIL_TOP)

    # All layers spanned by the coil
    coil_layers = np.arange(layer_coil_low, layer_coil_high + 1, dtype=int)
    # Representative layer for COP sink temperature (use upper part of coil)
    coil_sink_layer = int(coil_layers[-1])



    # === Vaillant 150 L uniSTOR ports (approx) ===
    Z_DHW_OUTLET   = 675.0  # mm  (port 7)
    Z_COIL_TOP     = 520.0  # mm  (port 2: heating flow from HP)
    Z_COIL_BOTTOM  = 393.0  # mm  (port 1 or 5: cylinder heating return/inlet)
    Z_COLD_INLET   = 154.0  # mm  (bottom cold feed / drain region, from drawing


    # ===============================
    # BUFFER SETUP (simplified 4-layer)
    # ===============================
    N_buf = 4
    buffer_layers = None
    buffer_mass = 0.0
    if buffer_enabled and buffer_volume_L > 0:
        total_m3 = buffer_volume_L / 1000.0
        buffer_mass = total_m3 * BUFFER_RHO
        layer_mass_buf = buffer_mass / N_buf
        # Allocate large array; will trim later
        max_steps = int(sim_hrs * 3600.0 / (dt_min * 60.0)) * 3 + 10
        buffer_layers = np.zeros((max_steps, N_buf), dtype=float)
        buffer_layers[0, :] = buffer_temp_init
    else:
        layer_mass_buf = 0.0

    # ===============================
    # STATE ARRAYS (DYNAMIC LISTS)
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
    Mixer_flow_before_array = []
    Mixer_flow_after_array = []
    Mixer_return_temp_array = []
    Buffer_pump_flow_array = []
    HP_T_sink_array = []
    HP_T_source_array = []
    Frost_active_array = []

    # ===============================
    # INITIAL CONDITIONS
    # ===============================
    T_prev = np.full(N_layers, initial_tank_temp, dtype=float)

    tnow_s = 0.0
    step_count = 0
    hp_on_prev = False

    # Legionella tracking
    legionella_cycle_start_min = None
    legionella_in_progress = False

    # ===============================
    # MAIN SIMULATION LOOP
    # ===============================
    while tnow_s < sim_end_s:
        tnow_min = tnow_s / 60.0
        tnow_hrs = tnow_s / 3600.0

        # ---------------------------------
        # PICK ADAPTIVE TIME STEP
        # ---------------------------------
        tap_check_end = tnow_min + (dt_base / 60.0)
        idx_check = np.where((tap_time >= tnow_min) & (tap_time < tap_check_end))[0]
        is_tapping_soon = len(idx_check) > 0

        if adaptive_timestep and is_tapping_soon:
            dt_s = dt_tap
        else:
            dt_s = dt_base
        dt_current_min = dt_s / 60.0

        # Clamp end of sim
        if tnow_s + dt_s > sim_end_s:
            dt_s = sim_end_s - tnow_s
            dt_current_min = dt_s / 60.0

        # ---------------------------------
        # SENSOCOMFORT / SETPOINT LOGIC
        # ---------------------------------
        active_setpoint = setp
        current_mode = "Standard"

        hour_of_day = tnow_hrs % 24.0

        if senso_enabled:
            if holiday_mode:
                # Holiday mode: fixed holiday temperature
                active_setpoint = holiday_temp
                current_mode = "Holiday"

            elif absence_mode:
                # Ausencia: reduced temperature, DHW / circulation conceptually off
                active_setpoint = absence_temp
                current_mode = "Absence"

            elif boost_mode:
                # Quick heat-up – raise comfort temp a bit but respect 65 °C cap
                active_setpoint = min(comfort_temp + 5.0, 65.0)
                current_mode = "Boost"

            elif enable_time_program:
                in_morning = morning_start <= hour_of_day < morning_end
                in_evening = evening_start <= hour_of_day < evening_end

                if in_morning or in_evening:
                    # Comfort during programmed periods
                    active_setpoint = comfort_temp
                    current_mode = "Comfort"
                else:
                    # Behaviour outside time-program → Modo noche
                    if night_mode == "Eco":
                        # Eco: heating off, only frost protection (set to 0, overridden later)
                        active_setpoint = 0.0
                        current_mode = "Night-Eco"
                    else:
                        # Normal: reduced “Ausencia” temperature
                        active_setpoint = absence_temp
                        current_mode = "Night-Normal"
            else:
                # No time program, just manual Comfort / ECO / Auto as before
                if comfort_mode == "ECO":
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

        # ---------------------------------
        # FROST PROTECTION (anti-freeze)
        # Approximation of sensoCOMFORT frost function:
        # if outside < ~4 °C, keep at least frost_protection_temp
        # ---------------------------------
        frost_active = 0.0
        if frost_protection_enabled and Tamb < 4.0:
            if active_setpoint < frost_protection_temp:
                active_setpoint = frost_protection_temp
            frost_active = 1.0
            # mark mode as frost if we were “off” / eco
            if current_mode in ["Standard", "Night-Eco", "Holiday", "Absence"]:
                current_mode = "FrostProt"


        # ---------------------------------
        # LEGIONELLA LOGIC (OVERRIDES SETPOINT)
        # ---------------------------------
        legionella_active = 0.0
        if legionella_enabled:
            current_day = int(tnow_hrs / 24.0)
            hour_of_day = tnow_hrs % 24.0

            # Start of a legionella day at configured hour
            if (
                current_day % int(legionella_frequency) == 0
                and abs(hour_of_day - legionella_start_hour) < dt_current_min / 60.0 + 1e-6
                and not legionella_in_progress
            ):
                legionella_cycle_start_min = tnow_min
                legionella_in_progress = True

            if legionella_in_progress:
                elapsed_min = tnow_min - legionella_cycle_start_min
                if elapsed_min < legionella_duration:
                    active_setpoint = legionella_temp
                    legionella_active = 1.0
                else:
                    legionella_in_progress = False

        # ---------------------------------
        # TAPPING IN CURRENT STEP
        # ---------------------------------
        idx = np.where((tap_time >= tnow_min) & (tap_time < tnow_min + dt_current_min))[0]
        Vtap_L = np.sum(tap_vol[idx]) if idx.size > 0 else 0.0
        tap_rate_current = np.sum(tap_rate[idx]) if idx.size > 0 else 0.0

        tap_duration_s = 0.0
        if len(idx) > 0:
            for i_idx in idx:
                if tap_rate[i_idx] > 0:
                    duration = (tap_vol[i_idx] / tap_rate[i_idx]) * 60.0  # sec
                    tap_duration_s += duration

        if tap_duration_s > 0:
            mdot_tap = (Vtap_L / 1000.0) * RHO / tap_duration_s   # kg/s
        else:
            mdot_tap = 0.0

        tap_active = tap_duration_s > 0

        # ---------------------------------
        # HEAT PUMP CONTROL (HYSTERESIS + MOD)
        # ---------------------------------
        T_top = T_prev[-1]

        # ON/OFF with hysteresis
        if T_top < active_setpoint - hyst / 2.0:
            hp_on = True
        elif T_top > active_setpoint + hyst / 2.0:
            hp_on = False
        else:
            hp_on = hp_on_prev

        # Modulation proportional to error
        if hp_on:
            temp_error = max(0.0, active_setpoint - T_top)
            mod = Pmin + (1.0 - Pmin) * np.clip(temp_error / 10.0, 0.0, 1.0)
        else:
            mod = 0.0

        # Default: HP sink flow based on upper coil layer + 5 K
        T_flow_for_COP = float(T_prev[coil_sink_layer] + 5.0)
        T_source_for_HP = float(Tsrc)


        # ---------------------------------
        # CALL aroTHERM HP MODEL
        # ---------------------------------
        if hp_on and mod > 0:
            Q_hp, Pel_hp, COP_hp, _meta = solve_hp_arotherm(
                T_sink=T_flow_for_COP,
                T_source=T_source_for_HP,
                mod=mod,
                hp_model=hp_model_name,
                Pmax_ratio=Pmax,
                Pmin_ratio=Pmin,
                quiet_mode=quiet_mode_enabled,
                defrost_enabled=defrost_enabled,
            )
        else:
            Q_hp, Pel_hp, COP_hp = 0.0, 0.0, 0.0

        # ---------------------------------
        # CYLINDER LAYER ENERGY BALANCE
        # ---------------------------------
        T_new = T_prev.copy()
        Q_loss_total = 0.0

        for i in range(N_layers):
            Q_net = 0.0

            # Vertical conduction between layers
            if i > 0:
                Q_net += UA_int_layer[i] * (T_prev[i - 1] - T_prev[i])
            if i < N_layers - 1:
                Q_net += UA_int_layer[i] * (T_prev[i + 1] - T_prev[i])

            # Loss to ambient
            Q_loss = UA_loss_layer[i] * (T_prev[i] - Tamb)
            Q_net -= Q_loss
            Q_loss_total += Q_loss

            # HP injection into all layers spanned by the coil
            if hp_on and i in coil_layers and Q_hp > 0:
                Q_net += Q_hp / len(coil_layers)


            # hot draw-off at DHW outlet height
            if tap_active and i == layer_dhw_out and mdot_tap > 0:
                tap_fraction = min(tap_duration_s / dt_s, 1.0)
                Q_tap_out = mdot_tap * C_P * T_prev[i]
                Q_net -= Q_tap_out * tap_fraction

            # cold feed into cold inlet height
            if tap_active and i == layer_cold_in and mdot_tap > 0:
                tap_fraction = min(tap_duration_s / dt_s, 1.0)
                Q_tap_in = mdot_tap * C_P * Tamb  # or mains_T if you model it separately
                Q_net += Q_tap_in * tap_fraction


            # Update layer temperature
            dT = Q_net * dt_s / (m_layer[i] * C_P)
            T_new[i] = np.clip(T_prev[i] + dT, 0.0, 100.0)

            # ---------------------------------
            # BUOYANCY / STRATIFICATION REORDERING
            # ---------------------------------
            # Enforce physically stable stratification:
            # coldest at bottom (index 0), hottest at top (index N_layers-1)
            T_new = np.sort(T_new)


        # ---------------------------------
        # SIMPLE BUFFER MODEL (OPTIONAL)
        # ---------------------------------
        mixer_flow_before = 0.0
        mixer_flow_after = 0.0
        mixer_return_temp = 0.0
        buffer_pump_flow = 0.0

        if buffer_enabled and buffer_layers is not None:
            # previous buffer temps
            prev_idx = max(step_count - 1, 0)
            Tbuf_prev = buffer_layers[prev_idx, :].copy()
            Tbuf_new = Tbuf_prev.copy()

            # Very simple: if HP on, dump Q_hp into bottom buffer layer
            if hp_on and Q_hp > 0:
                Q_net_buf = Q_hp
                dT_buf = Q_net_buf * dt_s / (layer_mass_buf * BUFFER_CP)
                Tbuf_new[-1] = np.clip(Tbuf_prev[-1] + dT_buf, 5.0, 95.0)

            # Immersion heater into top layer (if enabled)
            if immersion_enabled and immersion_power_kw > 0.0:
                P_imm = immersion_power_kw * 1000.0
                dT_imm = P_imm * dt_s / (layer_mass_buf * BUFFER_CP)
                Tbuf_new[0] = np.clip(Tbuf_new[0] + dT_imm, 5.0, 95.0)

            # Simple wall losses
            U_buf = 0.15
            A_buf = 2.5
            for k in range(N_buf):
                Q_loss_buf = U_buf * (Tbuf_prev[k] - Tamb) * (A_buf / N_buf)
                dT_loss = Q_loss_buf * dt_s / (layer_mass_buf * BUFFER_CP)
                Tbuf_new[k] = np.clip(Tbuf_new[k] - dT_loss, 5.0, 95.0)

            # Soft buoyancy correction
            for k in range(N_buf - 1):
                if Tbuf_new[k] > Tbuf_new[k + 1]:
                    Th = Tbuf_new[k]
                    Tc = Tbuf_new[k + 1]
                    alpha = 0.15
                    Tbuf_new[k] = (1 - alpha) * Th + alpha * Tc
                    Tbuf_new[k + 1] = alpha * Th + (1 - alpha) * Tc

            buffer_layers[step_count, :] = Tbuf_new

            # Mixer diagnostic temps (if enabled)
            if mixer_enabled:
                T_buffer_supply = float(Tbuf_new[0])
                T_return_est = flow_temp_after_mixer - 10.0
                mixer_flow_before = T_buffer_supply
                mixer_flow_after = flow_temp_after_mixer
                mixer_return_temp = T_return_est

            if boost_cylinder and pump_flow_lpm > 0:
                buffer_pump_flow = pump_flow_lpm

        # ---------------------------------
        # LOG STEP TO ARRAYS
        # ---------------------------------
        time_h = tnow_s / 3600.0

        time_array.append(time_h)
        T_array.append(T_new.copy())
        Qhp_array.append(Q_hp)
        Pel_array.append(Pel_hp)
        COPs_array.append(COP_hp)
        HP_on_array.append(1.0 if hp_on else 0.0)
        Qloss_total_array.append(Q_loss_total)
        Tap_flow_array.append(tap_rate_current if tap_active else 0.0)
        Tap_temp_array.append(T_prev[-1])   # draw from top
        Mod_frac_array.append(mod)
        Legionella_active_array.append(legionella_active)
        Senso_mode_array.append(current_mode)
        Senso_setpoint_array.append(active_setpoint)
        Tap_active_array.append(tap_active)
        Mixer_flow_before_array.append(mixer_flow_before)
        Mixer_flow_after_array.append(mixer_flow_after)
        Mixer_return_temp_array.append(mixer_return_temp)
        Buffer_pump_flow_array.append(buffer_pump_flow)
        HP_T_sink_array.append(T_flow_for_COP)
        HP_T_source_array.append(T_source_for_HP)
        Frost_active_array.append(frost_active)

        # Advance
        hp_on_prev = hp_on
        T_prev = T_new.copy()
        tnow_s += dt_s
        step_count += 1

    # ===============================
    # BUILD DATAFRAME
    # ===============================
    print(f"🔍 Debug: T_array length = {len(T_array)}")
    if len(T_array) > 0:
        print(f"🔍 Debug: First T_array element = {T_array[0]}")
        print(f"🔍 Debug: T_array[0] shape = {np.array(T_array[0]).shape}")
    time_h = np.array(time_array)
    T_matrix = np.array(T_array) if len(T_array) > 0 else np.zeros((0, N_layers))

    df = pd.DataFrame(
        {
            "Time (h)": time_h,
            "HP Power (W)": Pel_array,
            "HP Heat (W)": Qhp_array,
            "COP": COPs_array,
            "HP Sink Temp for COP (°C)": HP_T_sink_array,
            "HP Source Temp (°C)": HP_T_source_array,
            "HP_On": HP_on_array,
            "Modulation": Mod_frac_array,
            "Tap Flow (L/min)": Tap_flow_array,
            "Tap Temp (°C)": Tap_temp_array,
            "Q_Loss (W)": Qloss_total_array,
            "Legionella_Active": Legionella_active_array,
            "Senso_Mode": Senso_mode_array,
            "Senso_Setpoint (°C)": Senso_setpoint_array,
            "Tap_Active": Tap_active_array,
            "Mixer Flow Before (°C)": Mixer_flow_before_array,
            "Mixer Flow After (°C)": Mixer_flow_after_array,
            "Mixer Return Temp (°C)": Mixer_return_temp_array,
            "Buffer Pump Flow (L/min)": Buffer_pump_flow_array,
            "Frost_Protection_Active": Frost_active_array,
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

    # ===============================
    # BUFFER COLUMNS FOR UI (IF USED)
    # ===============================
    if buffer_enabled and buffer_layers is not None and step_count > 0:
        buf_slice = buffer_layers[:step_count, :]
        # Map internal order to user-friendly names (top→bottom)
        df["Buffer Top Layer (°C)"] = buf_slice[:, 0]
        df["Buffer Upper-Mid Layer (°C)"] = buf_slice[:, 1]
        df["Buffer Lower-Mid Layer (°C)"] = buf_slice[:, 2]
        df["Buffer Bottom Layer (°C)"] = buf_slice[:, 3]
        df["Buffer Avg Temp (°C)"] = buf_slice.mean(axis=1)

        # Simple buffer energy vs 0°C
        buffer_energy_kWh = (
            buffer_mass * BUFFER_CP * df["Buffer Avg Temp (°C)"] / 3.6e6
        )
        df["Buffer Energy (kWh vs 0°C)"] = buffer_energy_kWh

        # Very simple SOC split (HP vs Immersion, by zones)
        T_ref_cold = Tamb
        T_ref_hot = 60.0
        soc_hp = []
        soc_imm = []
        for row in buf_slice:
            T_hp_zone = row[2:].mean()   # lower 2 layers
            T_imm_zone = row[:2].mean()  # top 2 layers
            soc_hp.append(
                float(
                    np.clip(
                        (T_hp_zone - T_ref_cold) / (T_ref_hot - T_ref_cold) * 100.0,
                        0.0,
                        100.0,
                    )
                )
            )
            if immersion_enabled:
                soc_imm.append(
                    float(
                        np.clip(
                            (T_imm_zone - T_ref_cold)
                            / (T_ref_hot - T_ref_cold)
                            * 100.0,
                            0.0,
                            100.0,
                        )
                    )
                )
            else:
                soc_imm.append(0.0)

        df["SOC from HP (%)"] = soc_hp
        df["SOC from Immersion (%)"] = soc_imm

    # ===============================
    # ENERGY INTEGRATION
    # ===============================
    if len(df) > 1:
        df["dt_hours"] = df["Time (h)"].diff().fillna(
            df["Time (h)"].iloc[1] - df["Time (h)"].iloc[0]
        )
    else:
        df["dt_hours"] = 0.0

    # ✅ Always define end time
    df["Time_end"] = df["Time (h)"] + df["dt_hours"]



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

    # Optional buffer summary
    if buffer_enabled and buffer_layers is not None and step_count > 1:
        buf_slice = buffer_layers[:step_count, :]
        buf_avg = buf_slice.mean(axis=1)
        buffer_energy = buffer_mass * BUFFER_CP * buf_avg / 3.6e6  # kWh vs 0°C
        buffer_energy_change = buffer_energy[-1] - buffer_energy[0]
        summary["Buffer Energy Change (kWh)"] = float(buffer_energy_change)
        summary["Buffer Start Temp (°C)"] = float(buf_avg[0])
        summary["Buffer End Temp (°C)"] = float(buf_avg[-1])

    return df, summary, BASE_LAYER_PROPERTIES




# ---------- CHARTS ----------

def add_multiline_tooltip(base_chart, df, x_field, y_field, color_field):
    """
    Adds:
    - Synced global hover line
    - Tooltip that shows ALL series at the hover point
    - Cursor marker on all lines
    - Works with melted (long form) data
    """

    # Bind to global hover selector
    hover = global_hover

    # Tooltip fields: show one tooltip per row at the hovered x
    tooltip_fields = [
        alt.Tooltip(x_field, title="Time (h)"),
        alt.Tooltip(color_field, title="Series"),
        alt.Tooltip(y_field, title="Value", format=".2f"),
    ]

    # Hover points on all lines
    points = (
        alt.Chart(df)
        .mark_circle(size=60, color="black")
        .encode(
            x=x_field,
            y=y_field,
            tooltip=tooltip_fields,
            opacity=alt.value(1),
            color=color_field,
        )
        .transform_filter(hover)
    )

    # Each line is already in base_chart, add rule + points
    rule = (
        alt.Chart(df)
        .mark_rule(color="gray")
        .encode(x=x_field)
        .transform_filter(hover)
    )

    selectors = (
        alt.Chart(df)
        .mark_point(opacity=0)
        .encode(x=x_field)
        .add_params(hover)
    )

    return selectors + base_chart + rule + points


def add_state_shading(df: pd.DataFrame):
    """Generate shading rectangles for tap usage, compression, and immersion periods."""

    layers = []

    # --- Tap Active Shading ---
    if "Tap_Active" in df.columns and "Time_end" in df.columns:
        tap_df = df[df["Tap_Active"] == True]
        if not tap_df.empty:
            layers.append(
                alt.Chart(tap_df)
                .mark_rect(color="#FDBA74", opacity=0.25)
                .encode(
                    x="Time (h):Q",
                    x2="Time_end:Q"
                )
            )

    # --- Compression Shading ---
    if "Compression" in df.columns and "Time_end" in df.columns:
        comp_df = df[df["Compression"] == True]
        if not comp_df.empty:
            layers.append(
                alt.Chart(comp_df)
                .mark_rect(color="#93C5FD", opacity=0.20)
                .encode(
                    x="Time (h):Q",
                    x2="Time_end:Q"
                )
            )

    # --- Immersion Shading ---
    if "Immersion_On" in df.columns and "Time_end" in df.columns:
        imm_df = df[df["Immersion_On"] == True]
        if not imm_df.empty:
            layers.append(
                alt.Chart(imm_df)
                .mark_rect(color="#FCA5A5", opacity=0.25)
                .encode(
                    x="Time (h):Q",
                    x2="Time_end:Q"
                )
            )


    if not layers:
        return None

    return alt.layer(*layers)



# GLOBAL shared selection for synced hover across all charts
global_hover = alt.selection_point(
    fields=["Time (h)"],
    nearest=True,
    on="mousemove",
    empty=False
)

global_hover = alt.selection_point(
    fields=["Time (h)"],
    nearest=True,
    on="mousemove",
    empty=False
)

def chart_tapping_detail(df: pd.DataFrame):
    """
    Show tapping events and the cylinder temperature response
    with adaptive time stepping and background shading.
    """
    # Guard: if key columns are missing, return empty chart
    needed_cols = ["Time (h)", "Tap Flow (L/min)", "T_Top Layer (°C)", "T_Avg (°C)"]
    if not all(col in df.columns for col in needed_cols):
        return alt.Chart()

    base = alt.Chart(df).encode(
        x=alt.X("Time (h):Q", title="Time (hours)", axis=alt.Axis(grid=True))
    )

    # Top layer temperature line
    temp_line = base.mark_line(color="#1d4ed8", strokeWidth=2).encode(
        y=alt.Y("T_Top Layer (°C):Q",
                title="Top Layer Temperature (°C)",
                scale=alt.Scale(zero=False))
    )

    # Tap flow as bars on a secondary y-axis (scaled)
    tap_bar = base.mark_bar(opacity=0.4, color="#f59e0b").encode(
        y=alt.Y("Tap Flow (L/min):Q",
                title="Tap Flow (L/min)",
                axis=alt.Axis(titleColor="#f59e0b"))
    )

    # Combine temp + tap
    combo = temp_line + tap_bar

    # Fields for hover tooltip
    fields = [
        ("T_Top Layer (°C)", "Top Temp (°C)"),
        ("T_Avg (°C)", "Avg Temp (°C)"),
        ("Tap Flow (L/min)", "Tap Flow (L/min)"),
        ("HP Power (W)", "HP Power (W)"),
        ("HP Heat (W)", "HP Heat (W)"),
    ]

    chart = add_hover_summary(combo, df, fields)

    # Day markers for multi-day sims
    max_time = df["Time (h)"].max()
    if max_time > 48:
        day_lines = pd.DataFrame({
            "x": [24 * i for i in range(1, int(max_time / 24) + 1)]
        })
        rule = (
            alt.Chart(day_lines)
            .mark_rule(strokeDash=[4, 4], color="gray", opacity=0.5)
            .encode(x="x:Q")
        )
        chart = chart + rule

    return chart.properties(
        height=350,
        title="Tapping Detail – Top Layer Temperature vs Tap Flow"
    )

def add_hover_summary(base_chart, df_full, fields, df_plot=None):
    """
    Adds:
    - Synced hover rule across charts (global_hover)
    - Multi-field tooltip
    - Cursor point
    - Background shading (from df_full)
    """
    df_tips = df_full
    df_pts = df_plot if df_plot is not None else df_full

    # invisible selector to bind hover
    selectors = (
        alt.Chart(df_pts)
        .mark_point(opacity=0)
        .encode(x="Time (h):Q")
        .add_params(global_hover)
    )

    rule = (
        alt.Chart(df_pts)
        .mark_rule(color="gray")
        .encode(x="Time (h):Q")
        .transform_filter(global_hover)
    )

    # ✅ use df_pts for type checks & existence
    tooltip_fields = []
    for col, title in fields:
        if col not in df_pts.columns:
            continue  # skip fields not in the actual chart data

        if pd.api.types.is_numeric_dtype(df_pts[col]):
            tooltip_fields.append(alt.Tooltip(col, title=title, format=".2f"))
        else:
            tooltip_fields.append(alt.Tooltip(col, title=title))

    points = (
        alt.Chart(df_pts)
        .mark_circle(size=60, color="black")
        .encode(
            x="Time (h):Q",
            y=fields[0][0] + ":Q",
            tooltip=tooltip_fields,
        )
        .transform_filter(global_hover)
    )

    shading = add_state_shading(df_tips)

    if shading is not None:
        return selectors + base_chart + rule + points + shading
    else:
        return selectors + base_chart + rule + points




def chart_soc(df: pd.DataFrame):
    """SOC of buffer: HP vs Immersion."""
    
    if "SOC from HP (%)" not in df.columns:
        return alt.Chart()

    melted = df.melt(
        id_vars="Time (h)",
        value_vars=["SOC from HP (%)", "SOC from Immersion (%)"],
        var_name="Series",
        value_name="SOC (%)"
    )

    base = (
        alt.Chart(melted)
        .mark_line(strokeWidth=2)
        .encode(
            x="Time (h):Q",
            y=alt.Y("SOC (%):Q", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Series:N", title="Series"),
        )
        .properties(height=300, title="Buffer State of Charge (HP vs Immersion)")
    )

    return add_multiline_tooltip(base, melted, "Time (h):Q", "SOC (%):Q", "Series:N")





def chart_stratification(df: pd.DataFrame):
    """
    Plot 4-layer cylinder stratification with comprehensive error handling.
    """
    # 🔍 Debug: 
    st.write("**Debug - Looking for stratification columns...**")
    st.write(f"Total columns in df: {len(df.columns)}")
    
    # Define expected column names (exact match)
    expected_cols = [
        "T_Bottom Layer (°C)",
        "T_Lower-Mid Layer (°C)",
        "T_Upper-Mid Layer (°C)",
        "T_Top Layer (°C)"
    ]
    
    # Check which columns actually exist
    cols = [c for c in expected_cols if c in df.columns]
    st.write(f"Found {len(cols)} of 4 expected columns: {cols}")
    
    if not cols:
        # Fallback: try to find ANY layer columns
        cols = [c for c in df.columns if c.startswith("T_") and "Layer" in c and "Avg" not in c]
        st.write(f"Fallback search found: {cols}")
    
    if not cols:
        st.error("⚠️ No stratification data available")
        st.info("**Possible causes:**\n"
                "1. Simulation may not have run successfully\n"
                "2. Layer temperature data wasn't created\n"
                "3. Column names don't match expected format")
        return alt.Chart()
    
    # Check if columns have data
    for col in cols:
        if df[col].isna().all():
            st.warning(f"⚠️ Column '{col}' exists but contains only NaN values")
        elif len(df[col]) == 0:
            st.warning(f"⚠️ Column '{col}' exists but is empty")
    
    # Melt to long form
    try:
        d = df.melt("Time (h)", value_vars=cols, var_name="Layer", value_name="Temp (°C)")
        st.write(f"Melted dataframe shape: {d.shape}")
    except Exception as e:
        st.error(f"❌ Error melting dataframe: {e}")
        return alt.Chart()
    
    # Check for valid data
    if d["Temp (°C)"].isna().all():
        st.error("❌ All temperature values are NaN")
        return alt.Chart()
    
    if len(d) == 0:
        st.error("❌ Melted dataframe is empty")
        return alt.Chart()
    
    # Color scale mapping
    color_scale = alt.Scale(
        domain=[
            "T_Top Layer (°C)",
            "T_Upper-Mid Layer (°C)",
            "T_Lower-Mid Layer (°C)",
            "T_Bottom Layer (°C)",
        ],
        range=["#dc2626", "#f97316", "#60a5fa", "#1e3a8a"],
    )
    
    base = (
        alt.Chart(d)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)", axis=alt.Axis(grid=True)),
            y=alt.Y("Temp (°C):Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Layer:N", scale=color_scale, legend=alt.Legend(title="Tank Layer")),
        )
        .properties(height=400, title="Tank Stratification (4 Layers)")
    )
    
    chart = add_multiline_tooltip(base, d, "Time (h):Q", "Temp (°C):Q", "Layer:N")

    
    
    # Add day markers for 7-day view with labels
    max_time = d["Time (h)"].max()
    if max_time > 48:
        day_data = pd.DataFrame({
            'x': [24 * i for i in range(0, int(max_time / 24) + 1)],
            'label': [f'Day {i+1}' for i in range(int(max_time / 24) + 1)]
        })
        
        rule = alt.Chart(day_data).mark_rule(
            strokeDash=[4, 4], color='gray', opacity=0.5
        ).encode(x='x:Q')
        
        text = alt.Chart(day_data[day_data['x'] < max_time]).mark_text(
            align='left', dx=5, dy=-180, fontSize=10, color='gray'
        ).encode(x='x:Q', text='label:N')
        
        chart = chart + rule + text
    
    return chart

def chart_power(df: pd.DataFrame):
    base = (
        alt.Chart(df)
        .mark_line(color="#ef4444", strokeWidth=2)
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)", axis=alt.Axis(grid=True)),
            y="HP Power (W)"
        )
        .properties(height=300, title="Heat Pump Electrical Power (W)")
    )
    
    fields = [
        ("HP Power (W)", "HP Power (W)"),
        ("HP Heat (W)", "HP Heat (W)"),
        ("COP", "COP"),
        ("T_Top Layer (°C)", "Top Temp (°C)"),
        ("T_Avg (°C)", "Avg Temp (°C)")
    ]
    
    chart = add_hover_summary(base, df, fields)
    
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
    base = (
        alt.Chart(df)
        .mark_line(color="#1d4ed8", strokeWidth=2)
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)", axis=alt.Axis(grid=True)),
            y="HP Heat (W)"
        )
        .properties(height=300, title="Heat Pump Heat Output (W)")
    )
    
    fields = [
        ("HP Heat (W)", "HP Heat (W)"),
        ("HP Power (W)", "HP Power (W)"),
        ("COP", "COP"),
        ("T_Top Layer (°C)", "Top Temp (°C)"),
        ("T_Avg (°C)", "Avg Temp (°C)")
    ]
    
    chart = add_hover_summary(base, df, fields)
    
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

def chart_tank_temperature(df: pd.DataFrame):
    if "T_Avg (°C)" not in df.columns:
        return alt.Chart()

    melted = df.melt(
        id_vars="Time (h)",
        value_vars=["T_Avg (°C)", "Senso_Setpoint (°C)"],
        var_name="Series",
        value_name="Temp (°C)"
    )

    base = (
        alt.Chart(melted)
        .mark_line(strokeWidth=2)
        .encode(
            x="Time (h):Q",
            y="Temp (°C):Q",
            color=alt.Color("Series:N", title="Series"),
        )
        .properties(height=300, title="Cylinder Average Temperature")
    )

    return add_multiline_tooltip(base, melted, "Time (h):Q", "Temp (°C):Q", "Series:N")



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
        .properties(height=300, title="Cylinder heat losses (W)")
    )

    fields = [
        ("Q_Loss (W)", "Tank Loss (W)"),
        ("T_Top Layer (°C)", "Top Temp (°C)"),
        ("T_Avg (°C)", "Tank Avg Temp (°C)"),
        ("HP Power (W)", "HP Power (W)"),
        ("HP Heat (W)", "HP Heat (W)")
    ]

    chart = add_hover_summary(base, df, fields,)


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
    base = (
        alt.Chart(df)
        .mark_line(color="green", strokeWidth=2)
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)", axis=alt.Axis(grid=True)),
            y="COP"
        )
        .properties(height=300, title="Coefficient of Performance (COP)")
    )
    
    fields = [
        ("COP", "COP"),
        ("T_Top Layer (°C)", "Top Temp (°C)"),
        ("T_Bottom Layer (°C)", "Bottom Temp (°C)"),
        ("HP Power (W)", "HP Power (W)"),
        ("HP Heat (W)", "HP Heat (W)")
    ]
    
    chart = add_hover_summary(base, df, fields)
    
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
    base = alt.Chart(df).encode(x=alt.X("Time (h):Q", title="Time (hours)", axis=alt.Axis(grid=True)))
    mod = base.mark_area(opacity=0.6, color="#2563eb").encode(y="Modulation:Q")
    hp_on = base.mark_line(color="orange", strokeWidth=2).encode(y="HP_On:Q")
    
    combo = mod + hp_on
    
    fields = [
        ("Modulation", "Modulation"),
        ("HP_On", "HP On/Off"),
        ("HP Power (W)", "HP Power (W)"),
        ("T_Top Layer (°C)", "Top Temp (°C)")
    ]
    
    chart = add_hover_summary(combo, df, fields)
    
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
    
    return chart.properties(height=300, title="HP Modulation & On/Off State")

def chart_tap(df: pd.DataFrame):
    base = (
        alt.Chart(df)
        .mark_bar(color="#f59e0b")
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)"),
            y="Tap Flow (L/min)"
        )
        .properties(height=250, title="Tap Flow (L/min)")
    )
    
    fields = [
        ("Tap Flow (L/min)", "Tap Flow"),
        ("Tap Temp (°C)", "Tap Temp"),
        ("T_Top Layer (°C)", "Top Layer Temp")
    ]
    
    chart = add_hover_summary(base, df, fields)
    
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
    """Buffer stratification - WITH SAFETY CHECKS."""
    # Check for buffer columns
    layer_cols = [c for c in df.columns if c.startswith("Buffer ") and "Layer" in c]
    
    if not layer_cols or len(layer_cols) < 4:
        print("⚠️ No buffer layer data")
        return None  # Return None instead of empty chart!
    
    try:
        # Melt data
        d = df.melt(
            id_vars="Time (h)",
            value_vars=layer_cols,
            var_name="Layer",
            value_name="Temp (°C)"
        )
        
        # CHECK: Do we have actual data?
        if d.empty or len(d) == 0:
            print("⚠️ Buffer data melted to empty")
            return None
        
        if d["Temp (°C)"].isna().all():
            print("⚠️ All buffer temps are NaN")
            return None
        
    except Exception as e:
        print(f"⚠️ Buffer melt failed: {e}")
        return None
    
    # Color scale
    color_scale = alt.Scale(
        domain=[
            "Buffer Top Layer (°C)",
            "Buffer Upper-Mid Layer (°C)",
            "Buffer Lower-Mid Layer (°C)",
            "Buffer Bottom Layer (°C)",
        ],
        range=["#dc2626", "#f97316", "#60a5fa", "#1e3a8a"],
    )
    
    # Build chart
    base = (
        alt.Chart(d)
        .mark_line(strokeWidth=2)
        .encode(
            x="Time (h):Q",
            y="Temp (°C):Q",
            color=alt.Color("Layer:N", scale=color_scale),
        )
        .properties(title="Buffer Stratification (4-Layer)", height=400)
    )
    
    # Add interactions
    try:
        fields = [("Temp (°C)", "Buffer Temp"), ("Layer", "Layer")]
        chart = add_hover_summary(base, df, fields, df_plot=d)
    except:
        # If hover fails, return base chart
        chart = base
    
    return chart

    # --------------------------------------------------------------------
    # 3) Tooltip fields (only fields that exist in the melted dataframe)
    # --------------------------------------------------------------------
    fields = [
        ("Temp (°C)", "Buffer Temp"),
        ("Layer", "Layer"),
    ]

    chart = add_hover_summary(base, df, fields, df_plot=d)

    # ====================================================================
    # 4) ADD A SINGLE TAP LABEL ON LEFT SIDE (if any tapping occurs)
    # ====================================================================

    if "Tap_Active" in df.columns and df["Tap_Active"].any():

        # place label at fixed position near max temp
        ymax = d["Temp (°C)"].max()
        label_y = ymax + 1

        tap_label_df = pd.DataFrame({
            "Time (h)": [df["Time (h)"].min()],   # leftmost point only
            "label": ["Tap"]                      # single label
        })

        text_layer = (
            alt.Chart(tap_label_df)
            .mark_text(
                align="left",
                dx=4,
                dy=-5,
                fontSize=12,
                color="#d97706"
            )
            .encode(
                x="Time (h):Q",
                y=alt.value(label_y),
                text="label:N"
            )
        )

        chart = chart + text_layer


    return chart




    




def chart_legionella(df: pd.DataFrame):
    import altair as alt

    if "Legionella_Active" not in df.columns:
        return None

    # Detect top layer column safely
    candidates = [c for c in df.columns if c.startswith("T_Layer_")]
    if not candidates:
        return None

    top_col = candidates[0]

    d = df.copy()
    d["Legionella_Active_num"] = d["Legionella_Active"].astype(int)

    base = alt.Chart(d).encode(
        x=alt.X("Time (h):Q", title="Time (hours)")
    )

    temp_line = base.mark_line().encode(
        y=alt.Y(f"{top_col}:Q", title="Top Layer Temperature (°C)")
    )

    leg_df = d[d["Legionella_Active_num"] == 1]

    if len(leg_df) > 0:
        legionella_area = (
            alt.Chart(leg_df)
            .mark_area(opacity=0.25)
            .encode(
                x="Time (h):Q",
                y=alt.Y(f"{top_col}:Q")
            )
        )
        return (temp_line + legionella_area).properties(
            height=300,
            title="Legionella Cycle Activity"
        )
    else:
        return temp_line.properties(
            height=300,
            title="Legionella Cycle Activity"
        )



def chart_sensocomfort(df: pd.DataFrame):
    melted = df.melt(
        id_vars="Time (h)",
        value_vars=["T_Top Layer (°C)", "Senso_Setpoint (°C)"],
        var_name="Series",
        value_name="Temp (°C)"
    )

    base = (
        alt.Chart(melted)
        .mark_line(strokeWidth=2)
        .encode(
            x="Time (h):Q",
            y="Temp (°C):Q",
            color=alt.Color("Series:N", title="Series"),
        )
        .properties(height=300, title="SensoComfort: Temperature vs Setpoint")
    )

    return add_multiline_tooltip(base, melted, "Time (h):Q", "Temp (°C):Q", "Series:N")


def chart_buffer_pump(df: pd.DataFrame):
    """Chart showing buffer pump flow over time."""

    if "Buffer Pump Flow (L/min)" not in df.columns:
        return alt.Chart()

    base = (
        alt.Chart(df)
        .mark_bar(color="#ef4444")
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)"),
            y=alt.Y("Buffer Pump Flow (L/min):Q", title="Buffer Pump Flow (L/min)")
        )
        .properties(height=250, title="Buffer Pump Flow (L/min)")
    )

    fields = [
        ("Buffer Pump Flow (L/min)", "Pump Flow"),
        ("T_Top Layer (°C)", "Top Temp")
    ]
    
    chart = add_hover_summary(base, df, fields)

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

def chart_mixer_temps(df: pd.DataFrame):
    df2 = df[df["Mixer Flow Before (°C)"] > 0].copy()
    if df2.empty:
        return alt.Chart()

    melted = df2.melt(
        id_vars="Time (h)",
        value_vars=[
            "Mixer Flow Before (°C)",
            "Mixer Flow After (°C)",
            "Mixer Return Temp (°C)",
        ],
        var_name="Series",
        value_name="Temp (°C)"
    )

    base = (
        alt.Chart(melted)
        .mark_line(strokeWidth=2)
        .encode(
            x="Time (h):Q",
            y="Temp (°C):Q",
            color=alt.Color("Series:N", title="Series"),
        )
        .properties(height=300, title="Heating Circuit Mixer Temperatures")
    )

    return add_multiline_tooltip(base, melted, "Time (h):Q", "Temp (°C):Q", "Series:N")


def chart_buffer_temperature(df: pd.DataFrame):
    """Chart showing buffer tank average temperature over time."""
    if "Buffer Avg Temp (°C)" not in df.columns:
        return alt.Chart()
    
    base = alt.Chart(df).encode(
        x=alt.X("Time (h):Q", title="Time (hours)", axis=alt.Axis(grid=True))
    )
    
    # Average buffer temperature line
    avg_temp = base.mark_line(color="#dc2626", strokeWidth=2).encode(
        y=alt.Y("Buffer Avg Temp (°C):Q", title="Temperature (°C)", scale=alt.Scale(zero=False))
    )
    
    fields = [
        ("Buffer Avg Temp (°C)", "Buffer Avg Temp"),
        ("HP Power (W)", "HP Power"),
        ("HP Heat (W)", "HP Heat")
    ]
    
    chart = add_hover_summary(avg_temp, df, fields)
    
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
    
    return chart.properties(
        height=300,
        title="Buffer Tank Average Temperature"
    )


def chart_buffer_losses(df: pd.DataFrame):
    """Chart showing buffer tank heat losses over time."""
    # Calculate buffer heat losses if buffer data exists
    if "Buffer Avg Temp (°C)" not in df.columns:
        return alt.Chart()
    
    # Calculate heat losses: Q_loss = U * A * (T_buffer - T_ambient)
    # Assuming buffer tank with U=0.15 W/m²K and surface area ~2.5 m²
    U_buffer = 0.15  # W/m²K
    A_buffer = 2.5   # m² (approximate for 150L cylindrical tank)
    T_ambient = 15.0  # °C (from typical ambient)
    
    # Calculate losses
    df_copy = df.copy()
    df_copy["Buffer Heat Loss (W)"] = U_buffer * A_buffer * (df_copy["Buffer Avg Temp (°C)"] - T_ambient)
    
    base = (
        alt.Chart(df_copy)
        .mark_line(color="#f97316", strokeWidth=2)
        .encode(
            x=alt.X("Time (h):Q", title="Time (hours)", axis=alt.Axis(grid=True)),
            y=alt.Y("Buffer Heat Loss (W):Q", title="Buffer Heat Loss (W)", scale=alt.Scale(zero=False))
        )
        .properties(height=300, title="Buffer Tank Heat Losses (W)")
    )
    
    fields = [
        ("Buffer Heat Loss (W)", "Buffer Loss (W)"),
        ("Buffer Avg Temp (°C)", "Buffer Avg Temp (°C)"),
    ]
    
    chart = add_hover_summary(base, df_copy, fields)
    
    # Day markers for 2+ days
    max_time = df_copy["Time (h)"].max()
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

def chart_cylinder_soc(df: pd.DataFrame):
    if "T_Avg (°C)" not in df.columns:
        return alt.Chart()

    T_ref_cold = 15.0
    T_ref_hot = 60.0

    d = df.copy()
    d["Avg SOC (%)"] = np.clip((d["T_Avg (°C)"] - T_ref_cold) / (T_ref_hot - T_ref_cold) * 100, 0, 100)
    d["Top SOC (%)"] = np.clip((d["T_Top Layer (°C)"] - T_ref_cold) / (T_ref_hot - T_ref_cold) * 100, 0, 100)
    d["Bottom SOC (%)"] = np.clip((d["T_Bottom Layer (°C)"] - T_ref_cold) / (T_ref_hot - T_ref_cold) * 100, 0, 100)

    melted = d.melt(
        id_vars="Time (h)",
        value_vars=["Avg SOC (%)", "Top SOC (%)", "Bottom SOC (%)"],
        var_name="Series",
        value_name="SOC (%)"
    )

    base = (
        alt.Chart(melted)
        .mark_line(strokeWidth=2)
        .encode(
            x="Time (h):Q",
            y=alt.Y("SOC (%):Q", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Series:N", title="Series")
        )
        .properties(height=300, title="Cylinder State of Charge")
    )

    return add_multiline_tooltip(base, melted, "Time (h):Q", "SOC (%):Q", "Series:N")







# ==============================================================
# MAIN APPLICATION
# ==============================================================
def main():
    with st.sidebar:
        st.header("⚙️ Simulation Parameters")

        # ==============================================================
        # 🔥 HEAT PUMP CONTROL (MOVED TO TOP)
        # ==============================================================
        st.markdown("### 🔥 Heat Pump Control")
        
        # Heat Pump Model Selection
        hp_model_name = st.selectbox(
            "Heat Pump Model",
            ["3.5 kW", "5 kW", "7 kW", "10 kW", "12 kW"],
            index=2,
            key="hp_model"
        )
        
        # Setpoint Temperatures (NEW - Split into Heating and DHW)
        col1, col2 = st.columns(2)
        with col1:
            heating_setp = st.slider(
                "Heating Setpoint (°C)", 
                30, 65, 45, 1, 
                key="heating_setp",
                help="Target temperature for space heating circuit"
            )
        with col2:
            dhw_setp = st.slider(
                "DHW Setpoint (°C)", 
                40, 90, 50, 1, 
                key="dhw_setp",
                help="Target temperature for domestic hot water cylinder"
            )
        
        # Control Parameters
        hyst = st.slider(
            "Hysteresis (°C)", 
            1.0, 10.0, 5.0, 0.5, 
            key="param_hyst",
            help="Temperature band for HP on/off switching"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            Pmax = st.slider(
                "Max Power Ratio", 
                0.5, 1.0, 1.0, 0.05, 
                key="param_pmax",
                help="Maximum HP output capacity"
            )
        with col2:
            Pmin = st.slider(
                "Min Modulation", 
                0.1, 0.5, 0.2, 0.05, 
                key="param_pmin",
                help="Minimum modulation level"
            )
        
        # Operating Modes
        col1, col2 = st.columns(2)
        with col1:
            quiet_mode_enabled = st.checkbox(
                "Quiet Mode", 
                value=False,
                help="Reduces noise but slightly decreases efficiency"
            )
        with col2:
            defrost_enabled = st.checkbox(
                "Enable Defrost", 
                value=True,
                help="Automatic defrost cycle in cold conditions"
            )

        # ==============================================================
        # 🏠 TANK AND ENVIRONMENT
        # ==============================================================
        st.markdown("---")
        st.markdown("### 🏠 Tank and Environment")
        
        st.write("Tank Volume (L): **150 L (fixed)**")
        
        # Initial Tank Temperature
        initial_tank_temp = st.slider(
            "Initial Tank Temperature (°C)",
            10, 80, 45, 1,
            key="initial_tank_temp"
        )

        # Simulation Period
        sim_period = st.radio(
            "Simulation Period",
            ["24 Hours", "7 Days"],
            index=0,
            horizontal=True,
            key="sim_period"
        )

        if sim_period == "24 Hours":
            simh = 24
            st.info("📊 Viewing: 24-hour detailed analysis")
        else:
            simh = 168
            st.info("📊 Viewing: 7-day weekly pattern")

        # Environment Parameters
        col1, col2 = st.columns(2)
        with col1:
            Utank = st.slider(
                "Tank U-value (W/m²K)", 
                0.1, 2.0, 0.4, 0.1,
                key="utank"
            )
        with col2:
            Tamb = st.slider(
                "Ambient Temp (°C)", 
                0, 30, 15, 
                key="param_tamb"
            )
        
        dtm = st.slider(
            "Time Step (min)", 
            1, 10, 5, 
            key="param_dtm",
            help="Simulation time resolution"
        )

        # ==============================================================
        # 🎛️ SENSOCOMFORT CONTROLS
        # ==============================================================
        st.markdown("---")
        st.markdown("### 🎛️ SensoComfort Controls")

        senso_enabled = st.radio(
            "Enable SensoComfort Smart Control",
            ["Off", "On"],
            horizontal=True,
            key="senso_toggle"
        )

        if senso_enabled == "On":
            comfort_mode = st.selectbox(
                "Comfort Mode",
                ["ECO", "Comfort", "Auto"],
                index=1,
                key="comfort_mode"
            )

            # Night behaviour (Modo noche)
            night_mode_ui = st.selectbox(
                "Night Mode Behaviour (Modo noche)",
                ["Eco (heating off, frost only)", "Normal (reduced temperature)"],
                index=0,
                key="night_mode_ui",
            )
            if "Eco" in night_mode_ui:
                night_mode = "Eco"
            else:
                night_mode = "Normal"

            # Absence mode (Ausencia)
            absence_mode = st.checkbox("Absence Mode (Ausencia)", key="absence_mode")
            if absence_mode:
                absence_temp = st.slider(
                    "Absence Temperature (°C)",
                    5, 25, 15, 1,
                    key="absence_temp"
                )
            else:
                absence_temp = 15.0

            # Frost protection
            frost_protection_enabled = st.checkbox(
                "Enable Frost Protection (anti-freeze)",
                value=True,
                key="frost_protection_enabled"
            )
            frost_protection_temp = st.slider(
                "Frost Protection Temperature (°C)",
                3, 10, 5, 1,
                key="frost_protection_temp"
            )

            # Comfort / ECO temperatures
            col1, col2 = st.columns(2)
            with col1:
                eco_temp = st.number_input(
                    "ECO Temperature (°C)", 40, 60, 45, 1, key="eco_temp"
                )
            with col2:
                comfort_temp = st.number_input(
                    "Comfort Temperature (°C)", 45, 65, 55, 1, key="comfort_temp"
                )

            enable_time_program = st.checkbox(
                "Enable Time Program", value=True, key="enable_time_program"
            )

            if enable_time_program:
                col1, col2 = st.columns(2)
                with col1:
                    morning_start = st.slider(
                        "Morning Start Hour", 0, 23, 6, 1, key="morning_start"
                    )
                with col2:
                    morning_end = st.slider(
                        "Morning End Hour", 0, 23, 9, 1, key="morning_end"
                    )

                col1, col2 = st.columns(2)
                with col1:
                    evening_start = st.slider(
                        "Evening Start Hour", 0, 23, 17, 1, key="evening_start"
                    )
                with col2:
                    evening_end = st.slider(
                        "Evening End Hour", 0, 23, 22, 1, key="evening_end"
                    )
            else:
                morning_start, morning_end = 6, 9
                evening_start, evening_end = 17, 22

            boost_mode = st.checkbox("Quick Heat-up Mode", key="boost_mode")
            holiday_mode = st.checkbox("Holiday Mode", key="holiday_mode")

            if holiday_mode:
                holiday_temp = st.slider(
                    "Holiday Temperature (°C)", 35, 50, 40, 1, key="holiday_temp"
                )
            else:
                holiday_temp = 40

        else:
            # Defaults when SensoComfort is OFF
            comfort_mode = "Comfort"
            eco_temp = 45
            comfort_temp = 55
            enable_time_program = False
            morning_start, morning_end = 6, 9
            evening_start, evening_end = 17, 22
            boost_mode = False
            holiday_mode = False
            holiday_temp = 40
            night_mode = "Eco"
            absence_mode = False
            absence_temp = 15.0
            frost_protection_enabled = True
            frost_protection_temp = 5.0

        # ==============================================================
        # 🦠 LEGIONELLA PROTECTION
        # ==============================================================
        st.markdown("---")
        st.markdown("### 🦠 Legionella Protection")

        legionella_enabled = st.radio(
            "Enable Legionella",
            ["Off", "On"],
            horizontal=True,
            key="legionella_toggle"
        )

        if legionella_enabled == "On":
            legionella_temp = st.slider("Target Temp (°C)", 60, 75, 65, 1, key="legionella_temp")
            legionella_duration = st.slider("Hold Duration (min)", 10, 60, 30, 5, key="legionella_duration")
            legionella_frequency = st.slider("Frequency (days)", 1, 14, 7, 1, key="legionella_freq")
            legionella_start_hour = st.slider("Start Hour", 0, 23, 2, 1, key="legionella_hour")
        else:
            legionella_temp = 65
            legionella_duration = 30
            legionella_frequency = 7
            legionella_start_hour = 2

        # ==============================================================
        # 🔥 BUFFER TANK
        # ==============================================================
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

            # Mixer
            st.markdown("#### 🌡️ Heating Circuit Mixer")
            mixer_enabled = st.checkbox("Enable Heating Circuit Mixer", key="mixer_enabled")

            if mixer_enabled:
                col1, col2 = st.columns(2)
                with col1:
                    flow_temp_before_mixer = st.slider(
                        "Flow Temp Before Mixer (°C)", 30, 80, 60, 1, key="flow_temp_before_mixer"
                    )
                with col2:
                    flow_temp_after_mixer = st.slider(
                        "Flow Temp After Mixer (°C)", 20, 70, 45, 1, key="flow_temp_after_mixer"
                    )

                heating_circuit_flow_lpm = st.slider(
                    "Heating Circuit Flow (L/min)", 2, 50, 20, 1, key="heating_circuit_flow_lpm"
                )
            else:
                flow_temp_before_mixer = 60
                flow_temp_after_mixer = 45
                heating_circuit_flow_lpm = 20
        else:
            buffer_volume_L = 0
            buffer_temp_init = 45
            immersion_enabled = False
            immersion_power_kw = 0.0
            boost_cylinder = False
            pump_flow_lpm = 0
            mixer_enabled = False
            flow_temp_before_mixer = 60
            flow_temp_after_mixer = 45
            heating_circuit_flow_lpm = 20

       


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
        "dhw_setp": dhw_setp,          
        "heating_setp": heating_setp,
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
        "mixer_enabled": st.session_state.get("mixer_enabled", False),
        "flow_temp_before_mixer": st.session_state.get("flow_temp_before_mixer", 60),
        "flow_temp_after_mixer": st.session_state.get("flow_temp_after_mixer", 45),
        "heating_circuit_flow_lpm": st.session_state.get("heating_circuit_flow_lpm", 20),
        "hp_model_name": hp_model_name,
        "quiet_mode_enabled": quiet_mode_enabled,
        "defrost_enabled": defrost_enabled,
        "night_mode": night_mode,
        "absence_mode": absence_mode,
        "absence_temp": absence_temp,
        "frost_protection_enabled": frost_protection_enabled,
        "frost_protection_temp": frost_protection_temp,

}

    # Keep full parameter dict for PDF export
    st.session_state["last_params_dict"] = params

    
    param_key = tuple(params.values())

    # Detect parameter changes
    param_changed = (
        "last_params" not in st.session_state
        or st.session_state["last_params"] != param_key
    )

    # --- Run simulation if params changed or first run ---
    if param_changed:
        with st.spinner("🔄 Running simulation..."):
            df, summary, layer_properties = run_sim(
                dt_min=params["dtm"],
                sim_hrs=params["simh"],
                Utank=params["Utank"],
                Tamb=params["Tamb"],
                setp=params["dhw_setp"],
                hyst=params["hyst"],
                Pmin=params["Pmin"],
                Pmax=params["Pmax"],
                Tsrc=5.0,
                tap=tap,
                initial_tank_temp=initial_tank_temp,

                legionella_enabled=params["legionella_enabled"],
                legionella_temp=params["legionella_temp"],
                legionella_duration=params["legionella_duration"],
                legionella_frequency=params["legionella_frequency"],
                legionella_start_hour=params["legionella_start_hour"],

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

                buffer_enabled=params["buffer_enabled"],
                buffer_volume_L=params["buffer_volume_L"],
                buffer_temp_init=params["buffer_temp_init"],
                immersion_enabled=params["immersion_enabled"],
                immersion_power_kw=params["immersion_power_kw"],
                boost_cylinder=params["boost_cylinder"],
                pump_flow_lpm=params["pump_flow_lpm"],

                mixer_enabled=params["mixer_enabled"],
                flow_temp_before_mixer=params["flow_temp_before_mixer"],
                flow_temp_after_mixer=params["flow_temp_after_mixer"],
                heating_circuit_flow_lpm=params["heating_circuit_flow_lpm"],

                hp_model_name=params["hp_model_name"],
                quiet_mode_enabled=params["quiet_mode_enabled"],
                defrost_enabled=params["defrost_enabled"],
                night_mode=params["night_mode"],
                absence_mode=params["absence_mode"],
                absence_temp=params["absence_temp"],
                frost_protection_enabled=params["frost_protection_enabled"],
                frost_protection_temp=params["frost_protection_temp"],
            )

        st.session_state["df"] = df
        st.session_state["summary"] = summary
        st.session_state["layer_properties"] = layer_properties
        st.session_state["last_params"] = param_key
        st.session_state["ai_paused"] = True

 

    

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
        "Cylinder Stratification (Multi-Layer)",
        "Heat Pump Power (W)",
        "Heat Pump Heat Output (W)",
        "HP Modulation & On/Off State",
        "Tap Flow (L/min)",
        "Legionella Cycle Activity",
        "Cylinder State of Charge",
        "Cylinder Average Temperature",
        "Cylinder Heat Losses (W)",
        "SensoComfort Control",
        "Tapping Detail (Temperature Response)",
        "Buffer Stratification (4-Layer)",
        "Buffer Pump Flow (L/min)",
        "Heating Circuit Mixer Temperatures", 
        "Buffer State of Charge (HP vs Immersion)", 
        "Buffer Tank Average Temperature", 
        "Buffer Tank Heat Losses (W)", 

        




    ],
    key="graph_choice"
)

        

        if choice == "Coefficient of Performance (COP)":
            st.altair_chart(chart_cop(df), use_container_width=True)
        elif choice == "Cylinder Stratification (Multi-Layer)":
            st.altair_chart(chart_stratification(df), use_container_width=True)
        elif choice == "Heat Pump Power (W)":
            st.altair_chart(chart_power(df), use_container_width=True)
        elif choice == "Heat Pump Heat Output (W)":
            st.altair_chart(chart_heat(df), use_container_width=True)
        elif choice == "Cylinder Heat Losses (W)":
            st.altair_chart(chart_tank_losses(df), use_container_width=True)
        elif choice == "Cylinder Average Temperature":
            st.altair_chart(chart_tank_temperature(df), use_container_width=True)
            
            # Show temperature statistics
            if "T_Avg (°C)" in df.columns:
                avg_temp = df["T_Avg (°C)"].mean()
                min_temp = df["T_Avg (°C)"].min()
                max_temp = df["T_Avg (°C)"].max()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Temperature", f"{avg_temp:.1f}°C")
                with col2:
                    st.metric("Minimum Temperature", f"{min_temp:.1f}°C")
                with col3:
                    st.metric("Maximum Temperature", f"{max_temp:.1f}°C")
                
                # Show temperature stability
                temp_std = df["T_Avg (°C)"].std()
                st.info(f"📊 Temperature Stability: Standard deviation = {temp_std:.2f}°C")
        elif choice == "HP Modulation & On/Off State":
            st.altair_chart(chart_modulation(df), use_container_width=True)
        elif choice == "Tap Flow (L/min)":
            st.altair_chart(chart_tap(df), use_container_width=True)
        elif choice == "🦠 Legionella Cycle Activity":
            leg_chart = chart_legionella(df)
            if leg_chart is not None:
                st.altair_chart(leg_chart, use_container_width=True)
            else:
                st.info("No Legionella data available for this simulation.")

        elif choice == "SensoComfort Control":
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
            layer_cols = [
                c for c in df.columns
                if c.startswith("Buffer ") and "Layer" in c
            ]

            if len(layer_cols) < 4:
                st.warning("⚠️ Buffer tank is not enabled. Please enable buffer tank in the sidebar to view this graph.")
                st.info("💡 **To enable:** Sidebar → Buffer Tank → Enable Buffer: **On**")
            else:
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


        elif choice == "Buffer Pump Flow (L/min)":
                if "Buffer Pump Flow (L/min)" in df.columns:
                    st.altair_chart(chart_buffer_pump(df), use_container_width=True)
                    st.info(
                    f"Buffer pump is modeled as a constant {st.session_state.get('pump_flow_lpm', 0)} L/min "
                    f"whenever 'Boost Cylinder via Buffer Pump' is enabled."
                )
                else:
                    st.warning("⚠️ Buffer pump is not enabled or no buffer data is available.")
                    st.info("💡 Enable buffer tank and boost pump in the sidebar to see this graph.")

        elif choice == "🌡️ Heating Circuit Mixer Temperatures":
            mixer_chart = chart_mixer_temps(df)
            
            # ✅ CORRECT: Check if columns exist in dataframe instead
            if "Mixer Flow Before (°C)" in df.columns and df["Mixer Flow Before (°C)"].sum() > 0:
                st.altair_chart(mixer_chart, use_container_width=True)
                st.info(
                    f"🌡️ Mixer Control: "
                    f"Target flow temperature: {st.session_state.get('flow_temp_after_mixer', 45)}°C | "
                    f"Buffer supply: {st.session_state.get('flow_temp_before_mixer', 60)}°C | "
                    f"Circuit flow: {st.session_state.get('heating_circuit_flow_lpm', 20)} L/min"
                )
            else:
                st.warning("⚠️ Heating circuit mixer is not enabled.")
                st.info("💡 **To enable:** Sidebar → 🔥 Buffer Tank → Enable Heating Circuit Mixer")

        elif choice == "📊 Buffer State of Charge (HP vs Immersion)":
            # ✅ CORRECT: Check dataframe columns instead of chart.data
            if "SOC from HP (%)" in df.columns:
                soc_chart = chart_soc(df)
                st.altair_chart(soc_chart, use_container_width=True)
        
                # Show SOC statistics
                avg_soc_hp = df["SOC from HP (%)"].mean()
                avg_soc_imm = df["SOC from Immersion (%)"].mean()
            
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg SOC from HP", f"{avg_soc_hp:.1f}%")
                with col2:
                    st.metric("Avg SOC from Immersion", f"{avg_soc_imm:.1f}%")
            else:
                st.warning("⚠️ Buffer tank is not enabled.")
                st.info("💡 **To enable:** Sidebar → 🔥 Buffer Tank → Enable Buffer: **On**")
        elif choice == "🌡️ Buffer Tank Average Temperature":
            if "Buffer Avg Temp (°C)" in df.columns:
                st.altair_chart(chart_buffer_temperature(df), use_container_width=True)
        
            # Show temperature statistics
                avg_temp = df["Buffer Avg Temp (°C)"].mean()
                min_temp = df["Buffer Avg Temp (°C)"].min()
                max_temp = df["Buffer Avg Temp (°C)"].max()
        
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Temperature", f"{avg_temp:.1f}°C")
                with col2:
                    st.metric("Minimum Temperature", f"{min_temp:.1f}°C")
                with col3:
                    st.metric("Maximum Temperature", f"{max_temp:.1f}°C")
        
        # Show temperature stability
                temp_std = df["Buffer Avg Temp (°C)"].std()
                st.info(f"📊 Temperature Stability: Standard deviation = {temp_std:.2f}°C")
            else:
                st.warning("⚠️ Buffer tank is not enabled.")
                st.info("💡 **To enable:** Sidebar → 🔥 Buffer Tank → Enable Buffer: **On**")

        elif choice == "🔥 Buffer Tank Heat Losses (W)":
                if "Buffer Avg Temp (°C)" in df.columns:
                    st.altair_chart(chart_buffer_losses(df), use_container_width=True)
            
                    # Calculate and show total losses
                    U_buffer = 0.15
                    A_buffer = 2.5
                    T_ambient = 15.0
                    df_copy = df.copy()
                    df_copy["Buffer Heat Loss (W)"] = U_buffer * A_buffer * (df_copy["Buffer Avg Temp (°C)"] - T_ambient)
            
                    if "dt_hours" in df_copy.columns:
                        total_buffer_losses_kWh = (df_copy["Buffer Heat Loss (W)"] * df_copy["dt_hours"]).sum() / 1000.0
                        avg_loss_W = df_copy["Buffer Heat Loss (W)"].mean()
                
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Buffer Heat Loss", f"{total_buffer_losses_kWh:.2f} kWh")
                        with col2:
                            st.metric("Average Heat Loss", f"{avg_loss_W:.1f} W")
                
                        st.info(f"📊 Buffer tank losses calculated using U-value of {U_buffer} W/m²K and surface area of {A_buffer} m²")
                else:
                    st.warning("⚠️ Buffer tank is not enabled.")
                    st.info("💡 **To enable:** Sidebar → 🔥 Buffer Tank → Enable Buffer: **On**")

        elif choice == "Cylinder State of Charge":
            st.altair_chart(chart_cylinder_soc(df), use_container_width=True)
    
            # Calculate SOC statistics
            T_ref_cold = 15.0
            T_ref_hot = 60.0
        
            df_soc = df.copy()
            df_soc["Cylinder SOC (%)"] = np.clip(
                (df_soc["T_Avg (°C)"] - T_ref_cold) / (T_ref_hot - T_ref_cold) * 100.0,
                0.0, 100.0
            )
        
            avg_soc = df_soc["Cylinder SOC (%)"].mean()
            min_soc = df_soc["Cylinder SOC (%)"].min()
            max_soc = df_soc["Cylinder SOC (%)"].max()
        
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average SOC", f"{avg_soc:.1f}%")
            with col2:
                st.metric("Minimum SOC", f"{min_soc:.1f}%")
            with col3:
                st.metric("Maximum SOC", f"{max_soc:.1f}%")
        
            # Calculate energy stored
            cylinder_volume_L = 150.0
            cylinder_mass_kg = cylinder_volume_L
            cp = 4.186  # kJ/kg·K
        
            # Energy stored at average temperature
            T_avg = df["T_Avg (°C)"].mean()
            energy_stored_kWh = (cylinder_mass_kg * cp * (T_avg - T_ref_cold)) / 3600.0
            max_energy_capacity_kWh = (cylinder_mass_kg * cp * (T_ref_hot - T_ref_cold)) / 3600.0
        
            st.markdown("---")
            st.markdown("#### Energy Storage Analysis")
        
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Energy Stored", f"{energy_stored_kWh:.2f} kWh")
            with col2:
                st.metric("Maximum Capacity", f"{max_energy_capacity_kWh:.2f} kWh")
        
            st.info(
                f"📊 **SOC Calculation**: Based on temperature range from {T_ref_cold}°C (0% SOC) "
                f"to {T_ref_hot}°C (100% SOC). The blue area shows average cylinder SOC, "
                f"with red dashed line for top layer and blue dashed line for bottom layer."
            )
        
            # Show stratification impact
            top_avg = df["T_Top Layer (°C)"].mean()
            bottom_avg = df["T_Bottom Layer (°C)"].mean()
            stratification_delta = top_avg - bottom_avg
        
            st.info(
                f"🌡️ **Stratification Impact**: Average temperature difference between top and bottom layers: "
                f"{stratification_delta:.1f}°C. Good stratification (>5°C) improves hot water availability."
            )
            
        if "summary" in st.session_state:
            df = st.session_state["df"]
            summary = st.session_state["summary"]
            layer_properties = st.session_state["layer_properties"]
            ...


        # --------------------------------------
        # 📄 PDF export – full technical report
        # --------------------------------------
        st.markdown("---")
        st.subheader("📄 Export Full PDF Report")

        # Get latest params dict back
        params_for_pdf = st.session_state.get("last_params_dict", {})
        
        # Generate on demand and cache bytes in session_state
        if st.button("📄 Generate full PDF report", key="btn_generate_pdf"):
            pdf_bytes = generate_pdf_report(
                df=df,
                summary=summary,
                params=params_for_pdf,
                sim_period=sim_period,
            )
            st.session_state["pdf_report_bytes"] = pdf_bytes
            st.success("PDF report generated – scroll down to download ⬇️")

        if "pdf_report_bytes" in st.session_state:
            st.download_button(
                "⬇️ Download PDF report",
                data=st.session_state["pdf_report_bytes"],
                file_name=f"Vaillant_150L_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )

            
     

# ============================
# CONSTANTS
# ============================
RHO = 1000
CP = 4186

BUFFER_RHO = 1000
BUFFER_CP = 4186

P_EL_MAX = 5000

def build_pdf_charts_selective(df: pd.DataFrame, chart_selections: Dict[str, bool]) -> List[alt.Chart]:
    """
    Build charts with ROBUST error handling.
    Skips any chart that fails instead of crashing.
    """
    charts = []
    
    def try_add_chart(key: str, chart_func, name: str):
        """Helper to safely try adding a chart."""
        if not chart_selections.get(key, False):
            return
        try:
            chart = chart_func()
            if chart is not None:
                # Validate by trying to convert to dict
                try:
                    _ = chart.to_dict()
                    charts.append(chart)
                    print(f"✅ Added: {name}")
                except:
                    print(f"⚠️ Skipped {name}: Invalid chart structure")
        except Exception as e:
            print(f"⚠️ Skipped {name}: {str(e)[:40]}")
    
    # Try each chart individually
    try_add_chart("cop", lambda: chart_cop(df), "COP")
    try_add_chart("stratification", lambda: chart_stratification(df), "Stratification")
    try_add_chart("power", lambda: chart_power(df), "Power")
    try_add_chart("heat", lambda: chart_heat(df), "Heat")
    try_add_chart("tank_temp", lambda: chart_tank_temperature(df), "Tank Temp")
    try_add_chart("tank_losses", lambda: chart_tank_losses(df), "Tank Losses")
    try_add_chart("modulation", lambda: chart_modulation(df), "Modulation")
    try_add_chart("tap", lambda: chart_tap(df), "Tap")
    try_add_chart("soc", lambda: chart_cylinder_soc(df), "Cylinder SOC")
    
    # Conditional charts
    if chart_selections.get("legionella", False):
        if "Legionella_Active" in df.columns:
            try_add_chart("legionella", lambda: chart_legionella(df), "Legionella")
    
    if chart_selections.get("senso", False):
        if "Senso_Setpoint (°C)" in df.columns:
            try_add_chart("senso", lambda: chart_sensocomfort(df), "SensoComfort")
    
    # Buffer charts - MORE CAREFUL HERE
    if chart_selections.get("buffer", False):
        if any(c.startswith("Buffer ") for c in df.columns):
            try_add_chart("buffer", lambda: chart_buffer_strat(df), "Buffer Strat")
            try_add_chart("buffer", lambda: chart_buffer_temperature(df), "Buffer Temp")
            try_add_chart("buffer", lambda: chart_buffer_losses(df), "Buffer Losses")
            try_add_chart("buffer", lambda: chart_soc(df), "Buffer SOC")
            try_add_chart("buffer", lambda: chart_buffer_pump(df), "Buffer Pump")
            try_add_chart("buffer", lambda: chart_mixer_temps(df), "Mixer Temps")
    
    print(f"\n📊 Total valid charts for PDF: {len(charts)}")
    return charts


def render_pdf_preview_first_page(pdf_bytes: bytes) -> bytes:
    """
    Return a PNG image (bytes) of the first page of the given PDF.
    Requires PyMuPDF (fitz).
    
    Args:
        pdf_bytes: PDF document as bytes
    
    Returns:
        PNG image as bytes, or empty bytes if rendering fails
    """
    if not FITZ_AVAILABLE:
        return b""

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) == 0:
            return b""

        # Get first page
        page = doc[0]
        
        # Render page to pixmap (image)
        # Scale factor for better quality preview
        zoom = 2.0  # Increase for higher quality
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        
        doc.close()
        return img_bytes
        
    except Exception as e:
        print(f"⚠️ Error rendering PDF preview: {e}")
        return b""

def generate_pdf_report(df: pd.DataFrame,
                        summary: Dict[str, Any],
                        sim_period: str,
                        params: Dict[str, Any],
                        chart_selections: Dict[str, bool] = None) -> bytes:
    """
    Generate a multi-page PDF report with:
      - Vaillant logo on title page (if available)
      - Title + summary on page 1
      - Subsequent pages: 2×2 chart grids (4 charts / page)
      - Only includes charts selected by user
    
    Args:
        df: Simulation results dataframe
        summary: Dictionary of summary statistics
        sim_period: "24 Hours" or "7 Days"
        params: Dictionary of simulation parameters
        chart_selections: Dictionary of chart names to include (optional)
    
    Returns:
        PDF as bytes ready for download
    """

    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed – cannot generate PDF.")

    if not ALTAIR_EXPORT_AVAILABLE:
        raise RuntimeError("altair_saver/vl-convert is not installed – cannot export charts to PNG.")

    # Default to all charts if no selection provided
    if chart_selections is None:
        chart_selections = {
            "cop": True,
            "stratification": True,
            "power": True,
            "heat": True,
            "tank_temp": True,
            "tank_losses": True,
            "modulation": True,
            "tap": True,
            "soc": True,
            "legionella": True,
            "senso": True,
            "buffer": True,
        }

    charts = build_pdf_charts_selective(df, chart_selections)
    if not charts:
        raise RuntimeError("No charts selected for export. Please select at least one chart.")

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4

    # ---------- Page 1: Vaillant Logo + Title & summary ----------
    margin = 40
    y = page_height - margin

    # Try to add Vaillant logo at the top
    logo_loaded = False
    try:
        logo_path = "vaillant_logo.png"
        if os.path.exists(logo_path):
            img = ImageReader(logo_path)
            # Center the logo, scale to reasonable size
            logo_width = 150
            logo_height = 50
            x_logo = (page_width - logo_width) / 2
            c.drawImage(
                img,
                x_logo,
                y - logo_height,
                width=logo_width,
                height=logo_height,
                preserveAspectRatio=True,
                anchor="c",
            )
            y -= (logo_height + 20)
            logo_loaded = True
        else:
            print("ℹ️ Logo file 'vaillant_logo.png' not found - proceeding without logo")
    except Exception as e:
        print(f"⚠️ Could not load logo: {e}")
        # If logo not found, just skip it

    # Title (centered)
    title = "Vaillant 150 L Cylinder – Simulation Report"
    c.setFont("Helvetica-Bold", 18)
    title_width = c.stringWidth(title, "Helvetica-Bold", 18)
    c.drawString((page_width - title_width) / 2, y, title)
    y -= 30

    # Metadata
    c.setFont("Helvetica", 11)
    timestamp = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    c.drawString(margin, y, timestamp)
    y -= 18
    c.drawString(margin, y, f"Simulation Period: {sim_period}")
    y -= 24

    # Summary block
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Simulation Summary:")
    y -= 18
    c.setFont("Helvetica", 10)

    for key, val in summary.items():
        if isinstance(val, float):
            text = f"{key}: {val:.3f}"
        else:
            text = f"{key}: {val}"
        c.drawString(margin + 10, y, text)
        y -= 14
        if y < 80:  # avoid overflow
            c.showPage()
            y = page_height - margin
            c.setFont("Helvetica", 10)

    # Add selected charts info
    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, f"Report includes {len(charts)} selected chart(s)")
    y -= 20

    c.showPage()

    # ---------- Subsequent pages: 2×2 chart grid ----------
    # Layout: 2 columns × 2 rows per page
    cols = 2
    rows = 2
    cell_w = (page_width - 2 * margin) / cols
    cell_h = (page_height - 2 * margin) / rows

    chart_index = 0
    total_charts = len(charts)

    while chart_index < total_charts:
        for row in range(rows):
            for col in range(cols):
                if chart_index >= total_charts:
                    break

                chart = charts[chart_index]

                # Export Altair chart to PNG bytes
                img_buf = io.BytesIO()
                # Slightly higher scale for better print quality
                chart.save(img_buf, format="png", scale=2)
                img_buf.seek(0)

                img = ImageReader(img_buf)

                # Compute position
                x = margin + col * cell_w
                # reportlab coords start bottom-left
                y = margin + (rows - 1 - row) * cell_h

                # Keep aspect ratio – fit within cell
                c.drawImage(
                    img,
                    x,
                    y,
                    width=cell_w - 10,
                    height=cell_h - 10,
                    preserveAspectRatio=True,
                    anchor="c",
                )

                chart_index += 1

        c.showPage()

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes



    # ---------- Subsequent pages: 2×2 chart grid ----------
    # Layout: 2 columns × 2 rows per page
    cols = 2
    rows = 2
    cell_w = (page_width - 2 * margin) / cols
    cell_h = (page_height - 2 * margin) / rows

    chart_index = 0
    total_charts = len(charts)

    while chart_index < total_charts:
        for row in range(rows):
            for col in range(cols):
                if chart_index >= total_charts:
                    break

                chart = charts[chart_index]

                # Export Altair chart to PNG bytes
                img_buf = io.BytesIO()
                # Slightly higher scale for better print quality
                chart.save(img_buf, format="png", scale=2)
                img_buf.seek(0)

                img = ImageReader(img_buf)

                # Compute position
                x = margin + col * cell_w
                # reportlab coords start bottom-left
                y = margin + (rows - 1 - row) * cell_h

                # Keep aspect ratio – fit within cell
                c.drawImage(
                    img,
                    x,
                    y,
                    width=cell_w - 10,
                    height=cell_h - 10,
                    preserveAspectRatio=True,
                    anchor="c",
                )

                chart_index += 1

        c.showPage()

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ==============================================================
    # 📄 PDF Report Preview & Export
    # ==============================================================
    st.markdown("---")
    st.subheader("📄 PDF Report Preview & Export")

    # Pack the parameters
    params_for_pdf = st.session_state.get("last_params_dict", {})
    
    # --- Graph Selection Checkboxes ---
    st.markdown("#### Select Graphs to Include in PDF")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Core Metrics**")
        include_cop = st.checkbox("COP", value=True, key="pdf_cop")
        include_stratification = st.checkbox("Cylinder Stratification", value=True, key="pdf_strat")
        include_power = st.checkbox("HP Power", value=True, key="pdf_power")
        include_heat = st.checkbox("HP Heat Output", value=True, key="pdf_heat")
        
    with col2:
        st.markdown("**Temperature & Losses**")
        include_tank_temp = st.checkbox("Cylinder Avg Temperature", value=True, key="pdf_tank_temp")
        include_tank_losses = st.checkbox("Cylinder Heat Losses", value=True, key="pdf_tank_losses")
        include_modulation = st.checkbox("HP Modulation", value=True, key="pdf_mod")
        include_tap = st.checkbox("Tap Flow", value=True, key="pdf_tap")
        
    with col3:
        st.markdown("**Advanced Features**")
        include_soc = st.checkbox("Cylinder SOC", value=True, key="pdf_soc")
        include_legionella = st.checkbox("Legionella Cycle", value=legionella_enabled=="On", key="pdf_leg")
        include_senso = st.checkbox("SensoComfort", value=senso_enabled=="On", key="pdf_senso")
        include_buffer = st.checkbox("Buffer Charts", value=buffer_enabled=="On", key="pdf_buffer")

    # Store selections in a dict
    chart_selections = {
        "cop": include_cop,
        "stratification": include_stratification,
        "power": include_power,
        "heat": include_heat,
        "tank_temp": include_tank_temp,
        "tank_losses": include_tank_losses,
        "modulation": include_modulation,
        "tap": include_tap,
        "soc": include_soc,
        "legionella": include_legionella,
        "senso": include_senso,
        "buffer": include_buffer,
    }

    st.markdown("---")
    
    # --- Generate PDF Button ---
    col_btn, col_info = st.columns([1, 2])
    
    with col_btn:
        generate_clicked = st.button("📄 Generate PDF Report", key="btn_generate_pdf_unified")
    
    with col_info:
        num_selected = sum(chart_selections.values())
        st.info(f"📊 {num_selected} chart(s) selected for PDF export")

    # Generate PDF when button clicked
    if generate_clicked:
        if num_selected == 0:
            st.warning("⚠️ Please select at least one chart to include in the PDF.")
        else:
            try:
                with st.spinner("📄 Building PDF report with selected charts..."):
                    pdf_bytes = generate_pdf_report(
                        df=df,
                        summary=summary,
                        params=params_for_pdf,
                        sim_period=sim_period,
                        chart_selections=chart_selections,
                    )
                st.session_state["pdf_report_bytes"] = pdf_bytes
                st.success("✅ PDF report generated successfully!")
            except Exception as e:
                st.error(f"❌ Failed to generate PDF: {e}")
                with st.expander("🔍 Show error details"):
                    import traceback
                    st.code(traceback.format_exc())

    # --- Preview & Download ---
    if "pdf_report_bytes" in st.session_state:
        pdf_bytes = st.session_state["pdf_report_bytes"]
        
        st.markdown("#### 📋 Report Preview")
        
        # Preview first page as image (if PyMuPDF available)
        if FITZ_AVAILABLE:
            try:
                preview_img_bytes = render_pdf_preview_first_page(pdf_bytes)
                if preview_img_bytes:
                    st.image(
                        preview_img_bytes,
                        caption="PDF Preview – Page 1 (with Vaillant logo)",
                        use_column_width=True,
                    )
                else:
                    st.info("Preview rendering failed – but PDF should be valid for download.")
            except Exception as e:
                st.warning(f"Preview error: {e}")
        else:
            st.info("💡 Install PyMuPDF for preview: `pip install PyMuPDF`")

        # Download button
        st.markdown("---")
        filename = f"Vaillant_150L_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
        )
    else:
        st.info(
            "💡 **How to generate your report:**\n\n"
            "1. ✅ Select the graphs you want using the checkboxes above\n"
            "2. 🖱️ Click **'Generate PDF Report'** button\n"
            "3. 👁️ Preview will appear below\n"
            "4. ⬇️ Download your custom PDF report\n\n"
            "📋 The report includes:\n"
            "- Vaillant logo (if vaillant_logo.png exists)\n"
            "- Simulation summary and parameters\n"
            "- Your selected charts in a professional 2×2 grid layout"
        )
    # -----------------------
    # Title / cover section
    # -----------------------
    story.append(Paragraph("Vaillant 150 L Cylinder – Simulation Report", style_title))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            style_normal,
        )
    )
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Simulation period: {sim_period}", style_normal))
    story.append(Spacer(1, 24))

    # -----------------------
    # Simulation summary
    # -----------------------
    story.append(Paragraph("Simulation Summary", style_h1))
    story.append(Spacer(1, 6))

    summary_data = [["Metric", "Value"]]
    for k, v in summary.items():
        if isinstance(v, float):
            val = f"{v:.3f}"
        else:
            val = str(v)
        summary_data.append([k, val])

    summary_table = Table(summary_data, colWidths=[7 * cm, 8 * cm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 18))

    # -----------------------
    # Input parameters section
    # -----------------------
    story.append(Paragraph("Input Parameters (Sidebar Settings)", style_h1))
    story.append(Spacer(1, 6))

    if params is None:
        params = {}

    # Sort keys for stable output
    param_items = sorted(params.items(), key=lambda x: x[0])

    params_data = [["Parameter", "Value"]]
    for k, v in param_items:
        params_data.append([str(k), str(v)])

    params_table = Table(params_data, colWidths=[7 * cm, 8 * cm])
    params_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(params_table)
    story.append(Spacer(1, 24))

    # -----------------------
    # Daily breakdown (7 days)
    # -----------------------
    if sim_period == "7 Days" and "Time (h)" in df.columns:
        story.append(Paragraph("Daily Energy & Performance Breakdown", style_h1))
        story.append(Spacer(1, 6))

        max_day = int(df["Time (h)"].max() // 24)
        daily_rows = [["Day", "Energy (kWh)", "Heat (kWh)", "Losses (kWh)",
                    "Avg COP", "Tap Vol (L)", "HP Runtime (min)"]]

        total_energy = total_heat = total_losses = 0.0
        total_tap_vol = 0.0
        total_runtime = 0.0

        for day in range(max_day + 1):
            day_mask = (df["Time (h)"] >= day * 24) & (df["Time (h)"] < (day + 1) * 24)
            day_df = df[day_mask]
            if day_df.empty:
                continue

            dt_hours = day_df.get("dt_hours", pd.Series([0] * len(day_df)))
            energy_kWh = (day_df["HP Power (W)"] * dt_hours).sum() / 1000.0
            heat_kWh = (day_df["HP Heat (W)"] * dt_hours).sum() / 1000.0
            losses_kWh = (day_df["Q_Loss (W)"] * dt_hours).sum() / 1000.0
            tap_vol_L = (day_df["Tap Flow (L/min)"] * (dt_hours * 60)).sum()
            runtime_min = (day_df["HP_On"] * dt_hours * 60).sum()
            avg_cop = heat_kWh / energy_kWh if energy_kWh > 0 else 0.0

            total_energy += energy_kWh
            total_heat += heat_kWh
            total_losses += losses_kWh
            total_tap_vol += tap_vol_L
            total_runtime += runtime_min

            daily_rows.append(
                [
                    f"Day {day + 1}",
                    f"{energy_kWh:.3f}",
                    f"{heat_kWh:.3f}",
                    f"{losses_kWh:.3f}",
                    f"{avg_cop:.2f}",
                    f"{tap_vol_L:.1f}",
                    f"{runtime_min:.0f}",
                ]
            )

        if total_energy > 0:
            weekly_cop = total_heat / total_energy
        else:
            weekly_cop = 0.0

        daily_rows.append(
            [
                "WEEK TOTAL",
                f"{total_energy:.3f}",
                f"{total_heat:.3f}",
                f"{total_losses:.3f}",
                f"{weekly_cop:.2f}",
                f"{total_tap_vol:.1f}",
                f"{total_runtime:.0f}",
            ]
        )

        daily_table = Table(daily_rows, colWidths=[2.5 * cm, 2.8 * cm, 2.8 * cm,
                                                2.8 * cm, 2.2 * cm, 2.8 * cm, 3 * cm])
        daily_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, -1), (-1, -1), colors.whitesmoke),
                ]
            )
        )
        story.append(daily_table)
        story.append(Spacer(1, 24))

    # -----------------------
    # Helper: add Altair chart as PNG
    # -----------------------
    def add_chart(title: str, chart):
        if chart is None:
            return
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                # Requires kaleido / altair_saver installed
                chart.save(tmp.name, format="png", scale=2)
                img = RLImage(tmp.name, width=16 * cm)
            story.append(Paragraph(title, style_h2))
            story.append(Spacer(1, 6))
            story.append(img)
            story.append(Spacer(1, 18))
        except Exception as e:
            # Fall back to text if rendering fails
            story.append(
                Paragraph(f"{title} (chart could not be rendered: {e})", style_normal)
            )
            story.append(Spacer(1, 12))

    # -----------------------
    # Key charts pages
    # -----------------------
    # 1. COP
    add_chart("Coefficient of Performance (COP)", chart_cop(df))

    # 2. Cylinder Stratification
    add_chart("Cylinder Stratification (Multi-Layer)", chart_stratification(df))

    # 3. Power & Heat
    add_chart("Heat Pump Electrical Power (W)", chart_power(df))
    add_chart("Heat Pump Heat Output (W)", chart_heat(df))

    # 4. Cylinder temperatures & losses
    add_chart("Cylinder Average Temperature vs Setpoint", chart_tank_temperature(df))
    add_chart("Cylinder Heat Losses (W)", chart_tank_losses(df))

    # 5. Modulation & tapping
    add_chart("HP Modulation & On/Off State", chart_modulation(df))
    add_chart("Tap Flow (L/min)", chart_tap(df))

    # 6. SOC & buffers (if available)
    add_chart("Cylinder State of Charge", chart_cylinder_soc(df))

    if "Buffer Avg Temp (°C)" in df.columns:
        # These chart_* functions are already defined in your code
        add_chart("Buffer Stratification (4-Layer)", chart_buffer_strat(df))
        add_chart("Buffer State of Charge (HP vs Immersion)", chart_soc(df))
        add_chart("Buffer Tank Average Temperature", chart_buffer_temperature(df))
        add_chart("Buffer Tank Heat Losses (W)", chart_buffer_losses(df))

    # 7. Legionella & SensoComfort (if enabled)
    if "Legionella_Active" in df.columns:
        leg_chart = chart_legionella(df)
        if leg_chart is not None:
            add_chart("Legionella Cycle Activity", leg_chart)

    add_chart("SensoComfort Control: Temperature vs Setpoint", chart_sensocomfort(df))

    # -----------------------
    # Build PDF
    # -----------------------
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes





                            

# --- Entry point ---
if __name__ == "__main__":
    main()