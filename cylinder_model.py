# --- cylinder_model_multilayer.py ---
# Multi-layer stratified tank + modulating HP simulation
# Includes summary, dropdown graph viewer, AI assistant, and tapping validation

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import os

import streamlit as st
import base64

st.set_page_config(page_title="Vaillant Cylinder Model", layout="wide")

# Load and encode your logo
with open("src/src/vaillant_logo.png", "rb") as file:
    data = base64.b64encode(file.read()).decode("utf-8")

# Use an f-string to insert the base64 image data
st.markdown(
    f"""
    <div style="text-align: center; padding-top: 10px; padding-bottom: 10px;">
        <img src="data:image/png;base64,{data}" width="200">
        <h1 style="margin-top: 10px;">Vaillant Cylinder Model Simulation</h1>
    </div>
    <hr style="border:1px solid #ccc;">
    """,
    unsafe_allow_html=True
)




# Optional AI assistant
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
except Exception:
    client = None

# --- Global constants ---
rho = 1000.0
c_p = 4186.0
P_EL_MAX = 5000.0  # W nominal


# --- Empirical Heat Pump model ---
def solve_hp(T_tank, mod, Pmax, Pmin, Pmin_ratio):
    if mod < Pmin_ratio:
        Pel = 0.0
    else:
        frac = (mod - Pmin_ratio) / (1 - Pmin_ratio)
        Pel = Pmin + (Pmax - Pmin) * frac
    COP = max(4.0 - 0.1 * (T_tank - 45.0), 1.0)
    Qhp = Pel * COP
    return Qhp, Pel, COP


# --- Simulation function ---
@st.cache_data
def run_sim(dt_min, sim_hrs, Vtank, Utank, Tamb, setp, hyst,
             Pmin, Pmax, Tsrc, tap, N_layers=5):
    """Multi-layer stratified tank simulation with modulating HP."""

    # --- Validate tapping data ---
    if not all(k in tap for k in ["time", "volume", "rate_lpm"]):
        raise ValueError("Tapping data missing required keys (time, volume, rate_lpm).")
    if len(tap["time"]) == 0:
        st.warning("⚠️ No tapping events defined. Running without hot water draws.")

    # Convert tapping lists → NumPy arrays
    tap_time = np.array(tap["time"])
    tap_vol = np.array(tap["volume"])
    tap_rate = np.array(tap["rate_lpm"])

    # --- Derived constants ---
    Vtank_m3 = Vtank / 1000
    m_total = rho * Vtank_m3
    m_layer = m_total / N_layers
    dt_s = dt_min * 60
    steps = int(sim_hrs * 60 / dt_min)
    time = np.arange(0, steps * dt_min, dt_min) / 60

    # --- Geometry & heat transfer ---
    H = 1.5
    D = np.sqrt(Vtank_m3 / (H * np.pi / 4))
    A_cross = np.pi * (D / 2)**2
    A_side = np.pi * D * H
    A_total = A_side + 2 * A_cross
    UA_loss = Utank * A_total / N_layers
    U_int = 50.0
    A_int = A_cross
    UA_int = U_int * A_int / (H / N_layers)

    # --- HP control setup ---
    Pmax_W = P_EL_MAX * Pmax
    Pmin_W = Pmax_W * Pmin
    Ton = setp - hyst
    coil_bottom_idx = max(1, int(N_layers * 0.3))  # inject in bottom 30%

    # --- Initialize arrays ---
    T = np.zeros((steps, N_layers))
    Qhp = np.zeros(steps)
    Pel = np.zeros(steps)
    COPs = np.zeros(steps)
    HPon = np.zeros(steps, bool)
    Qloss_total = np.zeros(steps)
    Tap_flow = np.zeros(steps)
    Tap_temp = np.zeros(steps)
    Mod_frac = np.zeros(steps)

    # Initial temperature gradient
    for i in range(N_layers):
        T[0, i] = setp - 5 * (i / (N_layers - 1))

    # --- Main simulation loop ---
    for k in range(1, steps):
        T_prev = T[k - 1, :].copy()
        T_new = T_prev.copy()
        T_top = T_prev[-1]
        T_bottom = T_prev[0]
        tnow = k * dt_min

        # 1. HP control
        mod = 0
        on = False
        if T_top < setp:
            on = True
            if T_top <= Ton:
                mod = 1
            else:
                raw = (setp - T_top) / hyst
                mod = Pmin + (1 - Pmin) * np.clip(raw, 0, 1)
        if mod < Pmin:
            on = False
            mod = 0
        Mod_frac[k] = mod

        Q = Pe = COP = 0
        if on:
            Q, Pe, COP = solve_hp(T_bottom, mod, Pmax_W, Pmin_W, Pmin)
        HPon[k] = on
        Qhp[k] = Q
        Pel[k] = Pe
        COPs[k] = COP

        # 2. Tapping (top draw, bottom refill)
        idx = np.where((tap_time > tnow - dt_min) & (tap_time <= tnow))[0]
        Vtap = np.sum(tap_vol[idx]) if idx.size > 0 else 0
        mdot_tap = (Vtap / 1000) * rho / dt_s
        Tap_flow[k] = np.sum(tap_rate[idx]) if idx.size > 0 else 0
        Tap_temp[k] = T_top

        # 3. Layer balances
        for i in range(N_layers):
            Q_net = 0.0
            if i > 0:
                Q_up = UA_int * (T_prev[i - 1] - T_prev[i])
                Q_net += Q_up
            if i < N_layers - 1:
                Q_down = UA_int * (T_prev[i + 1] - T_prev[i])
                Q_net += Q_down
            Q_loss = UA_loss * (T_prev[i] - Tamb)
            Q_net -= Q_loss
            Qloss_total[k] += Q_loss
            if i < coil_bottom_idx:
                Q_net += Qhp[k] / coil_bottom_idx
            if i == N_layers - 1:
                Q_net -= mdot_tap * c_p * T_prev[i]
            elif i == 0:
                Q_net += mdot_tap * c_p * Tamb
            dT = Q_net * dt_s / (m_layer * c_p)
            T_new[i] = np.clip(T_prev[i] + dT, 0, 100)

        # Enforce stratification
        for i in range(N_layers - 1):
            if T_new[i] > T_new[i + 1]:
                avg = 0.5 * (T_new[i] + T_new[i + 1])
                T_new[i] = avg
                T_new[i + 1] = avg
        T[k, :] = T_new

    # --- Results ---
    df = pd.DataFrame({
        "Time (h)": time,
        "HP Power (W)": Pel,
        "HP Heat (W)": Qhp,
        "COP": COPs,
        "HP_On": HPon,
        "Modulation": Mod_frac,
        "Tap Flow (L/min)": Tap_flow,
        "Tap Temp (°C)": Tap_temp,
        "Q_Loss (W)": Qloss_total,
    })
    for i in range(N_layers):
        df[f"T_Layer{i+1} (°C)"] = T[:, i]
    df["T_Avg (°C)"] = T.mean(axis=1)

    # Energy summary
    dt_s = dt_min * 60
    total_heat_kWh = df["HP Heat (W)"].sum() * dt_s / 3.6e6
    total_power_kWh = df["HP Power (W)"].sum() * dt_s / 3.6e6
    total_losses_kWh = df["Q_Loss (W)"].sum() * dt_s / 3.6e6
    hp_runtime_min = np.sum(df["HP_On"]) * dt_min
    avg_cop = total_heat_kWh / total_power_kWh if total_power_kWh > 0 else 0

    summary = {
        "Simulation Hours": sim_hrs,
        "Tank Volume (L)": Vtank,
        "Total HP Electrical Energy (kWh)": total_power_kWh,
        "Total Heat Delivered by HP (kWh)": total_heat_kWh,
        "Total Heat from Coil (kWh)": total_heat_kWh,
        "Total Losses (kWh)": total_losses_kWh,
        "HP Run Time (minutes)": hp_runtime_min,
        "Average COP": avg_cop,
    }
    return df, summary


# --- Charts ---
def chart_stratification(df):
    cols = [c for c in df.columns if c.startswith("T_Layer")]
    d = df.melt("Time (h)", value_vars=cols, var_name="Layer", value_name="Temp (°C)")
    return alt.Chart(d).mark_line().encode(
        x="Time (h)", y="Temp (°C)", color="Layer:N"
    ).properties(height=400, title="Tank Stratification (Multi-Layer)").interactive()


def chart_modulation(df):
    base = alt.Chart(df).mark_area(opacity=0.5, color="#2563eb").encode(
        x="Time (h)",
        y=alt.Y("Modulation:Q", title="Modulation Fraction (0–1)"),
        tooltip=["Time (h)", "Modulation"]
    )

    on_overlay = alt.Chart(df).mark_line(color="#22c55e", strokeWidth=2).encode(
        x="Time (h)",
        y=alt.Y("HP_On:Q", title="HP On/Off"),
        tooltip=["Time (h)", "HP_On"]
    )

    # IDE-style inline labels
    mod_label = alt.Chart(pd.DataFrame({"x": [df["Time (h)"].max() * 0.95],
                                        "y": [0.9],
                                        "text": ["Modulation Fraction"]})).mark_text(
        color="#2563eb", align="right", dx=-5, dy=-5, fontSize=12
    ).encode(x="x", y="y", text="text")

    on_label = alt.Chart(pd.DataFrame({"x": [df["Time (h)"].max() * 0.95],
                                       "y": [1.05],
                                       "text": ["HP On/Off State"]})).mark_text(
        color="#22c55e", align="right", dx=-5, dy=5, fontSize=12
    ).encode(x="x", y="y", text="text")

    return (
        base + on_overlay + mod_label + on_label
    ).properties(
        height=280,
        title="Heat Pump Modulation & On/Off State (with Labels)"
    ).interactive()



def chart_power(df):
    return alt.Chart(df).mark_line(color="#ef4444").encode(
        x="Time (h)", y="HP Power (W)"
    ).properties(height=300, title="Heat Pump Electrical Power (W)")


def chart_heat(df):
    return alt.Chart(df).mark_line(color="#1d4ed8").encode(
        x="Time (h)", y="HP Heat (W)"
    ).properties(height=300, title="Heat Pump Heat Output (W)")


def chart_cop(df):
    return alt.Chart(df).mark_line(color="green").encode(
        x="Time (h)", y="COP"
    ).properties(height=300, title="Coefficient of Performance (COP)")


def chart_tap(df):
    return alt.Chart(df).mark_bar(color="#f59e0b").encode(
        x="Time (h)", y="Tap Flow (L/min)"
    ).properties(height=200, title="Tap Flow (L/min)")


# --- Streamlit UI ---
def main():

    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    Tsrc = st.sidebar.slider("Heat Source Temp (°C)", -5.0, 15.0, 5.0, 0.5)
    Vtank = st.sidebar.slider("Tank Volume (L)", 100, 500, 300, 10)
    Utank = st.sidebar.slider("Tank U-value (W/m²K)", 0.1, 2.0, 0.4, 0.1)
    Tamb = st.sidebar.slider("Ambient Temp (°C)", 0, 30, 15)
    simh = st.sidebar.slider("Simulation Duration (h)", 1, 48, 24)
    dtm = st.sidebar.slider("Time Step (min)", 1, 10, 5)
    setp = st.sidebar.slider("Setpoint (°C)", 40, 60, 50)
    hyst = st.sidebar.slider("Hysteresis (°C)", 1.0, 10.0, 5.0, 0.5)
    Pmax = st.sidebar.slider("Max HP Power Ratio", 0.5, 1.0, 1.0, 0.05)
    Pmin = st.sidebar.slider("Min Modulation Ratio", 0.1, 0.5, 0.2, 0.05)
    Nlayers = st.sidebar.slider("Number of Tank Layers", 2, 10, 5, 1)

    # Simple tapping schedule
    raw = [
        {"Hour": 7, "Minute": 0, "Flow_lpm": 3, "Duration_sec": 45},
        {"Hour": 7, "Minute": 5, "Flow_lpm": 6, "Duration_sec": 300},
        {"Hour": 20, "Minute": 30, "Flow_lpm": 4, "Duration_sec": 240},
        {"Hour": 21, "Minute": 30, "Flow_lpm": 6, "Duration_sec": 300},
    ]
    tap = {"time": [], "volume": [], "rate_lpm": []}
    for e in raw:
        tmin = e["Hour"] * 60 + e["Minute"]
        vol = e["Flow_lpm"] * e["Duration_sec"] / 60
        tap["time"].append(tmin)
        tap["volume"].append(vol)
        tap["rate_lpm"].append(e["Flow_lpm"])

    # Run simulation automatically
    df, summary = run_sim(dtm, simh, Vtank, Utank, Tamb, setp, hyst, Pmin, Pmax, Tsrc, tap, Nlayers)

    # --- Summary ---
    st.subheader("📊 Simulation Summary")
    for key, val in summary.items():
        st.write(f"**{key}:** {val:.2f}" if isinstance(val, float) else f"**{key}:** {val}")
    st.markdown("---")

    # --- Graph viewer ---
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

    # --- AI Assistant ---
    st.markdown("---")
    st.subheader("💬 AI Simulation Assistant")
    st.write("Ask: *Why did stratification change?*, *How to improve COP?*, etc.")

    q = st.text_input("Your question:")
    if q:
        if client is None:
            st.warning("⚠️ OpenAI API key not set.")
        else:
            prompt = f"""
            You are an energy systems expert analyzing a multi-layer stratified tank and modulating heat pump simulation.

            Summary:
            {summary}

            Columns: {', '.join(df.columns)}

            User question: {q}
            Provide a concise, technical answer.
            """
            with st.spinner("Analyzing..."):
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
            st.success(r.choices[0].message.content)


if __name__ == "__main__":
    main()
