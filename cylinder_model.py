import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from openai import OpenAI
import json
import re

# --------------------- STREAMLIT CONFIG ---------------------
st.set_page_config(page_title="Vaillant cylinder simlation", layout="wide")
st.title("Vaillant cylinder simlation")

# --------------------- PARAMETER PARSING FUNCTION ---------------------
def parse_parameter_change(prompt: str):
    patterns = {
        "setpoint": r"setpoint\s*(?:to|=)?\s*(\d+\.?\d*)",
        "tank_volume_l": r"(?:tank volume|volume)\s*(?:to|=)?\s*(\d+\.?\d*)",
        "T_ambient": r"(?:ambient temp|ambient temperature)\s*(?:to|=)?\s*(-?\d+\.?\d*)",
        "T_coil_in": r"(?:coil inlet temp|coil temperature)\s*(?:to|=)?\s*(\d+\.?\d*)",
        "sim_hours": r"(?:simulation hours|hours)\s*(?:to|=)?\s*(\d+\.?\d*)",
    }
    updates = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, prompt.lower())
        if match:
            try:
                updates[key] = float(match.group(1))
            except ValueError:
                pass
    return updates

# --------------------- USER INPUTS ---------------------
st.sidebar.header("Simulation Parameters")
dt_min = st.sidebar.number_input("Time step (minutes)", 1, 60, 1)
sim_hours = st.sidebar.slider("Simulation hours", 1, 72, 24)
tank_volume_l = st.sidebar.number_input("Tank volume (L)", 50, 1000, 200)
tank_height = st.sidebar.number_input("Tank height (m)", 0.5, 3.0, 1.2)
U_tank = st.sidebar.slider("Tank insulation U-value (W/m²K)", 0.5, 10.0, 3.0)
T_ambient = st.sidebar.number_input("Ambient temperature (°C)", -10.0, 30.0, 10.0)
setpoint = st.sidebar.slider("Tank setpoint (°C)", 30.0, 70.0, 50.0)
hysteresis = st.sidebar.slider("Thermostat hysteresis (°C)", 1.0, 10.0, 3.0)

st.sidebar.subheader("Coil Parameters")
coil_area = st.sidebar.number_input("Coil area (m²)", 0.1, 10.0, 1.5)
U_ref = st.sidebar.number_input("Reference U (W/m²K)", 100.0, 2000.0, 800.0)
flow_ref = st.sidebar.number_input("Reference flow rate (L/min)", 1.0, 50.0, 10.0)
coil_flow_rate_lpm = st.sidebar.number_input("Actual coil flow rate (L/min)", 1.0, 50.0, 10.0)
T_coil_in = st.sidebar.number_input("Coil inlet temperature (°C)", 10.0, 80.0, 55.0)
n_flow = 0.6

st.sidebar.subheader("Heat Pump Parameters")
P_el_max = 2000.0
COP_nominal = 3.5
T_ref_sink = 45.0
COP_slope_sink = 0.03
COP_slope_source = 0.02
T_source = 5.0
T_cold = 10.0

# --------------------- HOT WATER DRAWS ---------------------
st.sidebar.subheader("Hot Water Draws")

if "draws" not in st.session_state:
    st.session_state.draws = []

with st.sidebar.form("draw_form", clear_on_submit=True):
    new_hour = st.number_input("Hour (0–23)", 0, 23, 7, key="hour_input")
    new_minute = st.number_input("Minute (0–59)", 0, 59, 0, key="minute_input")
    new_volume = st.number_input("Volume (L)", 1, 500, 50, key="volume_input")
    add_clicked = st.form_submit_button("➕ Add Draw")

if add_clicked:
    st.session_state.draws.append((int(new_hour), int(new_minute), float(new_volume)))

if st.sidebar.button("🗑️ Clear All Draws"):
    st.session_state.draws = []

if st.session_state.draws:
    df_draws = pd.DataFrame(st.session_state.draws, columns=["Hour", "Minute", "Volume (L)"])
    st.sidebar.dataframe(df_draws, use_container_width=True, hide_index=True)
else:
    st.sidebar.info("No draws added yet.")

draws = st.session_state.draws

# --------------------- FIXED PARAMETERS ---------------------
rho = 1000.0
c_p = 4186.0
tank_volume_m3 = tank_volume_l / 1000.0
tank_radius = np.sqrt(tank_volume_m3 / (np.pi * tank_height))
tank_area = 2 * np.pi * tank_radius * tank_height + 2 * np.pi * tank_radius**2
tank_mass = tank_volume_m3 * rho
UA_tank = U_tank * tank_area

coil_U = U_ref * (coil_flow_rate_lpm / flow_ref) ** n_flow
coil_UA = coil_U * coil_area
m_dot_coil = coil_flow_rate_lpm / 60.0 / 1000.0 * rho

# --------------------- SIMULATION ---------------------
steps = int(sim_hours * 60 / dt_min)
time = np.arange(0, steps * dt_min, dt_min) / 60.0

T_tank = np.zeros(steps)
hp_power_el = np.zeros(steps)
hp_on = np.zeros(steps, dtype=bool)
Q_from_hp = np.zeros(steps)
Q_losses = np.zeros(steps)
COP_series = np.zeros(steps)
Q_coil_series = np.zeros(steps)
T_coil_out_series = np.zeros(steps)
T_tank[0] = 45.0

draw_events = {}
for h, m, v in draws:
    idx = int((h * 60 + m) / dt_min)
    if idx < steps:
        draw_events.setdefault(idx, 0.0)
        draw_events[idx] += v

for k in range(1, steps):
    T_prev = T_tank[k - 1]

    # Hot water draw
    if k in draw_events:
        vol = draw_events[k]
        if vol >= tank_volume_l:
            T_prev = T_cold
        else:
            m_tank = tank_mass
            m_draw = vol / 1000.0 * rho
            m_remain = m_tank - m_draw
            E_remain = m_remain * c_p * T_prev
            E_in = m_draw * c_p * T_cold
            T_prev = (E_remain + E_in) / (m_tank * c_p)

    # Thermostat
    if T_prev <= (setpoint - hysteresis):
        hp_should_on = True
    elif T_prev >= setpoint:
        hp_should_on = False
    else:
        hp_should_on = bool(hp_on[k - 1])

    # COP model
    COP = COP_nominal - COP_slope_sink * (T_prev - T_ref_sink) + COP_slope_source * (T_source - 5.0)
    COP = max(1.0, COP)
    COP_series[k] = COP

    # Heat pump operation
    if hp_should_on:
        Q_dot_hp = P_el_max * COP
        P_el = P_el_max
        hp_on[k] = True
    else:
        Q_dot_hp = 0.0
        P_el = 0.0
        hp_on[k] = False

    Q_loss = UA_tank * (T_prev - T_ambient)
    Q_dot_coil = coil_UA * (T_coil_in - T_prev)
    if m_dot_coil > 0:
        T_coil_out = T_coil_in - Q_dot_coil / (m_dot_coil * c_p)
    else:
        T_coil_out = T_coil_in

    Q_net = Q_dot_hp + Q_dot_coil - Q_loss
    dt_s = dt_min * 60.0
    dT = (Q_net * dt_s) / (tank_mass * c_p)
    T_new = T_prev + dT

    T_tank[k] = T_new
    hp_power_el[k] = P_el
    Q_from_hp[k] = Q_dot_hp
    Q_losses[k] = Q_loss
    Q_coil_series[k] = Q_dot_coil
    T_coil_out_series[k] = T_coil_out

# --------------------- POSTPROCESS ---------------------
total_el_kwh = np.sum(hp_power_el) * (dt_min / 60.0) / 1000.0
total_heat_kwh = np.sum(Q_from_hp) * (dt_min / 60.0) / 1000.0
total_losses_kwh = np.sum(Q_losses) * (dt_min / 60.0) / 1000.0
total_coil_kwh = np.sum(Q_coil_series) * (dt_min / 60.0) / 1000.0
on_minutes = np.sum(hp_on)

summary = {
    "Simulation hours": sim_hours,
    "Tank volume (L)": tank_volume_l,
    "Total HP electrical energy (kWh)": round(total_el_kwh, 3),
    "Total heat delivered by HP (kWh)": round(total_heat_kwh, 3),
    "Total heat from coil (kWh)": round(total_coil_kwh, 3),
    "Total losses (kWh)": round(total_losses_kwh, 3),
    "HP run time (minutes)": int(on_minutes),
    "Average COP": round((total_heat_kwh / total_el_kwh) if total_el_kwh > 0 else 0.0, 2),
}
st.subheader("Simulation Summary")
st.dataframe(pd.DataFrame.from_dict(summary, orient="index", columns=["Value"]))

# --------------------- DATAFRAMES FOR PLOTTING ---------------------
df = pd.DataFrame({
    "Time (hours)": time,
    "Tank Temperature (°C)": T_tank,
    "HP Power (W)": hp_power_el,
    "COP": COP_series,
    "Losses (W)": Q_losses,
    "Q_from_HP (W)": Q_from_hp,
    "Q_from_Coil (W)": Q_coil_series,
    "Coil Outlet Temp (°C)": T_coil_out_series
})
df["Energy_HP (kWh)"] = np.cumsum(Q_from_hp) * (dt_min / 60.0) / 1000.0
df["Energy_Coil (kWh)"] = np.cumsum(Q_coil_series) * (dt_min / 60.0) / 1000.0
df_energy = df.melt(
    id_vars=["Time (hours)"],
    value_vars=["Energy_HP (kWh)", "Energy_Coil (kWh)"],
    var_name="Source",
    value_name="Cumulative Energy (kWh)"
)

# --------------------- CHART SELECTION ---------------------
chart_mode = st.sidebar.selectbox(
    "Choose chart mode",
    options=["Tank Temperature", "Heat pump power", "COP vs Time", "Coil Temps", "Energy", "Losses"]
)

def reset_zoom_button():
    if st.button("🔄 Reset Zoom"):
        st.experimental_rerun()

# --------------------- PLOTTING ---------------------
if chart_mode == "Tank Temperature":
    chart_temp = alt.Chart(df).mark_line(color="orange").encode(
        x="Time (hours)", y="Tank Temperature (°C)",
        tooltip=["Time (hours)", "Tank Temperature (°C)"]
    ).properties(title="Tank Temperature vs Time", width=750, height=300).interactive()

    st.altair_chart(chart_temp, use_container_width=True)
    reset_zoom_button()

    # Separate heat pump power chart
elif chart_mode == "Heat pump power":
    chart_power = alt.Chart(df).mark_line(color="red").encode(
        x="Time (hours)", y="HP Power (W)",
        tooltip=["Time (hours)", "HP Power (W)"]
    ).properties(title="Heat Pump Power vs Time", width=750, height=300).interactive()

    st.altair_chart(chart_power, use_container_width=True)
    reset_zoom_button()

elif chart_mode == "COP vs Time":
    chart_cop = alt.Chart(df).mark_line(color="green", strokeWidth=2).encode(
        x="Time (hours)", y="COP",
        tooltip=["Time (hours)", "COP"]
    ).properties(title="Heat Pump COP vs Time", width=750, height=400).interactive()
    st.altair_chart(chart_cop, use_container_width=True)
    reset_zoom_button()

elif chart_mode == "Coil Temps":
    coil_temp_long = df.melt(
        id_vars=["Time (hours)"], value_vars=["Coil Outlet Temp (°C)"],
        var_name="Temperature Type", value_name="Temperature (°C)"
    )
    chart_coil_temps = alt.Chart(coil_temp_long).mark_line(strokeWidth=2).encode(
        x="Time (hours)", y="Temperature (°C)", color="Temperature Type",
        tooltip=["Time (hours)", "Temperature (°C)"]
    ).properties(title="Coil Outlet Temperature vs Time", width=750, height=400).interactive()
    st.altair_chart(chart_coil_temps, use_container_width=True)
    reset_zoom_button()

elif chart_mode == "Energy":
    chart_stacked = alt.Chart(df_energy).mark_area(opacity=0.8).encode(
        x="Time (hours)", y="Cumulative Energy (kWh)", color="Source",
        tooltip=["Time (hours)", "Source", "Cumulative Energy (kWh)"]
    ).properties(title="Cumulative Energy from Heat Pump and Coil", width=750, height=400).interactive()
    st.altair_chart(chart_stacked, use_container_width=True)
    reset_zoom_button()

elif chart_mode == "Losses":
    chart_losses = alt.Chart(df).mark_line(color="brown", strokeWidth=2).encode(
        x="Time (hours)", y="Losses (W)",
        tooltip=["Time (hours)", "Losses (W)"]
    ).properties(title="Tank Losses vs Time", width=750, height=400).interactive()
    st.altair_chart(chart_losses, use_container_width=True)
    reset_zoom_button()

    # --------------------- DATAFRAMES FOR PLOTTING ---------------------
df = pd.DataFrame({
    "Time (hours)": time,
    "Tank Temperature (°C)": T_tank,
    "HP Power (W)": hp_power_el,
    "COP": COP_series,
    "Losses (W)": Q_losses,
    "Q_from_HP (W)": Q_from_hp,
    "Q_from_Coil (W)": Q_coil_series,
    "Coil Outlet Temp (°C)": T_coil_out_series,
})
df["Energy_HP (kWh)"] = np.cumsum(Q_from_hp) * (dt_min / 60.0) / 1000.0
df["Energy_Coil (kWh)"] = np.cumsum(Q_coil_series) * (dt_min / 60.0) / 1000.0
df_energy = df.melt(id_vars=["Time (hours)"], value_vars=["Energy_HP (kWh)", "Energy_Coil (kWh)"],
                    var_name="Source", value_name="Cumulative Energy (kWh)")

df_cop = df[["Time (hours)", "COP"]]
df_temp = df[["Time (hours)", "Tank Temperature (°C)"]]
df_power = df[["Time (hours)", "HP Power (W)"]]
df_losses = df[["Time (hours)", "Losses (W)"]]
df_coil = df[["Time (hours)", "Coil Outlet Temp (°C)", "Q_from_Coil (W)"]]

    # --------------------- CYLINDER TAPPINGS UPLOAD ---------------------
st.subheader("🧾 Cylinder Tappings Upload and Visualization")
uploaded_tappings = st.file_uploader("Upload Cylinder Tappings CSV", type=["csv"])
df_tappings = None
if uploaded_tappings:
    try:
        df_tappings = pd.read_csv(uploaded_tappings)
        st.success("✅ Cylinder tapping data loaded successfully.")
        st.dataframe(df_tappings, use_container_width=True)
        height_col = next(c for c in df_tappings.columns if "height" in c.lower())
        temp_col = next(c for c in df_tappings.columns if "temp" in c.lower())
        chart_tappings = alt.Chart(df_tappings).mark_point(color="blue").encode(
            x=alt.X(f"{temp_col}:Q", title="Temperature (°C)"),
            y=alt.Y(f"{height_col}:Q", title="Height (m)")
        ).properties(title="Measured Cylinder Temperature Profile", width=600, height=400)
        st.altair_chart(chart_tappings, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

    # --------------------- HELPER: CHART GENERATOR ---------------------
def generate_chart(key: str):
    if key == "temperature":
        base = alt.Chart(df).mark_line(color="red").encode(x="Time (hours)", y="Tank Temperature (°C)")
        if df_tappings is not None:
            base += alt.Chart(df_tappings).mark_point(color="blue").encode(
                x=alt.X("Temperature:Q", title="Temperature (°C)"),
                y=alt.Y("Height:Q", title="Height (m)")
            )
        return base
    elif key == "cop":
        return alt.Chart(df_cop).mark_line(color="green").encode(x="Time (hours)", y="COP")
    elif key == "power":
        return alt.Chart(df_power).mark_line(color="orange").encode(x="Time (hours)", y="HP Power (W)")
    elif key == "coil":
        return alt.Chart(df_coil).mark_line(color="purple").encode(x="Time (hours)", y="Coil Outlet Temp (°C)")
    elif key == "loss":
        return alt.Chart(df_losses).mark_line(color="gray").encode(x="Time (hours)", y="Losses (W)")
    elif key == "energy":
        return alt.Chart(df_energy).mark_line().encode(x="Time (hours)", y="Cumulative Energy (kWh)", color="Source")
    return None

# --------------------- GPT-POWERED INTERACTIVE CHATBOT ---------------------
st.subheader("🤖 Interactive Simulation Copilot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "simulation_updates" not in st.session_state:
    st.session_state.simulation_updates = {}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

system_prompt = f"""
You are an expert assistant for hot water tank simulations.
You can explain results, detect patterns, compare measured and simulated data,
and interpret efficiency behavior from the charts.
Here is the latest simulation summary:
{json.dumps(summary, indent=2)}
"""

prompt = st.chat_input("Ask about the simulation or say 'show COP chart'...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    updates = parse_parameter_change(prompt)
    if updates:
        st.session_state.simulation_updates.update(updates)
        msg = f"Detected parameter change: {updates}. 🔁 Rerunning simulation..."
        st.chat_message("assistant").markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.experimental_rerun()
    elif any(word in prompt.lower() for word in ["chart", "plot", "graph", "show"]):
        for key in ["temperature", "cop", "power", "coil", "loss", "energy"]:
            if key in prompt.lower():
                chart = generate_chart(key)
                if chart:
                    st.chat_message("assistant").markdown(f"📊 Here's the **{key.capitalize()}** chart:")
                    st.altair_chart(chart, use_container_width=True)
                    explanation = f"I've plotted the {key} data — {('showing both simulated and measured data' if df_tappings is not None else 'based on the simulation results only')}."
                    st.chat_message("assistant").markdown(explanation)
                    st.session_state.messages.append({"role": "assistant", "content": explanation})
                    break
    elif df_tappings is not None and "compare" in prompt.lower():
        avg_sim = df_temp["Tank Temperature (°C)"].mean()
        avg_meas = df_tappings.iloc[:, 1].mean()
        diff = avg_meas - avg_sim
        diag = (
            f"The measured temperatures average {avg_meas:.1f}°C vs simulated {avg_sim:.1f}°C "
            f"→ difference of {diff:+.1f}°C. "
            + ("Measured tank is warmer — possible calibration or stratification effect." if diff > 0 else
               "Measured tank is cooler — could indicate higher losses or draw rates.")
        )
        st.chat_message("assistant").markdown("📈 Comparison between measured and simulated data:")
        st.chat_message("assistant").markdown(diag)
        st.session_state.messages.append({"role": "assistant", "content": diag})
    else:
        messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=400,
                    temperature=0.5
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
