import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --------------------- STREAMLIT CONFIG ---------------------
st.title("Hot Water Tank + Heat Pump Simulation")

# User inputs (sliders and number inputs)
dt_min = st.sidebar.number_input("Time step (minutes)", 1, 60, 1)
sim_hours = st.sidebar.slider("Simulation hours", 1, 72, 24)
tank_volume_l = st.sidebar.number_input("Tank volume (L)", 50, 1000, 200)
tank_height = st.sidebar.number_input("Tank height (m)", 0.5, 3.0, 1.2)
T_ambient = st.sidebar.number_input("Ambient temperature (°C)", -10.0, 30.0, 10.0)
setpoint = st.sidebar.slider("Tank setpoint (°C)", 30.0, 70.0, 50.0)
hysteresis = st.sidebar.slider("Thermostat hysteresis (°C)", 1.0, 10.0, 3.0)
U_value = st.sidebar.slider("U-value (W/m²K)", 0.1, 2.0, 0.7)

# --------------------- GEOMETRY ---------------------
# Convert volume from L to m³
tank_volume_m3 = tank_volume_l / 1000.0

# Cylinder radius (from V = π r² h)
tank_radius = np.sqrt(tank_volume_m3 / (np.pi * tank_height))

# Surface area (sides + top + bottom)
tank_area = 2 * np.pi * tank_radius * tank_height + 2 * np.pi * tank_radius**2

# UA value from geometry and insulation
UA = U_value * tank_area

# --------------------- FIXED PARAMETERS ---------------------
rho = 1000.0    ## water density
c_p = 4186.0   ## constant pressure heat capacity
tank_mass = tank_volume_l / 1000.0 * rho
P_el_max = 2000.0
COP_nominal = 3.5
T_ref_sink = 45.0
COP_slope_sink = 0.03
COP_slope_source = 0.02
T_source = 5.0
T_cold = 10.0

# Hot water draws (could make user-editable later)
draws = [(7, 0, 80), (7, 30, 20), (18, 0, 50)]

# --------------------- SIMULATION ---------------------
steps = int(sim_hours * 60 / dt_min)
time = np.arange(0, steps * dt_min, dt_min) / 60.0

T_tank = np.zeros(steps)
hp_power_el = np.zeros(steps)
hp_on = np.zeros(steps, dtype=bool)
Q_from_hp = np.zeros(steps)
Q_losses = np.zeros(steps)
COP_series = np.zeros(steps)

T_tank[0] = 45.0

draw_events = {}
for h, m, v in draws:
    idx = int((h * 60 + m) / dt_min)
    if idx < steps:
        draw_events.setdefault(idx, 0.0)
        draw_events[idx] += v

for k in range(1, steps):
    T_prev = T_tank[k-1]

    # hot water draw
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

    # thermostat
    if T_prev <= (setpoint - hysteresis):
        hp_should_on = True
    elif T_prev >= setpoint:
        hp_should_on = False
    else:
        hp_should_on = bool(hp_on[k-1])

    # COP model
    COP = COP_nominal - COP_slope_sink * (T_prev - T_ref_sink) + COP_slope_source * (T_source - 5.0)
    COP = max(1.0, COP)
    COP_series[k] = COP

    # heat pump
    if hp_should_on:
        Q_dot_hp = P_el_max * COP
        P_el = P_el_max
        hp_on[k] = True
    else:
        Q_dot_hp = 0.0
        P_el = 0.0
        hp_on[k] = False

    # losses
    Q_loss = UA * (T_prev - T_ambient)

    # update tank
    dt_s = dt_min * 60.0
    E_net = (Q_dot_hp - Q_loss) * dt_s
    dT = E_net / (tank_mass * c_p)
    T_new = T_prev + dT

    T_tank[k] = T_new
    hp_power_el[k] = P_el
    Q_from_hp[k] = Q_dot_hp
    Q_losses[k] = Q_loss

# --------------------- POSTPROCESS ---------------------
total_el_kwh = np.sum(hp_power_el) * (dt_min/60.0) / 1000.0
total_heat_kwh = np.sum(Q_from_hp) * (dt_min/60.0) / 1000.0
total_losses_kwh = np.sum(Q_losses) * (dt_min/60.0) / 1000.0
on_minutes = np.sum(hp_on)

summary = {
    "Simulation hours": sim_hours,
    "Tank volume (L)": tank_volume_l,
    "Tank height (m)": round(tank_height, 2),
    "Tank radius (m)": round(tank_radius, 3),
    "Surface area (m²)": round(tank_area, 2),
    "U-value (W/m²K)": U_value,
    "UA (W/K)": round(UA, 2),
    "Total HP electrical energy (kWh)": round(total_el_kwh, 3),
    "Total heat delivered by HP (kWh)": round(total_heat_kwh, 3),
    "Total losses (kWh)": round(total_losses_kwh, 3),
    "HP run time (minutes)": int(on_minutes),
    "Average COP": round((total_heat_kwh / total_el_kwh) if total_el_kwh > 0 else 0.0, 2)
}

df_summary = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
st.subheader("Simulation summary")
st.dataframe(df_summary)

# --------------------- PLOTS ---------------------
fig1, ax1 = plt.subplots()
ax1.plot(time, T_tank)
ax1.set_xlabel("Time (hours)")
ax1.set_ylabel("Tank temperature (°C)")
ax1.set_title("Tank temperature vs time")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.step(time, hp_power_el, where='post')
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("HP electrical power (W)")
ax2.set_title("Heat pump electrical power vs time")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
ax3.plot(time, COP_series)
ax3.set_xlabel("Time (hours)")
ax3.set_ylabel("COP")
ax3.set_title("Heat pump COP vs time")
st.pyplot(fig3)

cumulative_el = np.cumsum(hp_power_el) * (dt_min/60.0) / 1000.0
cumulative_heat = np.cumsum(Q_from_hp) * (dt_min/60.0) / 1000.0

fig4, ax4 = plt.subplots()
ax4.plot(time, cumulative_el, label="Cumulative electrical (kWh)")
ax4.plot(time, cumulative_heat, label="Cumulative heat (kWh)")
ax4.set_xlabel("Time (hours)")
ax4.set_ylabel("Energy (kWh)")
ax4.set_title("Cumulative")
ax4.legend()
st.pyplot(fig4)

fig5, ax5 = plt.subplots()
ax5.plot(time, Q_losses)
ax5.set_xlabel("Time (hours)")
ax5.set_ylabel("Heat loss (W)")
ax5.set_title("Heat losses vs time")
st.pyplot(fig5)
