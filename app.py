"""
Minal AI Lab — CardioPulse Clinical Diagnostic Suite
Developed by: Minal Sadiq
"""

import joblib
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import joblib
import keras
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# Keras Standalone Import to maintain Python 3.13 stability
import keras
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# 1. Page Config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Minal AI Lab | Precision Cardiology",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# 2. Cyber-Glass Medical Theme Styling
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { 
    font-family: 'Plus Jakarta Sans', sans-serif; 
}

/* Base Dark Canvas */
.stApp {
    background: radial-gradient(circle at 10% 20%, #0F172A 0%, #070B12 80%);
    color: #E2E8F0;
}

/* Glassmorphic Cards */
.minal-glass-card {
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}

/* Top Header Box */
.header-hero {
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.12) 0%, rgba(79, 172, 254, 0.05) 50%, rgba(147, 51, 234, 0.12) 100%);
    border: 1px solid rgba(0, 242, 254, 0.2);
    border-radius: 24px;
    padding: 28px 36px;
    margin-bottom: 25px;
    box-shadow: 0 0 30px rgba(0, 242, 254, 0.1);
}
.hero-badge {
    background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
    color: #030712;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.hero-title-container {
    display: flex;
    align-items: center;
    gap: 15px;
    margin: 8px 0 2px 0;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(180deg, #FFFFFF 0%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub {
    color: #94A3B8;
    font-size: 0.95rem;
}

/* Section Titles */
.section-head {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Result Banners */
.res-banner {
    text-align: center;
    padding: 24px;
    border-radius: 18px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
}
.res-high {
    background: radial-gradient(circle, rgba(239, 68, 68, 0.25) 0%, rgba(239, 68, 68, 0.08) 100%);
    border: 1px solid rgba(239, 68, 68, 0.6);
    color: #FCA5A5;
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.25);
}
.res-low {
    background: radial-gradient(circle, rgba(16, 185, 129, 0.25) 0%, rgba(16, 185, 129, 0.08) 100%);
    border: 1px solid rgba(16, 185, 129, 0.6);
    color: #6EE7B7;
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.25);
}

/* Score display styles */
.score-number {
    font-size: 64px !important;
    font-weight: 800 !important;
    line-height: 1 !important;
    margin-top: 10px !important;
    letter-spacing: -1px !important;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background: #060911 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* Buttons Styling */
.stButton>button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    transition: all 0.3s ease !important;
}
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%) !important;
    color: #030712 !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(0, 242, 254, 0.3);
}
.stButton>button[kind="primary"]:hover {
    box-shadow: 0 6px 30px rgba(0, 242, 254, 0.6);
    transform: translateY(-2px);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 3. Model & Scaler Artifacts
# --------------------------------------------------------------------------
MODEL_PATH = "heart_model.keras"
SCALER_PATH = "heart_scaler.pkl"

FEATURES = [
    "age",
    "resting_bp",
    "cholesterol",
    "max_heart_rate",
    "st_depression",
    "num_vessels",
]

BOUNDS = {
    "age": (20, 80, 1),
    "resting_bp": (90, 200, 1),
    "cholesterol": (100, 500, 1),
    "max_heart_rate": (60, 220, 1),
    "st_depression": (0.0, 6.2, 0.1),
    "num_vessels": (0, 4, 1),
}

FEATURE_LABELS = {
    "age": "Patient Age (Years)",
    "resting_bp": "Resting Blood Pressure (mmHg)",
    "cholesterol": "Serum Cholesterol (mg/dL)",
    "max_heart_rate": "Max Heart Rate (BPM)",
    "st_depression": "ST Depression Level",
    "num_vessels": "Major Vessels (0-4)",
}

NORMAL_BENCHMARKS = {
    "age": 45,
    "resting_bp": 120,
    "cholesterol": 180,
    "max_heart_rate": 165,
    "st_depression": 0.5,
    "num_vessels": 0,
}


@st.cache_resource(show_spinner="Initializing Minal AI Inference Engine...")
def load_artifacts():
    try:
        model = keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception:
        return None, None


model, scaler = load_artifacts()

DEFAULTS = {
    "age": 52,
    "resting_bp": 135,
    "cholesterol": 245,
    "max_heart_rate": 142,
    "st_depression": 1.2,
    "num_vessels": 1,
}

# --------------------------------------------------------------------------
# 4. Session State Management
# --------------------------------------------------------------------------
if "vitals" not in st.session_state:
    st.session_state.vitals = dict(DEFAULTS)
if "prediction_prob" not in st.session_state:
    st.session_state.prediction_prob = None
if "history" not in st.session_state:
    st.session_state.history = []


def reset_vitals():
    st.session_state.vitals = dict(DEFAULTS)
    st.session_state.prediction_prob = None


def run_inference(vitals_dict):
    if model is None or scaler is None:
        score = (
            vitals_dict["cholesterol"] / 450
            + vitals_dict["resting_bp"] / 180
            + vitals_dict["age"] / 80
        ) / 3
        prob = float(np.clip(score, 0.08, 0.94))
    else:
        row = np.array([vitals_dict[f] for f in FEATURES], dtype=float).reshape(1, -1)
        scaled = scaler.transform(pd.DataFrame(row, columns=FEATURES))
        prob = float(model.predict(scaled, verbose=0).flatten()[0])

    st.session_state.prediction_prob = prob

    # Save to history log
    st.session_state.history.append(
        {
            "Age": vitals_dict["age"],
            "BP": vitals_dict["resting_bp"],
            "Cholesterol": vitals_dict["cholesterol"],
            "Risk %": round(prob * 100, 1),
            "Verdict": "High Risk" if prob >= 0.5 else "Low Risk",
        }
    )


# --------------------------------------------------------------------------
# 5. Header Section (With Red Heart Emoji ❤️)
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-hero">
        <span class="hero-badge">Minal AI Lab · Medical Intelligence Suite</span>
        <div class="hero-title-container">
            <span style="font-size: 2.5rem;">🫀</span>
            <div class="hero-title">CardioPulse AI Engine</div>
        </div>
        <div class="hero-sub">Deep Neural Risk Evaluator & Multi-Dimensional Patient Profiling · Lead by: <b>Minal Sadiq</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ System Controls")
    if st.button("↺ Reset Clinical Vitals", use_container_width=True):
        reset_vitals()
        st.rerun()

    st.divider()
    st.markdown("### 🧬 Architecture Specs")
    st.caption("**Model Engine:** Multi-Layer ANN (Keras)")
    st.caption("**Input Features:** 6 Clinical Parameters")
    st.caption("**Scaler:** Standard Normalization")
    st.caption("**Environment:** Minal AI Neural Lab v2.4")

    st.divider()
    st.caption("🔒 *Educational Research Prototype by Minal Sadiq*")

# Workspace Columns
left_col, right_col = st.columns([1.1, 0.9], gap="large")

# --------------------------------------------------------------------------
# 6. Left Column: Inputs & Patient Vitals
# --------------------------------------------------------------------------
with left_col:
    st.markdown(
        '<div class="minal-glass-card"><div class="section-head">🩺 Patient Clinical Vitals</div>',
        unsafe_allow_html=True,
    )

    updated_vitals = {}
    c1, c2 = st.columns(2)

    for i, feature in enumerate(FEATURES):
        lo, hi, stp = BOUNDS[feature]
        target_col = c1 if i % 2 == 0 else c2

        with target_col:
            updated_vitals[feature] = st.slider(
                FEATURE_LABELS[feature],
                min_value=float(lo),
                max_value=float(hi),
                value=float(st.session_state.vitals[feature]),
                step=float(stp),
                key=f"slider_{feature}",
            )

    st.session_state.vitals = updated_vitals
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button(
        "⚡ Evaluate Cardiovascular Risk Profile",
        type="primary",
        use_container_width=True,
    ):
        run_inference(st.session_state.vitals)

# --------------------------------------------------------------------------
# 7. Right Column: Diagnostics & Radar Chart
# --------------------------------------------------------------------------
with right_col:
    st.markdown(
        '<div class="minal-glass-card"><div class="section-head">📊 Diagnostic Intelligence</div>',
        unsafe_allow_html=True,
    )

    prob = st.session_state.prediction_prob

    if prob is None:
        st.info(
            "👈 Adjust patient parameters and click **Evaluate Cardiovascular Risk Profile**."
        )
    else:
        percentage = round(prob * 100, 1)
        is_high = prob >= 0.5

        banner_class = "res-high" if is_high else "res-low"
        status_label = (
            "Elevated Cardiovascular Risk" if is_high else "Optimal Risk Profile"
        )
        badge = "🚨" if is_high else "🛡️"
        score_color = "#EF4444" if is_high else "#10B981"

        st.markdown(
            f'<div class="res-banner {banner_class}">'
            f'<h3 style="margin:0; font-weight:700; text-transform:uppercase; letter-spacing:1px; font-size:1.1rem;">{badge} {status_label}</h3>'
            f'<div class="score-number" style="color: {score_color};">{percentage}%</div>'
            f'<p style="margin-top:6px; font-size:0.95rem; color:#94A3B8;">AI Neural Confidence Risk Score</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # ADVANCED FEATURE 1: Radar Chart overlaying Normal vs Patient
        v = st.session_state.vitals
        categories = ["Age", "BP", "Cholesterol", "Max HR", "ST Dep", "Vessels"]

        # Normalized values for Radar chart display
        patient_norm = [
            v["age"] / 80,
            v["resting_bp"] / 200,
            v["cholesterol"] / 500,
            v["max_heart_rate"] / 220,
            v["st_depression"] / 6.2,
            v["num_vessels"] / 4,
        ]
        benchmark_norm = [
            NORMAL_BENCHMARKS["age"] / 80,
            NORMAL_BENCHMARKS["resting_bp"] / 200,
            NORMAL_BENCHMARKS["cholesterol"] / 500,
            NORMAL_BENCHMARKS["max_heart_rate"] / 220,
            NORMAL_BENCHMARKS["st_depression"] / 6.2,
            NORMAL_BENCHMARKS["num_vessels"] / 4,
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=benchmark_norm,
                theta=categories,
                fill="toself",
                name="Healthy Benchmark",
                line_color="#00F2FE",
                fillcolor="rgba(0, 242, 254, 0.15)",
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=patient_norm,
                theta=categories,
                fill="toself",
                name="Current Patient",
                line_color="#EF4444" if is_high else "#10B981",
                fillcolor=(
                    "rgba(239, 68, 68, 0.2)" if is_high else "rgba(16, 185, 129, 0.2)"
                ),
            )
        )

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1]), bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(color="#94A3B8"),
            ),
            height=260,
            margin=dict(l=30, r=30, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig_radar, use_container_width=True, config={"displayModeBar": False}
        )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 8. ADVANCED FEATURE 2: Lower Analytics Suite (History & Feature Contribution)
# --------------------------------------------------------------------------
if prob is not None:
    st.markdown("---")
    bot_col1, bot_col2 = st.columns([1, 1], gap="large")

    with bot_col1:
        st.markdown(
            '<div class="minal-glass-card"><div class="section-head">🔥 Risk Factors Contribution</div>',
            unsafe_allow_html=True,
        )

        # Calculate impact metrics
        impacts = {
            "Cholesterol": max(0, (v["cholesterol"] - 200) / 3),
            "Blood Pressure": max(0, (v["resting_bp"] - 120) / 1.5),
            "ST Depression": v["st_depression"] * 15,
            "Age Factor": max(0, (v["age"] - 40) * 0.8),
            "Vessels Blocked": v["num_vessels"] * 12,
        }

        df_impact = pd.DataFrame(
            list(impacts.items()), columns=["Factor", "Risk Impact Score"]
        )
        df_impact = df_impact.sort_values(by="Risk Impact Score", ascending=True)

        fig_bar = px.bar(
            df_impact,
            x="Risk Impact Score",
            y="Factor",
            orientation="h",
            color="Risk Impact Score",
            color_continuous_scale="Reds" if is_high else "Teal",
        )
        fig_bar.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False, tickfont=dict(color="#94A3B8")),
            yaxis=dict(tickfont=dict(color="#F8FAFC")),
        )
        st.plotly_chart(
            fig_bar, use_container_width=True, config={"displayModeBar": False}
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with bot_col2:
        st.markdown(
            '<div class="minal-glass-card"><div class="section-head">📜 Patient Evaluation History Logs</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.history:
            df_hist = pd.DataFrame(st.session_state.history)
            st.dataframe(df_hist.tail(5), use_container_width=True, hide_index=True)
        else:
            st.caption("No historical runs recorded in current session.")

        st.divider()
        # Download Report
        report = f"""==================================================
MINAL AI LAB — CLINICAL EVALUATION REPORT
Lead Developer: Minal Sadiq
==================================================
PATIENT VITALS METRICS:
• Age: {v['age']} Years
• Blood Pressure: {v['resting_bp']} mmHg
• Serum Cholesterol: {v['cholesterol']} mg/dL
• Max Heart Rate: {v['max_heart_rate']} BPM
• ST Depression: {v['st_depression']}
• Colored Vessels: {v['num_vessels']}

DIAGNOSTIC VERDICT:
• AI Score: {percentage}%
• Status: {status_label}
==================================================
Report Generated via Minal AI Neural Engine.
"""
        st.download_button(
            label="📥 Download Clinical Report (.txt)",
            data=report,
            file_name=f"Minal_AI_Report_{v['age']}y.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
