import streamlit as st
import pandas as pd
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="INNOVIXUS | Predictive Cyber Defence",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0b1220;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1 {
    font-weight: 800;
}

.metric-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.04);
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 10px;
}

.small-text {
    font-size: 14px;
    opacity: 0.75;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILES
# ============================================================

PREDICTION_FILE = Path(
    r".\data\infiltration_predictions.csv"
)

MITRE_FILE = Path(
    r".\data\mitre_predictions.csv"
)

EXPLANATION_FILE = Path(
    r".\data\feature_explanation.csv"
)

BENCHMARK_FILE = Path(
    r".\data\benchmark_results.csv"
)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ INNOVIXUS")

st.markdown(
    "**Predictive Cyber Defence using Temporal World Models**"
)

st.caption(
    "SIH26153 | LSTM-based network-state forecasting and "
    "explainable attacker progression prediction"
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

if not PREDICTION_FILE.exists():

    st.error(
        "infiltration_predictions.csv not found. "
        "Run the prediction pipeline first."
    )

    st.stop()


predictions = pd.read_csv(
    PREDICTION_FILE
)

predictions["Timestamp"] = pd.to_datetime(
    predictions["Timestamp"]
)


if MITRE_FILE.exists():

    mitre = pd.read_csv(
        MITRE_FILE
    )

    mitre["Timestamp"] = pd.to_datetime(
        mitre["Timestamp"]
    )

else:

    mitre = None


if EXPLANATION_FILE.exists():

    explanation = pd.read_csv(
        EXPLANATION_FILE
    )

else:

    explanation = None


if BENCHMARK_FILE.exists():

    benchmark = pd.read_csv(
        BENCHMARK_FILE
    )

else:

    benchmark = None


# ============================================================
# STATE SELECTOR
# ============================================================

st.markdown(
    '<div class="section-title">Network State Analysis</div>',
    unsafe_allow_html=True
)

selected_index = st.slider(
    "Select a network state",
    min_value=0,
    max_value=len(predictions) - 1,
    value=0
)

selected = predictions.iloc[selected_index]

timestamp = selected["Timestamp"]

probability = float(
    selected["Infiltration_Probability"]
)

actual_state = selected["Actual_State"]

# ------------------------------------------------------------
# MITRE INFORMATION
# ------------------------------------------------------------

if mitre is not None:

    matching = mitre[
        mitre["Timestamp"] == timestamp
    ]

    if len(matching) > 0:

        mitre_row = matching.iloc[0]

        scenario = mitre_row["Attack_Scenario"]
        technique = mitre_row["MITRE_Technique"]
        tactic = mitre_row["MITRE_Tactic"]
        stage = mitre_row["MITRE_Stage"]

    else:

        scenario = "Unknown"
        technique = "Not assigned"
        tactic = "Unknown"
        stage = "Monitoring"

else:

    scenario = "Unknown"
    technique = "Not assigned"
    tactic = "Unknown"
    stage = "Monitoring"


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Infiltration Probability",
        f"{probability * 100:.2f}%"
    )


with col2:

    st.metric(
        "Observed State",
        actual_state
    )


with col3:

    st.metric(
        "Attack Scenario",
        scenario
    )


with col4:

    st.metric(
        "MITRE Stage",
        stage
    )


# ============================================================
# RISK STATUS
# ============================================================

if probability >= 0.80:

    st.error(
        "🔴 HIGH RISK — Strong predicted malicious progression"
    )

elif probability >= 0.50:

    st.warning(
        "🟠 MEDIUM RISK — Suspicious activity requires investigation"
    )

else:

    st.success(
        "🟢 LOW RISK — No strong malicious progression signal"
    )


# ============================================================
# FUTURE ATTACK PROGRESSION
# ============================================================

st.markdown(
    '<div class="section-title">📈 Infiltration Probability Timeline</div>',
    unsafe_allow_html=True
)

timeline = predictions[
    [
        "Timestamp",
        "Infiltration_Probability"
    ]
].copy()

timeline = timeline.set_index(
    "Timestamp"
)

st.line_chart(
    timeline
)


# ============================================================
# PREDICTION EXPLANATION
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Why did the model make this prediction?</div>',
    unsafe_allow_html=True
)

if explanation is not None:

    top_features = explanation.head(10).copy()

    top_features = top_features[
        [
            "Feature",
            "SHAP_Value",
            "Direction"
        ]
    ]

    st.dataframe(
        top_features,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "SHAP values represent feature contribution to the "
        "classifier prediction. They are not probability percentages."
    )

else:

    st.info(
        "Run explain_prediction.py to generate SHAP explanations."
    )


# ============================================================
# MITRE ATT&CK CONTEXT
# ============================================================

st.markdown(
    '<div class="section-title">🎯 MITRE ATT&CK Context</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)


with m1:

    st.markdown("**Attack Scenario**")
    st.write(scenario)


with m2:

    st.markdown("**Technique**")
    st.write(technique)


with m3:

    st.markdown("**Tactic**")
    st.write(tactic)


st.info(
    "MITRE context is derived from the documented attack scenario "
    "and timeline. The LSTM predicts malicious progression; it does "
    "not independently identify the exact ATT&CK technique."
)


# ============================================================
# MODEL BENCHMARK
# ============================================================

st.markdown(
    '<div class="section-title">📊 Model Benchmark</div>',
    unsafe_allow_html=True
)

if benchmark is not None:

    display_benchmark = benchmark.copy()

    display_benchmark.columns = [
        column.replace("_", " ")
        for column in display_benchmark.columns
    ]

    st.dataframe(
        display_benchmark,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Metrics are from the held-out temporal benchmark on "
        "the current CIC-IDS2018 evaluation subset."
    )

else:

    st.info(
        "Run benchmark_temporal.py to generate benchmark results."
    )


# ============================================================
# ARCHITECTURE
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ INNOVIXUS Processing Pipeline</div>',
    unsafe_allow_html=True
)

st.code(
"""
Flow CSV / PCAP
      ↓
Feature Extraction
      ↓
Time-Window Network State
      ↓
Temporal Sequence
      ↓
LSTM World Model
      ↓
Future State Forecast
      ↓
Infiltration Probability
      ↓
SHAP Explainability
      ↓
MITRE ATT&CK Context
      ↓
Decision Support Dashboard
""",
language="text"
)


# ============================================================
# RAW DATA
# ============================================================

with st.expander("🔍 View raw prediction data"):

    st.dataframe(
        predictions,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "INNOVIXUS | Predictive Cyber Defence | SIH26153"
)