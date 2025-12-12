import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NASA Shuttle AI Diagnostic",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #f5f7fa; }
    [data-testid="stSidebar"] { background-color: #ffffff; }
    .main-container { background-color: #ffffff; padding: 2rem; border-radius: 15px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); }
    .success-box { padding: 25px; background-color: #d4edda; color: #155724; border-radius: 12px; border-left: 6px solid #28a745; box-shadow: 0 2px 8px rgba(40, 167, 69, 0.15); animation: fadeIn 0.5s; }
    .success-box h2 { color: #155724; margin-top: 0; }
    .error-box { padding: 25px; background-color: #f8d7da; color: #721c24; border-radius: 12px; border-left: 6px solid #dc3545; box-shadow: 0 2px 8px rgba(220, 53, 69, 0.15); animation: pulse 1s infinite; }
    .error-box h2 { color: #721c24; margin-top: 0; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.9; } }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if 'history' not in st.session_state:
    st.session_state.history = []

# ==========================================
# 3. MODEL LOADING FUNCTION
# ==========================================
@st.cache_resource
def load_pipeline(filename):
    if os.path.exists(filename):
        return joblib.load(filename)
    return None

# ==========================================
# 4. SIDEBAR: ENHANCED CONFIGURATION
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=180)
    st.title("🎛️ System Configuration")
    
    # Model Selection
    model_option = st.selectbox(
        "Active Model",
        ["Local Outlier Factor (LOF)", "Isolation Forest (iForest)", "One-Class SVM (OCSVM)"],
        help="Select the anomaly detection algorithm"
    )
    
    # Logic to load the correct file
    if model_option == "Local Outlier Factor (LOF)":
        model = load_pipeline("shuttle_lof_pipeline.pkl")
        status = "🟢 Active: Local Outlier Factor"
        status_color = "green"
    elif model_option == "Isolation Forest (iForest)":
        model = load_pipeline("best_anomaly_detection_pipeline.pkl")
        status = "🟡 Active: Isolation Forest"
        status_color = "orange"
    elif model_option == "One-Class SVM (OCSVM)":
        model = load_pipeline("shuttle_ocsvm_99f1.pkl")
        status = "🔵 Active: One-Class SVM"
        status_color = "blue"
    else:
        model = None
        status = "🔴 Inactive"
        status_color = "red"
    
    st.markdown("---")
    
    # System Statistics
    st.subheader("📊 Session Statistics")
    total_diagnostics = len(st.session_state.history)
    total_anomalies = sum(1 for h in st.session_state.history if h['prediction'] == -1)
    total_normal = total_diagnostics - total_anomalies
    
    st.metric("Total Diagnostics", total_diagnostics)
    st.metric("Normal Results", total_normal)
    st.metric("Anomalies Detected", total_anomalies, delta_color="inverse")
    
    st.markdown("---")
    
    # Quick Presets
    st.subheader("⚡ Quick Presets")
    if st.button("🟢 Normal Values", use_container_width=True):
        st.session_state.preset = 'normal'
        st.rerun()
    if st.button("🔴 Anomaly Test", use_container_width=True):
        st.session_state.preset = 'anomaly'
        st.rerun()
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.history = []
        st.session_state.preset = 'reset'
        st.rerun()

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
st.title("🚀 NASA Shuttle Radiator Anomaly Detection")
st.markdown(f"**System Status:** <span style='color:{status_color}; font-weight:bold'>{status}</span>", unsafe_allow_html=True)

# Check if model loaded
if model is None:
    st.error(f"❌ Error: Could not find the model file for {model_option}.")
    st.info(f"💡 **Tip:** Ensure the .pkl file is in the application folder.")
    st.stop()

# ==========================================
# 6. ENHANCED SENSOR INPUT WITH PRESETS (FIXED)
# ==========================================
st.markdown("---")
st.header("📡 Telemetry Input Panel")
st.caption("Enter real-time sensor readings from the radiator subsystem (Sensor A1/Time excluded)")

# Preset values
preset_values = {
    'normal': [0, 76, 0, 28, 18, 40, 48, 8],
    'anomaly': [0, 92, 0, 0, 26, 36, 92, 56],
    'reset': [0, 0, 0, 0, 0, 0, 0, 0]
}

# Initialize session_state for sensors
for i, sensor in enumerate(['s2','s3','s4','s5','s6','s7','s8','s9']):
    if sensor not in st.session_state:
        st.session_state[sensor] = 0

# Apply presets if clicked
if 'preset' in st.session_state:
    preset = preset_values.get(st.session_state.preset, preset_values['normal'])
    st.session_state.s2, st.session_state.s3, st.session_state.s4, st.session_state.s5, \
    st.session_state.s6, st.session_state.s7, st.session_state.s8, st.session_state.s9 = preset
    del st.session_state.preset  # remove after applying

# Sensor Input Grid
col1, col2, col3, col4 = st.columns(4)

with col1:
    s2 = st.number_input("🔵 Sensor A2", value=st.session_state.s2, min_value=-10000, max_value=10000, help="Temperature sensor reading")
    s6 = st.number_input("🔵 Sensor A6", value=st.session_state.s6, min_value=-40000, max_value=40000, help="Pressure sensor reading")

with col2:
    s3 = st.number_input("🟢 Sensor A3", value=st.session_state.s3, min_value=-500, max_value=500, help="Flow rate sensor")
    s7 = st.number_input("🟢 Sensor A7", value=st.session_state.s7, min_value=-500, max_value=500, help="Valve position sensor")

with col3:
    s4 = st.number_input("🟡 Sensor A4", value=st.session_state.s4, min_value=-5000, max_value=5000, help="Thermal gradient sensor")
    s8 = st.number_input("🟡 Sensor A8", value=st.session_state.s8, min_value=-500, max_value=500, help="Coolant level sensor")

with col4:
    s5 = st.number_input("🟠 Sensor A5", value=st.session_state.s5, min_value=-500, max_value=500, help="Radiation sensor")
    s9 = st.number_input("🟠 Sensor A9", value=st.session_state.s9, min_value=-500, max_value=500, help="Backup temperature sensor")

# Sensor visualization
df_sensors = pd.DataFrame({
    'Sensor': ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9'],
    'Value': [s2, s3, s4, s5, s6, s7, s8, s9]
})

fig_sensors = px.bar(df_sensors, x='Sensor', y='Value', title='Current Sensor Readings', color='Value', color_continuous_scale='RdYlGn_r')
fig_sensors.update_layout(height=300, showlegend=False)
st.plotly_chart(fig_sensors, use_container_width=True)

# ==========================================
# 7. PREDICTION LOGIC
# ==========================================
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
with col_btn1:
    run_diagnostics = st.button("🔍 Run Diagnostics", type="primary", use_container_width=True)

if run_diagnostics:
    
    # CRITICAL FIX FOR OCSVM PIPELINE:
    # The ColumnTransformer in the pipeline expects specific column names.
    # We must convert the input list to a DataFrame with headers.
    feature_names = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
    input_data = pd.DataFrame([[s2, s3, s4, s5, s6, s7, s8, s9]], columns=feature_names)
    
    with st.spinner("🔄 Analyzing Sensor Patterns..."):
        # Predict: Scikit-learn Anomaly Models return 1 for Inlier (Normal), -1 for Outlier (Anomaly)
        prediction = model.predict(input_data)[0]
        
        # Decision Function Score
        try:
            score = model.decision_function(input_data)[0]
        except:
            score = 0.0
        
        # Update session state
        st.session_state.history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'score': score,
            'sensors': [s2, s3, s4, s5, s6, s7, s8, s9]
        })
        
        st.rerun()

# ==========================================
# 8. RESULTS DISPLAY
# ==========================================
if len(st.session_state.history) > 0:
    st.markdown("### 📋 Diagnostic Results")
    latest = st.session_state.history[-1]
    prediction = latest['prediction']
    score = latest['score']
    
    # 1 represents NORMAL in Scikit-Learn OCSVM/LOF/IsoForest
    if prediction == 1:
        st.markdown(f"""
        <div class="success-box">
            <h2>✅ SYSTEM NOMINAL</h2>
            <p>All radiator subsystems are operating within acceptable parameters.</p>
            <hr style="border: 1px solid #28a745; opacity: 0.3;">
            <p><b>🎯 Stability Score:</b> <b>{score:.4f}</b> (Positive = Normal)</p>
            <p><b>⏰ Timestamp:</b> {latest['timestamp'].strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
        <div class="error-box">
            <h2>🚨 ANOMALY DETECTED</h2>
            <p>Sensor readings deviate significantly from normal operating parameters!</p>
            <hr style="border: 1px solid #dc3545; opacity: 0.3;">
            <p><b>⚠️ Deviation Score:</b> <b>{score:.4f}</b> (Negative = Anomaly)</p>
            <p><b>⏰ Timestamp:</b> {latest['timestamp'].strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 9. HISTORY TRACKING
# ==========================================
if len(st.session_state.history) > 0:
    st.markdown("---")
    st.header("📈 Diagnostic History")
    
    history_df = pd.DataFrame([
        {
            'Time': h['timestamp'].strftime('%H:%M:%S'),
            'Status': 'Normal' if h['prediction'] == 1 else 'Anomaly',
            'Score': f"{h['score']:.4f}"
        }
        for h in st.session_state.history[-10:]
    ])
    
    st.dataframe(history_df, use_container_width=True)
    
    # Trend Chart
    if len(st.session_state.history) > 1:
        scores = [h['score'] for h in st.session_state.history]
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(y=scores, mode='lines+markers', name='Anomaly Score', line=dict(color='royalblue')))
        fig_trend.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Boundary")
        fig_trend.update_layout(title="Anomaly Score Trend", height=350)
        st.plotly_chart(fig_trend, use_container_width=True)