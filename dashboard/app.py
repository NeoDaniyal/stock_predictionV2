from pathlib import Path
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# Configuration
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="ALPHA-V2 // Quant Intelligence Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# FEATURE DATASET LOADER
# -------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Locate and load the latest calculated features per ticker from local CSV storage."""
    project_root = Path(__file__).resolve().parent.parent
    candidate_paths = [
        project_root / "features" / "final_dataset.csv",
        project_root / "reports" / "processed_data.csv",
        project_root / "data" / "final_dataset.csv",
        project_root / "final_dataset.csv"
    ]
    data_path = next((p for p in candidate_paths if p.exists()), None)
    
    if data_path:
        df = pd.read_csv(data_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
        # Get the most recent feature record for each ticker
        latest_df = df.groupby("Ticker").last().reset_index()
        return latest_df, data_path.name
    return None, None

latest_features_df, source_file_name = load_dataset()

# -------------------------------------------------------------------
# CUSTOM CSS INJECTION: DARK FINTECH GLASSMORPHISM THEME
# -------------------------------------------------------------------
st.markdown("""
<style>
    /* Global background and font styling */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit default headers/footers */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Glassmorphic Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #00f2fe !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8 !important;
    }
    
    /* Custom CSS Containers for Signal Cards */
    .signal-card {
        background: rgba(18, 24, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    
    /* Signal Badge Styling */
    .badge-buy {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }
    .badge-sell {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }
    .badge-hold {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111622;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Streamlit Primary Button Overrides */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# SIDEBAR: MODEL INTELLIGENCE & AUDIT LOG
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ ALPHA-V2 QUANT ENGINE")
    st.caption("Production Signal Generation System")
    st.markdown("---")
    
    st.markdown("#### ⚙️ Engine Specs")
    st.markdown("- **Model:** CatBoost Classifier")
    st.markdown("- **Calibration:** Isotonic Regression")
    st.markdown("- **Validation:** Walk-Forward (2021–2026)")
    st.markdown("- **Dataset:** 10 Tickers (30k+ Samples)")

    st.markdown("---")
    st.markdown("#### 📊 Validated OOF Performance")
    
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("Accuracy", "37.88%")
    col_sb2.metric("Macro F1", "37.16%")
    
    col_sb3, col_sb4 = st.columns(2)
    col_sb3.metric("Bal. Acc", "37.38%")
    col_sb4.metric("Log Loss", "1.0929")

    st.markdown("---")
    st.markdown("#### 🛡️ Data Source Status")
    if latest_features_df is not None:
        st.success(f"Loaded feature matrix: `{source_file_name}`")
    else:
        st.warning("Dataset file not found. System falling back to neutral feature payload.")

# -------------------------------------------------------------------
# MAIN TERMINAL HEADER
# -------------------------------------------------------------------
st.title("⚡ QUANTITATIVE SIGNAL TERMINAL")
st.markdown("<p style='color: #64748b; margin-top: -15px;'>Real-Time Signal Generation & Probability Calibration Monitor</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------------------------
# CONTROL PANEL & INFERENCE TRIGGER
# -------------------------------------------------------------------
col_select, col_btn = st.columns([3, 1])

with col_select:
    ticker = st.selectbox(
        "SELECT ASSET INSTRUMENT",
        ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD", "JPM", "NFLX"],
        index=0
    )

with col_btn:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    generate_btn = st.button("RUN INFERENCE ⚡", use_container_width=True)

# -------------------------------------------------------------------
# SIGNAL DISPLAY & VISUALIZATIONS
# -------------------------------------------------------------------
if generate_btn:
    # Build payload using actual feature matrix row if available
    features_dict = {}
    if latest_features_df is not None and ticker in latest_features_df["Ticker"].values:
        row = latest_features_df[latest_features_df["Ticker"] == ticker].iloc[0]
        # Omit non-feature metadata columns
        ignore_cols = ["Date", "Ticker", "Target"]
        feature_cols = [c for c in row.index if c not in ignore_cols]
        features_dict = row[feature_cols].to_dict()

    payload = {"ticker": ticker, "features": features_dict}
    
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            sig = data["signal"]
            confidence = data["confidence"]
            probs = data["probabilities"]

            # Set Signal Badge CSS
            if sig == "BUY":
                badge_html = f'<span class="badge-buy">▲ BUY SIGNAL</span>'
            elif sig == "SELL":
                badge_html = f'<span class="badge-sell">▼ SELL SIGNAL</span>'
            else:
                badge_html = f'<span class="badge-hold">⯎ HOLD ZONE</span>'

            # Header Card
            st.markdown(f"""
            <div class="signal-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin:0; font-size: 2.2rem; color: #ffffff;">{ticker}</h2>
                        <p style="margin:0; color: #64748b;">Asset Execution Decision</p>
                    </div>
                    <div>
                        {badge_html}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics Grid
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Action", sig)
            m2.metric("Signal Confidence", f"{confidence:.2%}")
            m3.metric("Buy Probability", f"{probs['BUY']:.2%}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Plotly Distribution Chart
            fig = go.Figure()

            colors = ["#ef4444", "#f59e0b", "#10b981"]
            
            fig.add_trace(go.Bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                text=[f"{v:.2%}" for v in probs.values()],
                textposition='outside',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255, 255, 255, 0.1)', width=1)
                ),
                width=0.4
            ))

            fig.update_layout(
                title=dict(
                    text=f"Calibrated Class Probability Vector ({ticker})",
                    font=dict(color="#e2e8f0", size=16)
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(18, 24, 38, 0.5)',
                yaxis=dict(
                    title="Calibrated Probability",
                    range=[0, 1.15],
                    gridcolor='rgba(255, 255, 255, 0.05)',
                    zerolinecolor='rgba(255, 255, 255, 0.05)',
                    tickfont=dict(color="#94a3b8")
                ),
                xaxis=dict(
                    title="Signal Class",
                    tickfont=dict(color="#e2e8f0", size=14)
                ),
                font=dict(color="#e2e8f0"),
                height=380,
                margin=dict(l=20, r=20, t=50, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"API Error [{response.status_code}]: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection Refused: Ensure `uvicorn api.main:app` is running on `http://127.0.0.1:8000`.")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #475569; font-size: 0.8rem;'>"
    "⚡ ALPHA-V2 QUANT ENGINE // FOR DEMONSTRATION & PORTFOLIO PURPOSES ONLY // NOT FINANCIAL ADVICE"
    "</p>",
    unsafe_allow_html=True
)