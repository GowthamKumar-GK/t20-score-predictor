import streamlit as st
import numpy as np
import pandas as pd
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="T20 Score Predictor",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── BACKGROUND IMAGE FROM LOCAL FILE ─────────────────────────
def set_bg(image_file):
    with open(image_file, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{img_data}") !important;
        background-size: cover !important;
        background-position: center top !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
        min-height: 100vh;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("image.jpg")

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* DARK OVERLAY */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: linear-gradient(
        to bottom,
        rgba(0,0,0,0.40) 0%,
        rgba(5,10,30,0.62) 30%,
        rgba(5,10,30,0.85) 60%,
        rgba(5,10,30,0.97) 100%
    );
    z-index: 0;
    pointer-events: none;
}
.stApp > * { position: relative; z-index: 1; }

/* TOP GLOW LINE */
.top-bar {
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, transparent, #1E90FF, #63B3FF, #1E90FF, transparent);
    box-shadow: 0 0 12px rgba(30,144,255,0.8);
}

/* TITLE */
.title-wrap { text-align: center; padding: 2.2rem 1rem 0.5rem; }
.main-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #1E90FF, #63B3FF, #1E90FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 12px rgba(30,144,255,0.5));
    letter-spacing: 4px;
}
.sub-title {
    color: rgba(255,255,255,0.55);
    font-size: 0.82rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.blue-line {
    width: 100px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #1E90FF, #63B3FF, #1E90FF, transparent);
    box-shadow: 0 0 10px rgba(30,144,255,0.8);
    margin: 0.8rem auto 1.8rem;
}

/* GLASS INPUT CARD */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(30,144,255,0.2);
    border-radius: 22px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 50px rgba(0,0,0,0.5);
}
.sec-lbl {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 3px;
    color: #1E90FF;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* NUMBER INPUTS */
div[data-testid="stNumberInput"] label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.85rem !important;
}
div[data-testid="stNumberInput"] input {
    background: rgba(0,0,0,0.6) !important;
    border: 1px solid rgba(30,144,255,0.5) !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}
div[data-testid="stNumberInput"] input::placeholder {
    color: rgba(255,255,255,0.5) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.5) !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #1E90FF !important;
    box-shadow: 0 0 0 2px rgba(30,144,255,0.25) !important;
    outline: none !important;
}

/* SELECT BOX */
div[data-testid="stSelectbox"] label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.85rem !important;
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(30,144,255,0.3) !important;
    border-radius: 10px !important;
    color: white !important;
}
div[data-baseweb="popover"] li {
    background: #0d1b3e !important;
    color: white !important;
}

/* RESULT BOX */
.result-wrap {
    text-align: center;
    padding: 2.5rem 1.5rem;
    background: linear-gradient(135deg, rgba(30,144,255,0.12), rgba(5,10,30,0.75));
    border: 1.5px solid rgba(30,144,255,0.3);
    border-radius: 24px;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 50px rgba(30,144,255,0.1), 0 8px 40px rgba(0,0,0,0.5);
    margin-bottom: 1.2rem;
}
.res-label {
    font-size: 0.72rem;
    letter-spacing: 4px;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.res-score {
    font-family: 'Rajdhani', sans-serif;
    font-size: 7rem;
    font-weight: 700;
    color: #1E90FF;
    line-height: 1;
    filter: drop-shadow(0 0 20px rgba(30,144,255,0.6));
}
.res-range {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.35);
    margin-top: 0.4rem;
    letter-spacing: 1px;
}
.zone-pill {
    display: inline-block;
    padding: 6px 22px;
    border-radius: 30px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.8rem;
}

/* STAT GRID */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
}
.stat-tile {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(30,144,255,0.15);
    border-radius: 12px;
    padding: 0.8rem 0.5rem;
    text-align: center;
}
.stat-v {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1E90FF;
}
.stat-l {
    font-size: 0.67rem;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* FOOTER */
.footer {
    text-align: center;
    color: rgba(255,255,255,0.2);
    font-size: 0.7rem;
    letter-spacing: 2px;
    padding: 1rem 0 2rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 1rem;
}
.credit-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #5B9BD5, #89B8E0, #5B9BD5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 6px rgba(30,144,255,0.25));
    letter-spacing: 3px;
}

/* HIDE STREAMLIT CHROME */
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { 
    padding-top: 0 !important; 
    padding-bottom: 0 !important;
    max-width: 560px !important; 
}

.stAppViewContainer { padding-top: 0 !important; }
div[data-testid="stAppViewBlockContainer"] { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── DATASET ──────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=800, seed=42):
    rng = np.random.default_rng(seed)
    leagues = ["IPL","PSL","BBL","SA20","CPL","BPL",
               "T20 World Cup","T20I","T20 Blast","LPL"]
    lm = {"IPL":1.08,"PSL":0.97,"BBL":1.01,"SA20":0.99,"CPL":0.94,
          "BPL":0.92,"T20 World Cup":0.96,"T20I":0.97,"T20 Blast":0.90,"LPL":0.94}
    pf = {"slow":0.90,"flat":1.00,"bouncy":1.04}
    rows = []
    for i in range(n):
        lg  = rng.choice(leagues); l = lm[lg]
        pt  = rng.choice(["slow","flat","flat","bouncy"]); p = pf[pt]
        va  = rng.integers(138, 182)
        ppw = rng.choice([0,1,2,3,4,5,6], p=[0.07,0.22,0.27,0.22,0.13,0.06,0.03])
        base= {0:60,1:56,2:51,3:45,4:39,5:31,6:23}[ppw]
        ppr = int(np.clip(rng.normal(base*p*l, 7), 8, 120))
        pp4s= int(rng.integers(max(0,ppr//12), max(1,ppr//7)+1))
        pp6s= int(rng.integers(0, max(1,ppr//16)+1))
        ppdb= int(rng.integers(max(5,18-ppw*2), 23))
        wih = 10 - ppw
        mf  = 1.0 - (ppw * 0.030)
        df2 = 1.0 + (wih * 0.008)
        fs  = int(np.clip(ppr*3.0*mf*df2*p*l + rng.normal(0,9), ppr+15, 280))
        rows.append({
            "pp_runs":ppr, "pp_wickets":ppw,
            "pp_fours":pp4s, "pp_sixes":pp6s,
            "pp_dot_pct":round(ppdb/36*100,1),
            "pp_boundary_pct":round((pp4s+pp6s)/36*100,1),
            "wickets_in_hand":wih, "venue_avg":va,
            "pitch_type":{"slow":0,"flat":1,"bouncy":2}[pt],
            "innings":(i%2)+1, "final_score":fs
        })
    return pd.DataFrame(rows)


# ── TRAIN MODEL ──────────────────────────────────────────────
@st.cache_resource
def train_model(_df):
    F = ["pp_runs","pp_wickets","pp_fours","pp_sixes",
         "pp_dot_pct","pp_boundary_pct","wickets_in_hand",
         "venue_avg","pitch_type","innings"]
    X   = _df[F]
    y   = _df["final_score"]
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestRegressor(
        n_estimators=50, max_depth=12,
        min_samples_leaf=5, random_state=42, n_jobs=-1)
    m.fit(Xtr.values, ytr)
    return m, F


# ── PREDICT ──────────────────────────────────────────────────
def get_prediction(pp_runs, pp_wickets, pitch_num, model, F):
    wih    = 10 - pp_wickets
    ppdb   = max(15, 38 - int(pp_runs * 0.2))
    pp_dot = round(ppdb / 36 * 100, 1)
    pp4s   = max(0, int(pp_runs // 9))
    pp6s   = max(0, int(pp_runs // 18))
    pp_bnd = round((pp4s + pp6s) / 36 * 100, 1)
    inp = pd.DataFrame([{
        "pp_runs":pp_runs, "pp_wickets":pp_wickets,
        "pp_fours":pp4s,   "pp_sixes":pp6s,
        "pp_dot_pct":pp_dot, "pp_boundary_pct":pp_bnd,
        "wickets_in_hand":wih, "venue_avg":158,
        "pitch_type":pitch_num, "innings":1
    }])[F]
    return int(round(model.predict(inp.values)[0]))


# ── LOAD DATA & MODEL ─────────────────────────────────────────
df           = generate_dataset()
model, FEATS = train_model(df)


# ── PAGE RENDER ───────────────────────────────────────────────
st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="title-wrap">
    <div class="main-title">T20 PREDICTOR</div>
    <div class="sub-title">Powerplay Intelligence</div>
    <div class="blue-line"></div>
</div>
""", unsafe_allow_html=True)


# ── INPUT CARD ────────────────────────────────────────────────
st.markdown("""

<div class="sec-lbl" style="font-weight: bold; font-size: 23px;">
    ⚡ Enter Powerplay Stats — Overs 1 to 6
</div>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    pp_runs    = st.number_input("PP Runs",    min_value=0,  max_value=120, value=0, step=1, key="ppr")
with c2:
    pp_wickets = st.number_input("PP Wickets", min_value=0,  max_value=7,   value=0,  step=1, key="ppw")

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="sec-lbl">🏟️ Pitch Type</div>', unsafe_allow_html=True)

pitch_sel = st.selectbox(
    "Select pitch",
    ["Slow / Spin 🐌", "Flat 🏟️", "Bouncy / Pace 💨"],
    index=1,
    label_visibility="collapsed",
    key="pt"
)
pitch_num = {"Slow / Spin 🐌":0, "Flat 🏟️":1, "Bouncy / Pace 💨":2}[pitch_sel]




# ── AUTO PREDICT ──────────────────────────────────────────────
pred    = get_prediction(pp_runs, pp_wickets, pitch_num, model, FEATS)
lo      = max(pred - 18, 60)
hi      = pred + 18
rr      = round(pp_runs / 6, 2)
wih     = 10 - pp_wickets
contrib = round(pp_runs / pred * 100)
post    = pred - pp_runs
pred_rr = round(pred / 20, 2)

if   pred < 130: zone, zc = "LOW SCORE",    "#C62828"
elif pred < 150: zone, zc = "BELOW PAR",    "#E65100"
elif pred < 165: zone, zc = "PAR SCORE",    "#1565C0"
elif pred < 180: zone, zc = "GOOD SCORE",   "#2E7D32"
else:            zone, zc = "MATCH WINNER", "#1B5E20"


# ── RESULT ────────────────────────────────────────────────────
st.markdown(f"""
<div class="result-wrap">
    <div class="res-label">Predicted Final Score</div>
    <div class="res-score">{pred}</div>
    <div class="res-range">{lo} – {hi} runs &nbsp;·&nbsp; {pred_rr} rpo</div>
    <span class="zone-pill" style="background:{zc};color:white">{zone}</span>
</div>

<div class="stat-grid">
    <div class="stat-tile">
        <div class="stat-v">{rr}</div>
        <div class="stat-l">PP Run Rate</div>
    </div>
    <div class="stat-tile">
        <div class="stat-v">{wih}</div>
        <div class="stat-l">Wkts in Hand</div>
    </div>
    <div class="stat-tile">
        <div class="stat-v">{contrib}%</div>
        <div class="stat-l">PP Contrib</div>
    </div>
    <div class="stat-tile">
        <div class="stat-v">{post}</div>
        <div class="stat-l">Post PP Runs</div>
    </div>
    <div class="stat-tile">
        <div class="stat-v">{pred_rr}</div>
        <div class="stat-l">Final RR</div>
    </div>
    <div class="stat-tile">
        <div class="stat-v">{pred}</div>
        <div class="stat-l">Total Runs</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🏏 T20 SCORE PREDICTOR &nbsp;·&nbsp; RANDOM FOREST &nbsp;·&nbsp; STREAMLIT
    <br><br>
    <span style="color:rgba(255,255,255,0.35);font-size:0.7rem;letter-spacing:2px">
        DEVELOPED BY
    </span>
    <br>
    <span class="credit-name">GOWTHAM KUMAR G</span>
    <br>
    <span style="color:rgba(255,255,255,0.2);font-size:0.68rem;letter-spacing:1px">
        
    </span>
    <br><br>
    <span style="color:rgba(255,255,255,0.12);font-size:0.65rem">
        VIRAT KOHLI · MS DHONI · ROHIT SHARMA · MANISH PANDEY
    </span>
</div>
""", unsafe_allow_html=True)
