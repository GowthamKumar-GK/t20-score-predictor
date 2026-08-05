import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="T20 Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #1F4E79, #2E86C1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #666; margin-bottom: 1.5rem;
    }
    .pred-box {
        background: linear-gradient(135deg, #1F4E79, #2980B9);
        border-radius: 15px; padding: 2rem; text-align: center;
        color: white; margin: 1rem 0;
    }
    .pred-score { font-size: 4rem; font-weight: 700; line-height: 1; }
    .pred-label { font-size: 1rem; opacity: 0.85; margin-bottom: 0.5rem; }
    .pred-range { font-size: 0.9rem; opacity: 0.75; margin-top: 0.5rem; }
    .metric-card {
        background: #F0F4F8; border-radius: 10px;
        padding: 1rem; text-align: center; border-left: 4px solid #2E86C1;
    }
    .zone-high   { background:#1E8449; color:white; padding:6px 16px; border-radius:20px; font-weight:600; }
    .zone-good   { background:#27AE60; color:white; padding:6px 16px; border-radius:20px; font-weight:600; }
    .zone-par    { background:#2980B9; color:white; padding:6px 16px; border-radius:20px; font-weight:600; }
    .zone-below  { background:#D4AC0D; color:white; padding:6px 16px; border-radius:20px; font-weight:600; }
    .zone-low    { background:#C0392B; color:white; padding:6px 16px; border-radius:20px; font-weight:600; }
    .stSlider > div > div { background: #2E86C1 !important; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── DATASET ──────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    leagues = ["IPL","PSL","BBL","SA20","CPL","BPL","ILT20","MLC",
               "T20 World Cup","T20I","T20 Blast","LPL","Super Smash"]
    league_mult = {"IPL":1.08,"PSL":0.97,"BBL":1.01,"SA20":0.99,
                   "CPL":0.94,"BPL":0.92,"ILT20":1.00,"MLC":0.95,
                   "T20 World Cup":0.96,"T20I":0.97,"T20 Blast":0.90,
                   "LPL":0.94,"Super Smash":0.91}
    pitch_map    = {"slow":0,"flat":1,"bouncy":2}
    pitch_factor = {"slow":0.90,"flat":1.00,"bouncy":1.04}
    pitches = rng.choice(["slow","flat","flat","bouncy"], n)
    rows = []
    for i in range(n):
        lg  = rng.choice(leagues); lm = league_mult[lg]
        pt  = pitches[i];          pf = pitch_factor[pt]
        va  = rng.integers(138,182)
        ppw = rng.choice([0,1,2,3,4,5,6], p=[0.07,0.22,0.27,0.22,0.13,0.06,0.03])
        base= {0:60,1:56,2:51,3:45,4:39,5:31,6:23}[ppw]
        ppr = int(np.clip(rng.normal(base*pf*lm,7),8,90))
        pp4s= int(rng.integers(max(0,ppr//12), max(1,ppr//7)+1))
        pp6s= int(rng.integers(0, max(1,ppr//16)+1))
        ppdb= int(rng.integers(max(5,18-ppw*2),23))
        bnd_pct = round((pp4s+pp6s)/36*100,1)
        dot_pct = round(ppdb/36*100,1)
        wih = 10-ppw
        mf  = 1.0-(ppw*0.030); df2=1.0+(wih*0.008)
        fs  = int(np.clip(ppr*3.0*mf*df2*pf*lm+rng.normal(0,9),ppr+15,265))
        rows.append({"league":lg,"pitch_type":pitch_map[pt],"venue_avg":va,
                     "innings":(i%2)+1,"pp_runs":ppr,"pp_wickets":ppw,
                     "pp_run_rate":round(ppr/6,2),"pp_fours":pp4s,"pp_sixes":pp6s,
                     "pp_dot_balls":ppdb,"pp_boundary_pct":bnd_pct,
                     "pp_dot_pct":dot_pct,"wickets_in_hand":wih,
                     "final_score":fs,"final_wickets":min(10,ppw+int(rng.integers(2,7)))})
    return pd.DataFrame(rows)


# ── TRAIN MODELS ─────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    FEATURES = ['pp_runs','pp_wickets','pp_fours','pp_sixes',
                'pp_dot_pct','pp_boundary_pct','wickets_in_hand',
                'venue_avg','pitch_type','innings']
    X = df[FEATURES]; y = df['final_score']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(X_train); Xte_sc = sc.transform(X_test)

    models = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression":  (Ridge(alpha=1.0),   True),
        "Decision Tree":     (DecisionTreeRegressor(max_depth=8,random_state=42), False),
        "Random Forest":     (RandomForestRegressor(n_estimators=200,max_depth=15,
                                                     min_samples_leaf=5,random_state=42,n_jobs=-1), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=200,learning_rate=0.05,
                                                         max_depth=5,random_state=42), False),
    }
    results = {}
    for name,(model,is_lin) in models.items():
        Xtr = Xtr_sc if is_lin else X_train.values
        Xte = Xte_sc if is_lin else X_test.values
        model.fit(Xtr,y_train)
        preds = model.predict(Xte)
        results[name] = {
            "model":model,"preds":preds,"is_linear":is_lin,
            "MAE":round(mean_absolute_error(y_test,preds),2),
            "RMSE":round(np.sqrt(mean_squared_error(y_test,preds)),2),
            "R2":round(r2_score(y_test,preds),4),
        }
    return results, sc, X_test, y_test, FEATURES


def predict_score(pp_runs,pp_wickets,pp_fours,pp_sixes,pp_dot_pct,
                   pp_boundary_pct,venue_avg,pitch_type,innings,
                   model_name, results, scaler):
    wih = 10 - pp_wickets
    inp = pd.DataFrame([{"pp_runs":pp_runs,"pp_wickets":pp_wickets,
                          "pp_fours":pp_fours,"pp_sixes":pp_sixes,
                          "pp_dot_pct":pp_dot_pct,"pp_boundary_pct":pp_boundary_pct,
                          "wickets_in_hand":wih,"venue_avg":venue_avg,
                          "pitch_type":pitch_type,"innings":innings}])
    res = results[model_name]
    X   = scaler.transform(inp) if res["is_linear"] else inp.values
    return int(round(res["model"].predict(X)[0]))


# ── LOAD DATA & TRAIN ─────────────────────────────────────────
df = generate_dataset()
results, scaler, X_test, y_test, FEATURES = train_models(df)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/8/8d/Cricket_ball.jpg",
             width=60)
    st.markdown("## 🏏 T20 Score Predictor")
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("", ["🎯 Predict Score","📊 Model Metrics",
                          "🔍 EDA Charts","📋 Dataset"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model**")
    model_choice = st.selectbox("Select ML model", list(results.keys()), index=3)
    st.markdown("---")
    st.markdown("**Project by GK**")
    st.markdown("Stack: Python · Streamlit · Scikit-learn")
    best = max(results, key=lambda k: results[k]['R2'])
    st.success(f"⭐ Best: {best}\nR²={results[best]['R2']}")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════
if page == "🎯 Predict Score":
    st.markdown('<p class="main-title">🏏 T20 Final Score Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Enter Powerplay stats (Overs 1–6) → Get predicted final score</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.markdown("### Powerplay Stats (Overs 1–6)")

        pp_runs    = st.slider("PP Runs",          10, 90, 52, 1,
                                help="Total runs scored in overs 1–6")
        pp_wickets = st.slider("PP Wickets Lost",   0,  6,  2, 1,
                                help="Wickets lost in overs 1–6")
        pp_fours   = st.slider("Fours (4s) in PP",  0, 14,  5, 1)
        pp_sixes   = st.slider("Sixes (6s) in PP",  0,  8,  2, 1)
        pp_dot_pct = st.slider("Dot Ball %",        10, 65, 38, 1,
                                help="% of balls in PP that were dot balls")
        pp_bnd_pct = st.slider("Boundary Ball %",    5, 45, 20, 1,
                                help="% of balls in PP hit for boundaries")

        st.markdown("### Match Context")
        c1, c2, c3 = st.columns(3)
        with c1:
            venue_avg  = st.number_input("Venue Avg", 120, 200, 158,
                                          help="Historical average score at this ground")
        with c2:
            pitch_type = st.selectbox("Pitch", ["Slow/Spin","Flat","Bouncy"],
                                       index=1)
            pitch_num  = {"Slow/Spin":0,"Flat":1,"Bouncy":2}[pitch_type]
        with c3:
            innings    = st.selectbox("Innings", [1, 2], index=0)

        predict_btn = st.button("🎯 Predict Final Score", use_container_width=True,
                                  type="primary")

    with col2:
        # Always show live prediction
        pred = predict_score(pp_runs, pp_wickets, pp_fours, pp_sixes,
                              pp_dot_pct, pp_bnd_pct, venue_avg,
                              pitch_num, innings, model_choice, results, scaler)
        lo, hi = max(pred-18, 60), pred+18

        if   pred < 130: zone,zcls = "🔴 Low Score",   "zone-low"
        elif pred < 150: zone,zcls = "🟡 Below Par",   "zone-below"
        elif pred < 165: zone,zcls = "🔵 Par Score",   "zone-par"
        elif pred < 180: zone,zcls = "🟢 Good Score",  "zone-good"
        else:            zone,zcls = "🏆 High Score",  "zone-high"

        st.markdown(f"""
        <div class="pred-box">
            <div class="pred-label">Predicted Final Score</div>
            <div class="pred-score">{pred}</div>
            <div class="pred-range">Confidence range: {lo} – {hi} runs</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div style="text-align:center;margin:0.5rem 0"><span class="{zcls}">{zone}</span></div>',
                     unsafe_allow_html=True)

        st.markdown("### Derived Metrics")
        m1, m2 = st.columns(2)
        m1.metric("PP Run Rate",      f"{pp_runs/6:.2f} rpo")
        m2.metric("Wickets in Hand",  f"{10-pp_wickets}")
        m1.metric("Predicted RR",     f"{pred/20:.2f} rpo")
        m2.metric("PP Contribution",  f"{round(pp_runs/pred*100)}%")

        st.markdown("### All Models Comparison")
        for name in results:
            p = predict_score(pp_runs, pp_wickets, pp_fours, pp_sixes,
                               pp_dot_pct, pp_bnd_pct, venue_avg,
                               pitch_num, innings, name, results, scaler)
            star = "⭐" if name == model_choice else ""
            st.markdown(f"**{star}{name}** — `{p} runs`")

        st.markdown("### Score Zone Guide")
        zones = [("🔴 < 130","Low score"),("🟡 130–149","Below par"),
                 ("🔵 150–164","Par score"),("🟢 165–179","Good score"),
                 ("🏆 180+","High score")]
        for emoji, label in zones:
            st.markdown(f"{emoji} **{label}**")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — MODEL METRICS
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Metrics":
    st.markdown('<p class="main-title">📊 Model Performance</p>', unsafe_allow_html=True)
    st.markdown("Trained on **960 innings** | Tested on **240 innings**")

    # Metrics table
    mdf = pd.DataFrame({k:{"MAE":v["MAE"],"RMSE":v["RMSE"],"R² Score":v["R2"]}
                         for k,v in results.items()}).T.sort_values("R² Score",ascending=False)
    mdf.index.name = "Model"
    st.dataframe(mdf.style.background_gradient(subset=["R² Score"],cmap="Greens")
                          .background_gradient(subset=["MAE","RMSE"],cmap="Reds_r")
                          .format({"MAE":"{:.2f}","RMSE":"{:.2f}","R² Score":"{:.4f}"}),
                 use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Actual vs Predicted (Random Forest)")
        rf_res = results["Random Forest"]
        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(y_test, rf_res["preds"], alpha=0.4, s=20, color="#2E86C1")
        lo = min(y_test.min(), rf_res["preds"].min())-5
        hi = max(y_test.max(), rf_res["preds"].max())+5
        ax.plot([lo,hi],[lo,hi],"r--",linewidth=1.5,label="Perfect prediction")
        ax.set_xlabel("Actual Score"); ax.set_ylabel("Predicted Score")
        ax.set_title(f"Random Forest  R²={rf_res['R2']}  MAE={rf_res['MAE']}")
        ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("### Feature Importance (Random Forest)")
        rf_model = results["Random Forest"]["model"]
        imp_df = pd.DataFrame({"Feature":FEATURES,
                                "Importance":rf_model.feature_importances_})
        imp_df = imp_df.sort_values("Importance",ascending=True)
        fig2, ax2 = plt.subplots(figsize=(6,5))
        colors = ["#AED6F1"]*len(imp_df)
        colors[-1] = "#1F4E79"; colors[-2] = "#2980B9"; colors[-3] = "#2E86C1"
        ax2.barh(imp_df["Feature"], imp_df["Importance"],
                  color=colors, edgecolor="white")
        ax2.set_xlabel("Importance"); ax2.set_title("Feature Importance")
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    st.markdown("---")
    st.markdown("### Residual Plot (Gradient Boosting)")
    gb_res = results["Gradient Boosting"]
    residuals = y_test.values - gb_res["preds"]
    fig3, ax3 = plt.subplots(figsize=(10,3.5))
    ax3.scatter(gb_res["preds"], residuals, alpha=0.35, s=18, color="#1E8449")
    ax3.axhline(0, color="#C0392B", linestyle="--", linewidth=2)
    ax3.set_xlabel("Predicted Score"); ax3.set_ylabel("Residual")
    ax3.set_title("Residual Plot — Gradient Boosting (closer to 0 = better)")
    plt.tight_layout()
    st.pyplot(fig3); plt.close()


# ══════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ══════════════════════════════════════════════════════════════
elif page == "🔍 EDA Charts":
    st.markdown('<p class="main-title">🔍 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown(f"Dataset: **{len(df)} innings** across **13 leagues**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Final Score Distribution")
        fig,ax = plt.subplots(figsize=(6,4))
        ax.hist(df["final_score"],bins=30,color="#2E86C1",edgecolor="white",linewidth=0.5)
        ax.axvline(df["final_score"].mean(),color="#E74C3C",linestyle="--",linewidth=2,
                    label=f"Mean: {df['final_score'].mean():.0f}")
        ax.set_xlabel("Final Score"); ax.set_ylabel("Count")
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("### PP Wickets → Avg Final Score")
        grp = df.groupby("pp_wickets")["final_score"].mean().reset_index()
        colors_w = ["#1A5276","#1F618D","#2874A6","#2E86C1","#F39C12","#E67E22","#C0392B"]
        fig2,ax2 = plt.subplots(figsize=(6,4))
        bars = ax2.bar(grp["pp_wickets"],grp["final_score"],color=colors_w,edgecolor="white")
        for bar,val in zip(bars,grp["final_score"]):
            ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
                      f"{val:.0f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
        ax2.set_xlabel("PP Wickets Lost"); ax2.set_ylabel("Avg Final Score")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("### PP Runs vs Final Score")
    fig3,ax3 = plt.subplots(figsize=(10,4))
    sc = ax3.scatter(df["pp_runs"],df["final_score"],
                      c=df["pp_wickets"],cmap="RdYlGn_r",alpha=0.4,s=18,vmin=0,vmax=6)
    plt.colorbar(sc,ax=ax3,label="PP Wickets")
    m,b = np.polyfit(df["pp_runs"],df["final_score"],1)
    xs = np.linspace(df["pp_runs"].min(),df["pp_runs"].max(),100)
    ax3.plot(xs,m*xs+b,"navy",linewidth=2,label=f"Trend: y={m:.2f}x+{b:.0f}")
    ax3.set_xlabel("PP Runs"); ax3.set_ylabel("Final Score")
    ax3.legend(); plt.tight_layout(); st.pyplot(fig3); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Avg Final Score by League")
        league_means = df.groupby("league")["final_score"].mean().sort_values()
        fig4,ax4 = plt.subplots(figsize=(6,5))
        ax4.barh(league_means.index,league_means.values,color="#27AE60",edgecolor="white")
        ax4.set_xlabel("Avg Final Score")
        for i,(idx,val) in enumerate(league_means.items()):
            ax4.text(val+0.5,i,f"{val:.0f}",va="center",fontsize=8)
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    with col4:
        st.markdown("### Correlation Heatmap")
        feat_cols = ["pp_runs","pp_wickets","pp_fours","pp_sixes",
                     "pp_dot_pct","pp_boundary_pct","wickets_in_hand",
                     "venue_avg","pitch_type","final_score"]
        corr = df[feat_cols].corr()
        mask = np.triu(np.ones_like(corr,dtype=bool))
        fig5,ax5 = plt.subplots(figsize=(6,5))
        sns.heatmap(corr,ax=ax5,mask=mask,annot=True,fmt=".2f",
                     cmap="RdBu_r",center=0,linewidths=0.3,
                     annot_kws={"size":7},vmin=-1,vmax=1)
        ax5.tick_params(axis="x",rotation=45,labelsize=7)
        ax5.tick_params(axis="y",labelsize=7)
        plt.tight_layout(); st.pyplot(fig5); plt.close()


# ══════════════════════════════════════════════════════════════
# PAGE 4 — DATASET
# ══════════════════════════════════════════════════════════════
elif page == "📋 Dataset":
    st.markdown('<p class="main-title">📋 Dataset</p>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Innings",  len(df))
    c2.metric("Leagues",        df["league"].nunique())
    c3.metric("Features (X)",   10)
    c4.metric("Target (y)",     "final_score")

    st.markdown("---")
    st.markdown("### Filter & Explore")
    col1, col2 = st.columns(2)
    with col1:
        league_filter = st.multiselect("Filter by League",
                                        df["league"].unique().tolist(),
                                        default=df["league"].unique().tolist()[:3])
    with col2:
        pp_range = st.slider("Filter by PP Runs", int(df.pp_runs.min()),
                               int(df.pp_runs.max()), (20, 80))

    filtered = df[df["league"].isin(league_filter) &
                   df["pp_runs"].between(pp_range[0], pp_range[1])]
    st.markdown(f"Showing **{len(filtered)}** rows")
    st.dataframe(filtered.head(50), use_container_width=True)

    st.markdown("### Basic Statistics")
    stats_cols = ["pp_runs","pp_wickets","pp_run_rate","pp_boundary_pct",
                   "pp_dot_pct","wickets_in_hand","final_score"]
    st.dataframe(df[stats_cols].describe().round(2), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Full Dataset (CSV)", csv,
                        "t20_dataset.csv", "text/csv", use_container_width=True)
