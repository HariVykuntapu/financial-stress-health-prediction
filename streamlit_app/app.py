import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Stress & Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .main { background-color: #f5f8ff; }

    .header-banner {
        background: linear-gradient(135deg, #0d2b55 0%, #1a4a8a 100%);
        color: white;
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .header-banner h1 { color: white; font-size: 2rem; margin: 0; font-weight: 700; }
    .header-banner p  { color: #cde4ff; margin: 0.4rem 0 0 0; font-size: 1rem; }

    .metric-card {
        background: white;
        border-left: 5px solid #1a4a8a;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }
    .metric-card h3 { color: #0d2b55; margin: 0 0 0.3rem 0; font-size: 1.05rem; }
    .metric-card p  { color: #444; margin: 0; font-size: 0.92rem; line-height: 1.5; }

    .section-header {
        color: #0d2b55;
        font-size: 1.3rem;
        font-weight: 700;
        border-bottom: 2px solid #1a4a8a;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }

    .risk-low    { color: #27ae60; font-size: 1.6rem; font-weight: 800; }
    .risk-medium { color: #f39c12; font-size: 1.6rem; font-weight: 800; }
    .risk-high   { color: #e74c3c; font-size: 1.6rem; font-weight: 800; }

    .footer {
        text-align: center;
        color: #666;
        font-size: 0.82rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e8f0fe;
        border-radius: 6px 6px 0 0;
        color: #0d2b55;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d2b55 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>💙 Financial Stress & Heart Disease Risk Predictor</h1>
    <p>Research by <strong>Hari Vykuntapu</strong> | MS Artificial Intelligence, Southwest Baptist University</p>
</div>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_HERE)

@st.cache_resource
def load_artifacts():
    model_path   = os.path.join(BASE_DIR, "models", "xgb_best_model.pkl")
    scaler_path  = os.path.join(BASE_DIR, "models", "scaler.pkl")
    feature_path = os.path.join(BASE_DIR, "models", "feature_cols.pkl")
    model, scaler, features = None, None, None
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    if os.path.exists(feature_path):
        features = joblib.load(feature_path)
    return model, scaler, features

model, scaler, feature_cols = load_artifacts()

# ── FSS helpers ───────────────────────────────────────────────────────────────
def compute_fss(income2, medcost, employ1):
    income_inverse = (8 - income2) / 7.0
    medcost_burden = float(medcost)
    unemployment   = 1.0 if employ1 == 2 else 0.0
    return round((income_inverse * 0.40) + (medcost_burden * 0.35) + (unemployment * 0.25), 4)

def fss_label(score):
    if score < 0.33:
        return "Low", "#27ae60"
    elif score < 0.66:
        return "Medium", "#f39c12"
    else:
        return "High", "#e74c3c"

# ── Gauge chart ───────────────────────────────────────────────────────────────
def draw_gauge(risk_pct):
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw=dict(polar=False))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    for x0, x1, c, lbl in [(0, 3.33, "#27ae60", "Low"), (3.33, 6.66, "#f39c12", "Medium"), (6.66, 10, "#e74c3c", "High")]:
        ax.barh(2, x1 - x0, left=x0, height=1.2, color=c, alpha=0.85)
        ax.text((x0 + x1) / 2, 2, lbl, ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    needle_x = risk_pct / 10.0
    ax.annotate("", xy=(needle_x, 3.5), xytext=(needle_x, 2.6),
                arrowprops=dict(arrowstyle="->", color="#0d2b55", lw=2.5))
    ax.text(5, 0.7, f"{risk_pct:.1f}% Risk", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#0d2b55")
    fig.patch.set_facecolor("#f5f8ff")
    return fig

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_home, tab_predict, tab_research, tab_about = st.tabs([
    "🏠 Home", "🔮 Predict Risk", "📊 Research", "👩‍🔬 About"
])

# ════════════════════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown('<div class="section-header">The Question That Started This</div>', unsafe_allow_html=True)

    st.markdown("""
    Heart disease is the leading cause of death in America. And nearly every model built to predict it
    requires clinical data — cholesterol readings, blood pressure, EKG results — that a huge chunk of
    the population never gets, because they can't afford to see a doctor in the first place.

    That felt like a problem worth looking at. The CDC surveys 400,000+ Americans every year and asks
    them directly: *How much do you earn? Did you skip care because of cost? Are you employed?*
    Nobody had built a heart disease predictor anchored to those questions.

    **So I did.** I created the **Financial Stress Score** — a single number that captures how much
    economic pressure someone is under — and tested whether it predicts cardiovascular risk.
    The answer was yes, more clearly than I expected.
    """)

    st.markdown("---")
    st.markdown('<div class="section-header">What the Data Showed</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📉 The Income Gap Is Real</h3>
            <p>Walk the income ladder from &gt;$75K down to &lt;$10K and heart disease rates more than
            double. It's one of the sharpest gradients in the entire BRFSS dataset — and it's
            remarkably consistent across every demographic slice I checked.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🏥 Skipping the Doctor Has a Cost</h3>
            <p>People who couldn't afford care had measurably higher heart disease rates. That's
            not surprising in theory, but the magnitude was striking. Unaffordable healthcare
            isn't just a hardship — it shows up as a cardiovascular risk factor in the data.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 The Score Actually Predicts</h3>
            <p>When I ran SHAP analysis on the XGBoost model, the Financial Stress Score ranked
            as a top predictor — not a footnote. Economic pressure isn't just correlated with
            poor heart health. It predicts it well enough to be useful.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("**Dataset:** CDC BRFSS 2015 (n ≈ 441,000) | **Model:** XGBoost + SMOTE | "
            "**Author:** Hari Vykuntapu | "
            "**Code:** [github.com/HariVykuntapu/financial-stress-health-prediction]"
            "(https://github.com/HariVykuntapu/financial-stress-health-prediction)")

# ════════════════════════════════════════════════════════════════════════════
# PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="section-header">Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown("Fill in the fields below. Your Financial Stress Score is calculated live as a research metric. The XGBoost model uses the raw financial components (income, healthcare barrier, employment) alongside the clinical factors.")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 💰 Financial Stress Factors")

        income_map = {
            "Less than $10,000": 1, "$10,000 – $14,999": 2, "$15,000 – $19,999": 3,
            "$20,000 – $24,999": 4, "$25,000 – $34,999": 5, "$35,000 – $49,999": 6,
            "$50,000 – $74,999": 7, "$75,000 or more": 8,
        }
        income_label = st.selectbox("Annual Household Income", list(income_map.keys()), index=4)
        income2 = income_map[income_label]

        medcost_ans = st.radio(
            "In the past 12 months, was there a time when you couldn't see a doctor because of cost?",
            ["No, I could afford care", "Yes, I could not afford care"],
            index=0,
        )
        medcost = 1 if "Yes" in medcost_ans else 0

        employ_map = {
            "Employed (wage / self-employed)": 1,
            "Unemployed": 2,
            "Other (homemaker / student / retired / unable to work)": 3,
        }
        employ_label = st.selectbox("Current Employment Status", list(employ_map.keys()))
        employ1 = employ_map[employ_label]

        fss = compute_fss(income2, medcost, employ1)
        fss_level, fss_color = fss_label(fss)

        st.markdown("---")
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Your Financial Stress Score</h3>
            <p style="font-size:2rem; font-weight:800; color:{fss_color};">{fss:.3f}</p>
            <p>Level: <strong style="color:{fss_color};">{fss_level} Financial Stress</strong></p>
            <p style="font-size:0.82rem; color:#666;">Formula: (Income Inverse × 0.40) + (Care Barrier × 0.35) + (Unemployment × 0.25) — displayed as a research metric; model uses the raw components directly.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 🏥 Clinical & Demographic Factors")

        age = st.slider("Age", min_value=18, max_value=80, value=45, step=1)
        bmi = st.slider("BMI", min_value=15.0, max_value=60.0, value=27.0, step=0.5)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        sex_val = 2 if sex == "Male" else 1

        smokes = st.radio("Have you smoked at least 100 cigarettes in your lifetime?", ["No", "Yes"], horizontal=True)
        smoke_val = 1 if smokes == "Yes" else 2

        diabetes_map = {"No": 3, "Pre-diabetes / Borderline": 4, "Yes": 1, "Yes (during pregnancy only)": 2}
        diabetes_label = st.selectbox("Diabetes Status", list(diabetes_map.keys()))
        diabete3 = diabetes_map[diabetes_label]

        st.markdown("---")

        predict_btn = st.button("🔮 Predict My Heart Disease Risk", type="primary", use_container_width=True)

        if predict_btn:
            if model is None:
                st.error("Model files not found. Make sure models/xgb_best_model.pkl is present in the repo.")
            else:
                input_dict = {
                    "INCOME2": income2,
                    "MEDCOST": medcost,
                    "EMPLOY1": employ1,
                    "_BMI5": bmi * 100,
                    "SMOKE100": smoke_val,
                    "_AGE80": age,
                    "SEX": sex_val,
                    "DIABETE3": diabete3,
                }
                X_input = pd.DataFrame([[input_dict[c] for c in feature_cols]], columns=feature_cols)
                prob = model.predict_proba(X_input)[0][1] * 100

                st.markdown("---")
                st.markdown("#### Prediction Result")
                gauge_fig = draw_gauge(prob)
                st.pyplot(gauge_fig, use_container_width=False)

                if prob < 33:
                    risk_class, risk_text = "risk-low", "Low Risk"
                    advice = "Your predicted risk is low. Staying financially stable and keeping up with preventive care are the best things you can do."
                elif prob < 66:
                    risk_class, risk_text = "risk-medium", "Moderate Risk"
                    advice = "Moderate risk. Consider talking to a doctor — especially if financial stress is making it hard to access regular care."
                else:
                    risk_class, risk_text = "risk-high", "High Risk"
                    advice = "Elevated risk. Financial stress is likely contributing — please reach out to a healthcare provider and explore community health resources near you."

                st.markdown(f'<p class="{risk_class}">{prob:.1f}% — {risk_text}</p>', unsafe_allow_html=True)
                st.info(advice)
                st.caption("Research tool only — not a medical diagnosis. Please consult a licensed healthcare professional.")

# ════════════════════════════════════════════════════════════════════════════
# RESEARCH
# ════════════════════════════════════════════════════════════════════════════
with tab_research:
    st.markdown('<div class="section-header">EDA Charts</div>', unsafe_allow_html=True)

    eda_images = {
        "Class Distribution (MICHD)": "outputs/eda/class_distribution.png",
        "Heart Disease Rate by Income": "outputs/eda/hd_rate_by_income.png",
        "Heart Disease Rate by Healthcare Cost": "outputs/eda/hd_rate_by_medcost.png",
        "Heart Disease Rate by Employment": "outputs/eda/hd_rate_by_employment.png",
        "Heart Disease Rate by FSS Bucket": "outputs/eda/hd_rate_by_stress_bucket.png",
        "Correlation Heatmap": "outputs/eda/correlation_heatmap.png",
    }
    cols = st.columns(2)
    for i, (title, rel_path) in enumerate(eda_images.items()):
        full_path = os.path.join(BASE_DIR, rel_path)
        with cols[i % 2]:
            if os.path.exists(full_path):
                st.image(full_path, caption=title, use_container_width=True)
            else:
                st.info(f"Chart not found: {title}")

    st.markdown("---")
    st.markdown('<div class="section-header">How the Financial Stress Score Works</div>', unsafe_allow_html=True)
    st.markdown(r"""
    The **Financial Stress Score (FSS)** collapses three CDC survey variables into a single 0–1 number.
    The weights reflect how directly each component cuts off access to cardiovascular care:

    $$FSS = ({\rm income\_inverse\_norm} \times 0.40) + ({\rm medcost\_burden} \times 0.35) + ({\rm unemployment\_flag} \times 0.25)$$

    | Component | Weight | What it captures |
    |-----------|--------|-----------------|
    | Income (inverted & normalized) | 40% | Low income = less buffer for healthcare, food, housing |
    | Healthcare cost barrier | 35% | Literally couldn't afford care when needed |
    | Unemployment flag | 25% | Job loss = no income, often no insurance |

    **Score buckets:** Low [0 – 0.33) · Medium [0.33 – 0.66) · High [0.66 – 1.0]
    """)

    st.markdown("---")
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    results_path = os.path.join(BASE_DIR, "outputs/results/model_comparison.csv")
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path, index_col=0)
        st.dataframe(results_df.style.highlight_max(axis=0, color="#cde4ff").format("{:.4f}"),
                     use_container_width=True)
    else:
        st.info("Run notebook 03_model_training.ipynb to generate model results.")

    st.markdown("---")
    st.markdown('<div class="section-header">Result Charts</div>', unsafe_allow_html=True)

    result_images = {
        "ROC Curves": "outputs/results/roc_curves.png",
        "Precision-Recall Curves": "outputs/results/precision_recall_curves.png",
        "SHAP Feature Importance": "outputs/results/shap_importance.png",
        "SHAP Beeswarm": "outputs/results/shap_beeswarm.png",
        "Confusion Matrix (XGBoost)": "outputs/results/confusion_matrix.png",
        "Model Comparison": "outputs/results/model_comparison_chart.png",
        "FSS vs Heart Disease Risk": "outputs/results/fss_vs_risk_curve.png",
    }
    cols2 = st.columns(2)
    for i, (title, rel_path) in enumerate(result_images.items()):
        full_path = os.path.join(BASE_DIR, rel_path)
        with cols2[i % 2]:
            if os.path.exists(full_path):
                st.image(full_path, caption=title, use_container_width=True)
            else:
                st.info(f"Chart not found: {title}")

# ════════════════════════════════════════════════════════════════════════════
# ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    col_bio, col_links = st.columns([2, 1])

    with col_bio:
        st.markdown('<div class="section-header">About</div>', unsafe_allow_html=True)
        st.markdown("""
        I'm **Hari Vykuntapu**, an MS AI student, and this project grew out of something that kept
        bothering me: almost every heart disease risk model I read about assumes you have clinical data.
        Cholesterol panels, blood pressure cuffs, maybe an EKG. But what about the 30 million Americans
        who skip doctor visits because they can't afford them?

        I wanted to see what the data looked like if you started from the economic side instead. The
        CDC already asks 400,000 people every year whether they can afford care, what they earn, whether
        they're working. That's a lot of signal that most models just ignore.

        So I built the **Financial Stress Score** — weighting income deprivation most heavily, then
        healthcare affordability, then unemployment — and ran it through an XGBoost classifier alongside
        the usual clinical variables. The FSS landed as a top SHAP feature. I didn't expect it to be
        that strong relative to BMI and smoking history, but there it was.

        The honest caveat: AUROC in the low 0.79s means this isn't a replacement for a real clinical
        workup. But as a screening signal — especially for people who never make it to the clinic — I
        think it's worth taking seriously.

        I'm looking for data science and ML engineering roles, particularly anywhere health equity
        intersects with applied machine learning.

        ---

        ### Dataset
        Centers for Disease Control and Prevention. (2015). *Behavioral Risk Factor Surveillance System
        (BRFSS) Survey Data and Documentation*. U.S. Department of Health and Human Services.
        https://www.cdc.gov/brfss/
        """)

    with col_links:
        st.markdown('<div class="section-header">Contact</div>', unsafe_allow_html=True)
        st.markdown("""
        **Email**
        hpvykuntapu@gmail.com

        **LinkedIn**
        [linkedin.com/in/harivykuntapu](https://www.linkedin.com/in/harivykuntapu)

        **GitHub**
        [github.com/HariVykuntapu](https://github.com/HariVykuntapu)

        **Repository**
        [financial-stress-health-prediction](https://github.com/HariVykuntapu/financial-stress-health-prediction)

        ---

        **Program**
        MS Artificial Intelligence
        Southwest Baptist University
        United States
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    © 2025 Hari Vykuntapu · MS Artificial Intelligence, Southwest Baptist University ·
    Research: Financial Stress & Heart Disease Risk Prediction · Data: CDC BRFSS 2015
</div>
""", unsafe_allow_html=True)
