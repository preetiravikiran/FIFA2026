"""
FIFA 26 Player Ratings — Interactive Analytics Study Guide
MBA Business Analytics | Companion to the MLR / ANOVA / Logistic Regression study guides

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# PAGE CONFIG + THEME
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="FIFA 2026 Player Ratings",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#13294B"
TEAL = "#1B9AAA"
TEAL_LIGHT = "#7FD1D9"
TEAL_BG = "#E8F6F7"
GREY = "#6B7280"
PALETTE = [TEAL_LIGHT, TEAL, NAVY, "#3D6B8A", "#9FD8DD", "#0E5C66"]

st.markdown(f"""
<style>
    .main {{ background-color: #FAFBFC; }}
    h1, h2, h3 {{ color: {NAVY} !important; }}
    [data-testid="stSidebar"] {{ background-color: {NAVY}; }}
    [data-testid="stSidebar"] * {{ color: #F3F4F6 !important; }}
    [data-testid="stSidebar"] .stRadio label {{ color: #F3F4F6 !important; }}
    div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
    .teal-divider {{ border-top: 3px solid {TEAL}; margin: 0.4rem 0 1.2rem 0; }}
    .callout {{
        background-color: {TEAL_BG};
        border-left: 4px solid {TEAL};
        padding: 0.9rem 1.1rem;
        border-radius: 4px;
        margin: 0.8rem 0 1.2rem 0;
    }}
    .callout b {{ color: {NAVY}; }}
</style>
""", unsafe_allow_html=True)


def section_header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)


def callout(title, body):
    st.markdown(f'<div class="callout"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
DATA_URL = ("https://raw.githubusercontent.com/ismailoksuz/EAFC26-DataHub/"
            "main/data/players.csv")
LOCAL_FILE = "fifa26_player_data.csv"


@st.cache_data(show_spinner="Loading FIFA 26 player data...")
def load_data():
    import os
    if os.path.exists(LOCAL_FILE):
        df = pd.read_csv(LOCAL_FILE)
        return df

    # Fallback: pull and clean the raw dataset from GitHub
    raw = pd.read_csv(DATA_URL, low_memory=False)
    outfield = raw[raw["player_positions"] != "GK"].copy()
    outfield = outfield.dropna(
        subset=["pace", "shooting", "passing", "dribbling", "defending", "physic", "overall"]
    )
    first_pos = outfield["player_positions"].str.split(",").str[0].str.strip()
    pos_map = {
        "ST": "Forward", "CF": "Forward", "LW": "Forward", "RW": "Forward",
        "CAM": "Midfield", "CM": "Midfield", "CDM": "Midfield", "LM": "Midfield", "RM": "Midfield",
        "CB": "Defender", "LB": "Defender", "RB": "Defender", "LWB": "Defender", "RWB": "Defender",
    }
    outfield["position_group"] = first_pos.map(pos_map)
    outfield = outfield.dropna(subset=["position_group"])

    def age_bucket(a):
        if a < 23:
            return "Young (<23)"
        elif a <= 29:
            return "Prime (23-29)"
        return "Veteran (30+)"

    outfield["age_group"] = outfield["age"].apply(age_bucket)
    keep_cols = ["short_name", "long_name", "player_positions", "position_group", "club_name",
                 "league_name", "nationality_name", "age", "age_group", "height_cm", "weight_kg",
                 "preferred_foot", "weak_foot", "skill_moves", "international_reputation",
                 "work_rate", "overall", "potential", "value_eur", "wage_eur", "pace", "shooting",
                 "passing", "dribbling", "defending", "physic"]
    return outfield[keep_cols].copy()


df_full = load_data()

ATTRS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
NUMERIC_VARS = ["overall", "potential", "age", "height_cm", "weight_kg", "value_eur",
                 "wage_eur"] + ATTRS
NICE_NAMES = {
    "overall": "Overall", "potential": "Potential", "age": "Age", "height_cm": "Height (cm)",
    "weight_kg": "Weight (kg)", "value_eur": "Market Value (€)", "wage_eur": "Wage (€)",
    "pace": "Pace", "shooting": "Shooting", "passing": "Passing", "dribbling": "Dribbling",
    "defending": "Defending", "physic": "Physical",
}


def nice(col):
    return NICE_NAMES.get(col, col.replace("_", " ").title())


# ----------------------------------------------------------------------------
# SIDEBAR — NAVIGATION + GLOBAL FILTERS
# ----------------------------------------------------------------------------
st.sidebar.markdown("# ⚽ FIFA 26\n### Analytics Study Guide")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Navigate",
    ["📊 Descriptives", "📈 ANOVA", "📉 Linear Regression", "🎯 Logistic Regression"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters (apply everywhere)")

pos_options = sorted(df_full["position_group"].dropna().unique().tolist())
selected_pos = st.sidebar.multiselect("Position group", pos_options, default=pos_options)

age_min, age_max = int(df_full["age"].min()), int(df_full["age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

top_leagues = df_full["league_name"].value_counts().head(15).index.tolist()
league_filter = st.sidebar.multiselect(
    "League (leave empty = all leagues)", sorted(top_leagues), default=[]
)

df = df_full[
    df_full["position_group"].isin(selected_pos)
    & df_full["age"].between(age_range[0], age_range[1])
]
if league_filter:
    df = df[df["league_name"].isin(league_filter)]

st.sidebar.markdown("---")
st.sidebar.metric("Players in current filter", f"{len(df):,}")
st.sidebar.caption(
    "Source: EA Sports FC 26 (FIFA 26) player ratings, via SoFIFA. "
    "Outfield players only — goalkeepers are scored on a different attribute set."
)

if len(df) < 30:
    st.warning("Fewer than 30 players match the current filters — widen the filters in the "
               "sidebar for stable statistics.")
    st.stop()

st.title("FIFA 2026 Player Ratings")
st.markdown(
    "This page is based on the EA Sports FC 26 player ratings. Every player in the "
    "game is scored on an overall rating (0–99) built from six broad attributes — Pace, "
    "Shooting, Passing, Dribbling, Defending and Physical — which themselves aggregate dozens "
    "of finer sub-attributes. The dataset used here contains 18,405 players from the FIFA 26 / "
    "FC 26 player database. After removing 2,062 goalkeepers — who are rated on a separate set "
    "of goalkeeping attributes rather than the six outfield attributes — 16,343 outfield "
    "players remain."
)
st.markdown("---")

# ============================================================================
# SECTION 1 — DESCRIPTIVES
# ============================================================================
if section == "📊 Descriptives":
    section_header("Descriptive Statistics", "Explore the shape of the data before modeling it.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", f"{len(df):,}")
    c2.metric("Avg. Overall", f"{df['overall'].mean():.1f}")
    c3.metric("Avg. Age", f"{df['age'].mean():.1f}")
    c4.metric("Positions", df["position_group"].nunique())

    st.markdown("### Explore a single variable")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        var = st.selectbox("Variable", NUMERIC_VARS, index=0, format_func=nice)
        bins = st.slider("Histogram bins", 10, 80, 35)
        color_by_pos = st.checkbox("Color by position group", value=True)
        st.markdown(f"""
        | Stat | Value |
        |---|---|
        | Mean | {df[var].mean():,.2f} |
        | Median | {df[var].median():,.2f} |
        | Std. Dev. | {df[var].std():,.2f} |
        | Min | {df[var].min():,.2f} |
        | Max | {df[var].max():,.2f} |
        """)
    with col_b:
        fig = px.histogram(
            df, x=var, nbins=bins, color="position_group" if color_by_pos else None,
            color_discrete_sequence=PALETTE, opacity=0.85,
            title=f"Distribution of {nice(var)}",
        )
        fig.update_layout(bargap=0.02, title_font_color=NAVY, legend_title_text="Position")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Correlation matrix")
    corr_vars = st.multiselect(
        "Variables to include", NUMERIC_VARS, default=["overall"] + ATTRS, format_func=nice
    )
    if len(corr_vars) >= 2:
        corr = df[corr_vars].corr().round(2)
        fig = px.imshow(
            corr, text_auto=True, color_continuous_scale="Blues",
            x=[nice(c) for c in corr_vars], y=[nice(c) for c in corr_vars],
            zmin=-1, zmax=1, title="Pairwise Correlation",
        )
        fig.update_layout(title_font_color=NAVY)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Pick at least two variables to see a correlation matrix.")

    st.markdown("### Compare two variables")
    col_x, col_y, col_c = st.columns(3)
    with col_x:
        x_var = st.selectbox("X-axis", NUMERIC_VARS, index=NUMERIC_VARS.index("pace"), format_func=nice)
    with col_y:
        y_var = st.selectbox("Y-axis", NUMERIC_VARS, index=NUMERIC_VARS.index("overall"), format_func=nice)
    with col_c:
        trendline = st.checkbox("Add trendline", value=True)
    fig = px.scatter(
        df, x=x_var, y=y_var, color="position_group", color_discrete_sequence=PALETTE,
        opacity=0.5, trendline="ols" if trendline else None,
        title=f"{nice(y_var)} vs {nice(x_var)}",
        hover_data=["short_name", "club_name"],
    )
    fig.update_layout(title_font_color=NAVY, legend_title_text="Position")
    st.plotly_chart(fig, width="stretch")

    with st.expander("View / download filtered data"):
        st.dataframe(df.reset_index(drop=True), width="stretch", height=300)
        st.download_button(
            "Download filtered data as CSV", df.to_csv(index=False).encode("utf-8"),
            "fifa26_filtered.csv", "text/csv",
        )

# ============================================================================
# SECTION 2 — ANOVA
# ============================================================================
elif section == "📈 ANOVA":
    section_header("One-Way ANOVA", "Does the mean of a numeric outcome differ across groups?")

    col1, col2 = st.columns(2)
    with col1:
        outcome = st.selectbox(
            "Outcome variable (numeric)", NUMERIC_VARS,
            index=NUMERIC_VARS.index("overall"), format_func=nice,
        )
    with col2:
        group_options = {
            "Position group": "position_group",
            "Preferred foot": "preferred_foot",
            "Age group": "age_group",
            "Work rate": "work_rate",
        }
        group_label = st.selectbox("Grouping variable (categorical)", list(group_options.keys()))
        group_var = group_options[group_label]

    sub = df.dropna(subset=[outcome, group_var]).copy()
    group_counts = sub[group_var].value_counts()
    valid_groups = group_counts[group_counts >= 5].index.tolist()
    sub = sub[sub[group_var].isin(valid_groups)]
    k = sub[group_var].nunique()

    if k < 2:
        st.warning("Need at least two groups with 5+ players each — adjust filters or grouping variable.")
        st.stop()

    desc = sub.groupby(group_var)[outcome].agg(["count", "mean", "std"]).round(2)
    desc.columns = ["N", "Mean", "Std. Dev."]
    st.markdown(f"### Group descriptives — {nice(outcome)} by {group_label}")
    st.dataframe(desc, width="stretch")

    fig = px.box(
        sub, x=group_var, y=outcome, color=group_var, color_discrete_sequence=PALETTE,
        points="outliers", title=f"{nice(outcome)} by {group_label}",
    )
    fig.update_layout(showlegend=False, title_font_color=NAVY)
    st.plotly_chart(fig, width="stretch")

    groups = [sub.loc[sub[group_var] == g, outcome] for g in valid_groups]
    f_stat, p_val = stats.f_oneway(*groups)

    formula_safe_group = "C(Q('group_var'))"
    sub_renamed = sub.rename(columns={outcome: "y", group_var: "grp"})
    anova_model = smf.ols("y ~ C(grp)", data=sub_renamed).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)

    ss_between = anova_table["sum_sq"].iloc[0]
    ss_total = anova_table["sum_sq"].sum()
    eta_sq = ss_between / ss_total

    st.markdown("### ANOVA table")
    show_table = anova_table.copy()
    show_table.index = [group_label, "Residual"]
    show_table.columns = ["Sum of Squares", "df", "F", "p-value"]
    st.dataframe(show_table.round(4), width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric("F-statistic", f"{f_stat:.2f}")
    m2.metric("p-value", f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001")
    m3.metric("Eta-squared (effect size)", f"{eta_sq:.4f}")

    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
    if p_val < alpha:
        st.success(
            f"Significant at α = {alpha}: mean {nice(outcome)} is **not** equal across all "
            f"{group_label.lower()} groups (F = {f_stat:.2f}, p = {p_val:.4g})."
        )
    else:
        st.info(
            f"Not significant at α = {alpha}: we cannot conclude mean {nice(outcome)} differs "
            f"across {group_label.lower()} groups (F = {f_stat:.2f}, p = {p_val:.4g})."
        )

    if eta_sq < 0.01:
        effect_text = "negligible"
    elif eta_sq < 0.06:
        effect_text = "small"
    elif eta_sq < 0.14:
        effect_text = "medium"
    else:
        effect_text = "large"
    callout(
        "Statistical vs. practical significance",
        f"η² = {eta_sq:.4f} indicates a <b>{effect_text}</b> effect size. With N = {len(sub):,}, "
        "even tiny mean differences can be statistically significant — always check the effect size "
        "alongside the p-value before deciding a result is practically meaningful.",
    )

    if k > 2 and p_val < alpha:
        st.markdown("### Post-hoc comparison (Tukey HSD)")
        tukey = pairwise_tukeyhsd(sub[outcome], sub[group_var])
        tukey_df = pd.DataFrame(
            data=tukey.summary().data[1:], columns=tukey.summary().data[0]
        )
        st.dataframe(tukey_df, width="stretch")

# ============================================================================
# SECTION 3 — LINEAR REGRESSION
# ============================================================================
elif section == "📉 Linear Regression":
    section_header("Multiple Linear Regression", "Predict a numeric outcome from several predictors.")

    col1, col2 = st.columns([1, 2])
    with col1:
        outcome = st.selectbox(
            "Outcome (Y)", NUMERIC_VARS, index=NUMERIC_VARS.index("overall"), format_func=nice
        )
    with col2:
        predictor_choices = [v for v in NUMERIC_VARS if v != outcome]
        default_preds = [p for p in ATTRS if p != outcome] or predictor_choices[:3]
        predictors = st.multiselect(
            "Predictors (X)", predictor_choices, default=default_preds, format_func=nice
        )

    if len(predictors) == 0:
        st.warning("Select at least one predictor — defaulting back to the standard six attributes.")
        predictors = default_preds

    sub = df.dropna(subset=[outcome] + predictors).copy()
    X = sm.add_constant(sub[predictors])
    y = sub[outcome]
    model = sm.OLS(y, X).fit()

    st.markdown("### Model fit")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²", f"{model.rsquared:.3f}")
    m2.metric("Adj. R²", f"{model.rsquared_adj:.3f}")
    m3.metric("F-statistic", f"{model.fvalue:,.1f}")
    m4.metric("N", f"{int(model.nobs):,}")

    st.markdown("### Coefficients")
    coef_table = pd.DataFrame({
        "Coefficient": model.params, "Std. Error": model.bse,
        "t-stat": model.tvalues, "p-value": model.pvalues,
    })
    coef_table.index = ["Constant"] + [nice(p) for p in predictors]
    st.dataframe(coef_table.round(4), width="stretch")

    st.markdown("### Multicollinearity check (VIF)")
    if len(predictors) >= 2:
        vif_df = pd.DataFrame({
            "Variable": [nice(p) for p in predictors],
            "VIF": [variance_inflation_factor(X.values, i + 1) for i in range(len(predictors))],
        })
        st.dataframe(vif_df.round(2), width="stretch")
        if (vif_df["VIF"] > 5).any():
            callout(
                "Multicollinearity detected",
                "One or more predictors have VIF &gt; 5. Their individual coefficients should be "
                "interpreted cautiously, since their effects overlap with another predictor in the model.",
            )
    else:
        st.caption("Add a second predictor to compute VIF.")

    st.markdown("### Residual diagnostics")
    sub["fitted"] = model.fittedvalues
    sub["resid"] = model.resid

    d1, d2 = st.columns(2)
    with d1:
        fig = px.scatter(
            sub, x="fitted", y="resid", opacity=0.4,
            color_discrete_sequence=[TEAL], title="Residuals vs Fitted",
        )
        fig.add_hline(y=0, line_dash="dash", line_color=NAVY)
        fig.update_layout(title_font_color=NAVY, xaxis_title="Fitted values", yaxis_title="Residual")
        st.plotly_chart(fig, width="stretch")
    with d2:
        fig = px.histogram(
            sub, x="resid", nbins=40, color_discrete_sequence=[TEAL], title="Distribution of Residuals",
        )
        fig.update_layout(title_font_color=NAVY, xaxis_title="Residual")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Try a prediction")
    st.caption("Move the sliders to set a hypothetical player's attributes and see the predicted outcome.")
    input_vals = {}
    slider_cols = st.columns(min(len(predictors), 4))
    for i, p in enumerate(predictors):
        with slider_cols[i % len(slider_cols)]:
            lo, hi = float(df[p].min()), float(df[p].max())
            default = float(df[p].mean())
            input_vals[p] = st.slider(nice(p), lo, hi, default, key=f"lr_{p}")

    x_new = sm.add_constant(pd.DataFrame([input_vals])[predictors], has_constant="add")
    x_new = x_new[X.columns]
    pred = model.predict(x_new)[0]
    st.metric(f"Predicted {nice(outcome)}", f"{pred:,.1f}")

# ============================================================================
# SECTION 4 — LOGISTIC REGRESSION
# ============================================================================
elif section == "🎯 Logistic Regression":
    section_header("Logistic Regression", "Classify a binary outcome from several predictors.")

    col1, col2 = st.columns([1, 2])
    with col1:
        target_var = st.selectbox(
            "Variable to threshold into a binary outcome", ["overall", "potential"], index=0,
            format_func=nice,
        )
        default_thr = int(df[target_var].quantile(0.95))
        lo_bound = int(df[target_var].quantile(0.50))
        hi_bound = int(df[target_var].quantile(0.995))
        if hi_bound <= lo_bound:
            hi_bound = lo_bound + 1
        elite_threshold = st.slider(
            f"'Elite' threshold ({nice(target_var)} ≥ ...)", lo_bound, hi_bound, default_thr,
        )
    with col2:
        predictor_choices = [v for v in ATTRS]
        predictors = st.multiselect("Predictors (X)", predictor_choices, default=ATTRS, format_func=nice)

    if len(predictors) == 0:
        st.warning("Select at least one predictor — defaulting back to the standard six attributes.")
        predictors = ATTRS

    sub = df.dropna(subset=[target_var] + predictors).copy()
    sub["elite"] = (sub[target_var] >= elite_threshold).astype(int)
    elite_share = sub["elite"].mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total players", f"{len(sub):,}")
    m2.metric("'Elite' players", f"{sub['elite'].sum():,}")
    m3.metric("Elite share", f"{elite_share:.1%}")

    if sub["elite"].sum() < 10 or sub["elite"].sum() > len(sub) - 10:
        st.info("This threshold creates a fairly imbalanced class split — results below are still valid, "
                 "but keep that in mind when interpreting accuracy.")

    X = sm.add_constant(sub[predictors])
    y = sub["elite"]
    logit_model = sm.Logit(y, X).fit(disp=0)

    st.markdown("### Coefficients & odds ratios")
    logit_table = pd.DataFrame({
        "Coefficient": logit_model.params, "p-value": logit_model.pvalues,
        "Odds Ratio": np.exp(logit_model.params),
    })
    logit_table.index = ["Constant"] + [nice(p) for p in predictors]
    st.dataframe(logit_table.round(4), width="stretch")
    st.caption(f"Pseudo R² (McFadden): {logit_model.prsquared:.3f}")

    pred_prob = logit_model.predict(X)
    fpr, tpr, roc_thresholds = roc_curve(y, pred_prob)
    roc_auc = auc(fpr, tpr)

    st.markdown("### Classification threshold")
    cutoff = st.slider("Probability cutoff for predicting 'elite'", 0.05, 0.95, 0.50, 0.01)
    pred_class = (pred_prob >= cutoff).astype(int)
    cm = confusion_matrix(y, pred_class)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Accuracy", f"{accuracy:.1%}")
    cc2.metric("Precision (Elite)", f"{precision:.2f}")
    cc3.metric("Recall (Elite)", f"{recall:.2f}")
    cc4.metric("F1-score (Elite)", f"{f1:.2f}")

    d1, d2 = st.columns(2)
    with d1:
        cm_df = pd.DataFrame(
            cm, index=["Actual: Not Elite", "Actual: Elite"],
            columns=["Predicted: Not Elite", "Predicted: Elite"],
        )
        fig = px.imshow(
            cm_df, text_auto=True, color_continuous_scale="Blues",
            title="Confusion Matrix",
        )
        fig.update_layout(title_font_color=NAVY)
        st.plotly_chart(fig, width="stretch")
    with d2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc:.3f})",
                                  line=dict(color=TEAL, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                  line=dict(color=GREY, dash="dash")))
        # Mark current cutoff point on the ROC curve
        idx = (np.abs(roc_thresholds - cutoff)).argmin()
        fig.add_trace(go.Scatter(x=[fpr[idx]], y=[tpr[idx]], mode="markers",
                                  marker=dict(color=NAVY, size=11), name="Current cutoff"))
        fig.update_layout(title="ROC Curve", title_font_color=NAVY,
                           xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, width="stretch")

    if accuracy > 0.9 and recall < 0.7:
        callout(
            "Accuracy can be misleading",
            f"Accuracy looks high ({accuracy:.1%}), but recall for the elite class is only "
            f"{recall:.2f} — the model is missing a meaningful share of genuinely elite players. "
            "This is typical when the positive class is rare; precision/recall and the ROC curve "
            "tell a fuller story than accuracy alone.",
        )

    st.markdown("### Try a prediction")
    st.caption("Move the sliders to set a hypothetical player's attributes and see the predicted probability of being 'elite'.")
    input_vals = {}
    slider_cols = st.columns(min(len(predictors), 4))
    for i, p in enumerate(predictors):
        with slider_cols[i % len(slider_cols)]:
            lo, hi = float(df[p].min()), float(df[p].max())
            default = float(df[p].mean())
            input_vals[p] = st.slider(nice(p), lo, hi, default, key=f"logit_{p}")

    x_new = sm.add_constant(pd.DataFrame([input_vals])[predictors], has_constant="add")
    x_new = x_new[X.columns]
    prob = logit_model.predict(x_new)[0]
    st.metric("Predicted probability of being 'elite'", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
