# FIFA 26 Player Ratings — Interactive Analytics Study Guide

A Streamlit app companion to the FIFA 26 Worked Example study guide, covering Descriptive
Statistics, One-Way ANOVA, Multiple Linear Regression, and Logistic Regression on the EA Sports
FC 26 (FIFA 26) player ratings dataset (16,343 outfield players).

## Files

- `app.py` — the Streamlit app
- `fifa26_player_data.csv` — cleaned, trimmed dataset bundled with the app (outfield players only,
  with `position_group` and `age_group` already derived). If this file is missing, the app
  automatically downloads and cleans the raw dataset from GitHub instead.
- `requirements.txt` — Python dependencies

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploying for students (Streamlit Community Cloud — free)

1. Create a new GitHub repository and push these three files to it (keep `app.py` and
   `fifa26_player_data.csv` in the same folder).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click
   "New app."
3. Point it at your repository, branch, and `app.py` as the entry point.
4. Click "Deploy." You'll get a shareable URL (e.g.
   `https://yourname-fifa26-guide.streamlit.app`) you can post on your LMS or send to students
   directly — no installation needed on their end.

## What's in the app

**Sidebar (applies across all sections):** filter by position group, age range, and league —
useful for showing students how results can shift across subgroups.

**📊 Descriptives:** single-variable histograms with summary stats, a configurable correlation
heatmap, an X-vs-Y scatter plot with optional OLS trendline, and a filtered-data download button.

**📈 ANOVA:** choose any numeric outcome and any of four grouping variables (position group,
preferred foot, age group, work rate). Shows group descriptives, a boxplot, the ANOVA table,
F-statistic, p-value, eta-squared (effect size) with a plain-language interpretation, an adjustable
significance level, and Tukey HSD post-hoc comparisons when there are more than two groups.

**📉 Linear Regression:** pick any outcome and any set of predictors. Shows R²/adjusted R², the
coefficient table, VIF multicollinearity diagnostics, residual diagnostic plots, and a "Try a
prediction" panel with live sliders that compute a predicted value for a hypothetical player.

**🎯 Logistic Regression:** set your own "elite" threshold and predictor set. Shows odds ratios,
an adjustable classification cutoff with a live-updating confusion matrix and accuracy/precision/
recall/F1, an ROC curve with AUC and a marker showing the current cutoff, and the same kind of
live "Try a prediction" panel reporting a predicted probability.

## Notes for class use

- All defaults mirror the worked examples in the companion PDF/notebook (six attribute predictors,
  Overall as the outcome, position group for ANOVA, Overall ≥ 80 for "elite"), so you can walk
  through the static study guide first and then let students explore live in the app.
- Because everything is interactive, this doubles as a sandbox for the "Try It Yourself" exercises
  from the study guide — e.g. dropping a predictor to watch VIF change, or tightening the elite
  threshold to watch precision/recall trade off.
