# Crop Price Predictor — ML Upgrade (3 Algorithms)

## What's New
The original Linear Regression has been replaced with **3 ML algorithms**:

| Algorithm | File | Best For |
|---|---|---|
| 🌲 Random Forest | `app.py` | Robust, handles noise & outliers |
| ⚡ Gradient Boosting | `app.py` | High accuracy on tabular/time data |
| 📐 Polynomial Regression + Ridge | `app.py` | Seasonal curve fitting |

## How to Run
```bash
streamlit run app.py
```

## Files
- `app.py` — **New multi-algorithm app** (use this)
- `app_linear_original.py` — Your original linear regression app (backup)
- `dataset.csv` — Price data
- `crops.csv`, `users.csv`, `delivery.csv` — Persistent data files

## Algorithm Selection
In the sidebar, under **"ML Algorithm"**, choose:
- **Random Forest** — Ensemble of 200 decision trees
- **Gradient Boosting** — 300 boosted trees, learning rate 0.05
- **Polynomial Regression** — Degree-3 with Ridge regularization

Use the **"Compare all 3 algorithms"** checkbox to see all predictions at once.

## Dependencies
```
streamlit
pandas
numpy
scikit-learn
Pillow
```
Install with: `pip install streamlit pandas numpy scikit-learn Pillow`
