# Tucson New Home Price Predictor — Streamlit App

Interactive price prediction tool built on top of the Tucson market analysis project.

## What it does

- Predicts new home sale price for any spec in the Rocking K South development area
- Shows a price waterfall chart breaking down each spec decision's dollar impact
- Displays submarket scoring and recommendations from the analysis
- Updates instantly as you adjust inputs in the sidebar

## Running locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure models/ folder is in the same directory as app.py
#    Required files:
#      models/new_home_ridge.pkl
#      models/new_home_scaler.pkl
#      models/new_home_feature_config.pkl

# 3. Run the app
streamlit run app.py
```

The app will open at http://localhost:8501

## Folder structure

```
streamlit_app/
├── app.py
├── requirements.txt
├── README_app.md
└── models/
    ├── new_home_ridge.pkl
    ├── new_home_scaler.pkl
    └── new_home_feature_config.pkl
```

## Deploying to Streamlit Community Cloud (free)

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Click "New app" → connect your GitHub repo
4. Set main file path to `app.py`
5. Click Deploy

Note: you'll need to include the `models/` folder in your GitHub repo.
If model files are large, consider using Git LFS or storing them in
a cloud bucket and loading via URL.

## Model details

| Model | R² | RMSE | Training rows |
|---|---|---|---|
| Ridge Regression (production) | 0.793 | $27,253 | 99 |
| XGBoost (comparable) | 0.784 | $27,866 | 99 |
| Random Forest | 0.703 | $32,695 | 99 |
| Linear Regression (baseline) | 0.588 | $38,492 | 99 |

Data: Rocking K South closed sales, Dec 2025 – May 2026
