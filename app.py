# app_multi_algo.py — Crop Price Predictor with 3 ML Algorithms
# Algorithms: Random Forest, Gradient Boosting, Polynomial Regression
# Drop-in replacement for app.py — all UI/auth/delivery logic preserved.

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from io import StringIO
from PIL import Image
import hashlib
import warnings
warnings.filterwarnings("ignore")

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------------
# Theme colors (Choco-Truffle — Warm Cocoa)
# ----------------------------------------
PRIMARY = "#5D4037"
ACCENT = "#FFB74D"
BG = "#F6EEE6"
CARD_BG = "#FFFFFF"
TEXT = "#2E2E2E"

# page config
st.set_page_config(page_title="Multi-Crop Price Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
    <style>
    :root {{
        --primary: {PRIMARY};
        --accent: {ACCENT};
        --bg: {BG};
        --card: {CARD_BG};
        --text: {TEXT};
    }}
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(180deg, var(--bg) 0%, #fffaf6 100%);
        color: var(--text);
    }}
    .block-container {{
        padding: 1.25rem 2rem;
    }}
    h1, h2, h3 {{
        color: var(--primary) !important;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }}
    h1 {{ font-size: 2.1rem; }}
    h2 {{ font-size: 1.5rem; }}
    .stDownloadButton>button, .stButton>button {{
        background-color: var(--primary) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        box-shadow: none !important;
        border: none !important;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #fffaf6, var(--bg));
        border-right: 1px solid rgba(0,0,0,0.06);
    }}
    .stAlert {{
        border-left: 4px solid var(--primary) !important;
        background-color: rgba(93,64,55,0.04) !important;
    }}
    .stDataFrame table {{
        background: var(--card);
        border-radius: 8px;
        padding: 8px;
    }}
    .muted {{ color: #6f6f6f; font-size: 0.9rem; }}
    .small {{ font-size: 0.9rem; color: #6b6b6b; }}
    .accent-pill {{
        display:inline-block;
        background: var(--accent);
        color: #3a2b22;
        padding:4px 10px;
        border-radius: 999px;
        font-weight:600;
    }}
    .algo-card {{
        background: var(--card);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        border-left: 4px solid var(--accent);
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }}
    .algo-title {{
        font-weight: 700;
        font-size: 1.05rem;
        color: var(--primary);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------
# Utility: password hashing
# ----------------------------------------
def hash_password(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def verify_password(raw: str, hashed: str) -> bool:
    return hash_password(raw) == hashed

# ----------------------------------------
# Session state initialization
# ----------------------------------------
defaults = {
    "login_success": False, "role": None, "username": None,
    "name": None, "phone": None, "farmer_posts": [],
    "farmer_notifications": [], "delivery_requests": [],
    "show_signup": False, "show_delivery_form_farmer": False,
    "show_delivery_form_buyer": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------------------
# Ensure folders & files exist
# ----------------------------------------
os.makedirs("images", exist_ok=True)
for fname, cols in [
    ("crops.csv", ["farmer_id","farmer_name","location","crop_name","quantity","phone_number","image"]),
    ("users.csv", ["username","password","role","name","phone"]),
    ("delivery.csv", ["request_id","username","role","location","destination","mode","phone","timestamp"]),
]:
    if not os.path.exists(fname):
        pd.DataFrame(columns=cols).to_csv(fname, index=False)

# ----------------------------------------
# Load dataset
# ----------------------------------------
@st.cache_data
def load_dataset(path="dataset.csv"):
    if not os.path.exists(path):
        st.error(f"Dataset file not found at: {path}")
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df

df = load_dataset("dataset.csv")
if df is None:
    st.stop()

# ----------------------------------------
# ML Feature Engineering
# ----------------------------------------
def build_features(series_df):
    """Build time-series features from a sorted date/price dataframe."""
    s = series_df.sort_values("Date").copy()
    s = s.dropna(subset=["Modal Price"])
    t0 = s["Date"].min()
    s["t"] = (s["Date"] - t0).dt.days.astype(float)
    s["month"] = s["Date"].dt.month
    s["year"] = s["Date"].dt.year
    s["sin_month"] = np.sin(2 * np.pi * s["month"] / 12)
    s["cos_month"] = np.cos(2 * np.pi * s["month"] / 12)
    feature_cols = ["t", "year", "sin_month", "cos_month"]
    X = s[feature_cols].values
    y = s["Modal Price"].values
    return X, y, t0, feature_cols

def get_future_features(target_date, t0):
    td = pd.to_datetime(target_date)
    t = float((td - t0).days)
    month = td.month
    year = td.year
    sin_m = np.sin(2 * np.pi * month / 12)
    cos_m = np.cos(2 * np.pi * month / 12)
    return np.array([[t, year, sin_m, cos_m]])

def get_future_features_poly(target_date, t0):
    """Reduced feature set for polynomial regression (avoids ill-conditioning)."""
    td = pd.to_datetime(target_date)
    t = float((td - t0).days)
    month = td.month
    sin_m = np.sin(2 * np.pi * month / 12)
    cos_m = np.cos(2 * np.pi * month / 12)
    return np.array([[t, sin_m, cos_m]])

# ----------------------------------------
# Algorithm 1: Random Forest Regressor
# ----------------------------------------
@st.cache_data(show_spinner=False)
def predict_random_forest(crop_key, target_date_str, X_tuple, y_tuple, t0_str):
    X = np.array(X_tuple)
    y = np.array(y_tuple)
    t0 = pd.Timestamp(t0_str)
    target_date = pd.to_datetime(target_date_str)
    if len(y) < 5:
        return None, None, None
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X, y)
    X_future = get_future_features(target_date, t0)
    pred = model.predict(X_future)[0]
    # Cross-val approx on last 20%
    split = max(1, int(len(y) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model2 = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model2.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model2.predict(X_test)) if len(y_test) > 0 else None
    r2 = r2_score(y_test, model2.predict(X_test)) if len(y_test) > 1 else None
    return float(pred), mae, r2

# ----------------------------------------
# Algorithm 2: Gradient Boosting Regressor
# ----------------------------------------
@st.cache_data(show_spinner=False)
def predict_gradient_boosting(crop_key, target_date_str, X_tuple, y_tuple, t0_str):
    X = np.array(X_tuple)
    y = np.array(y_tuple)
    t0 = pd.Timestamp(t0_str)
    target_date = pd.to_datetime(target_date_str)
    if len(y) < 5:
        return None, None, None
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=4, subsample=0.8, random_state=42)
    model.fit(X, y)
    X_future = get_future_features(target_date, t0)
    pred = model.predict(X_future)[0]
    split = max(1, int(len(y) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model2 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                        max_depth=4, random_state=42)
    model2.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model2.predict(X_test)) if len(y_test) > 0 else None
    r2 = r2_score(y_test, model2.predict(X_test)) if len(y_test) > 1 else None
    return float(pred), mae, r2

# ----------------------------------------
# Algorithm 3: Polynomial Regression (degree 3) + Ridge
# ----------------------------------------
@st.cache_data(show_spinner=False)
def predict_polynomial(crop_key, target_date_str, X_tuple, y_tuple, t0_str):
    X_all = np.array(X_tuple)
    y = np.array(y_tuple)
    t0 = pd.Timestamp(t0_str)
    target_date = pd.to_datetime(target_date_str)
    if len(y) < 4:
        return None, None, None
    # Use only t, sin_month, cos_month (columns 0, 2, 3) to avoid ill-conditioning
    X = X_all[:, [0, 2, 3]]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("ridge", Ridge(alpha=100.0))
    ])
    model.fit(X, y)
    X_future = get_future_features_poly(target_date, t0)
    pred = model.predict(X_future)[0]
    split = max(1, int(len(y) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model2 = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("ridge", Ridge(alpha=100.0))
    ])
    model2.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model2.predict(X_test)) if len(y_test) > 0 else None
    r2 = r2_score(y_test, model2.predict(X_test)) if len(y_test) > 1 else None
    return float(pred), mae, r2

# ----------------------------------------
# Existing crop post / delivery / user helpers (unchanged)
# ----------------------------------------
def load_farmer_posts():
    try:
        if not os.path.exists("crops.csv") or os.stat("crops.csv").st_size == 0:
            return []
        df_p = pd.read_csv("crops.csv")
        return [] if df_p.empty else df_p.fillna("").to_dict("records")
    except pd.errors.EmptyDataError:
        return []

def save_farmer_posts(posts):
    cleaned = []
    for p in posts:
        p2 = p.copy()
        if "image" not in p2 or p2["image"] is None or (isinstance(p2["image"], float) and np.isnan(p2["image"])):
            p2["image"] = ""
        cleaned.append(p2)
    cols = ["farmer_id","farmer_name","location","crop_name","quantity","phone_number","image"]
    if not cleaned:
        pd.DataFrame(columns=cols).to_csv("crops.csv", index=False)
        return
    pd.DataFrame(cleaned).to_csv("crops.csv", index=False)

DELIVERY_CSV = "delivery.csv"

def load_delivery_requests():
    try:
        if not os.path.exists(DELIVERY_CSV) or os.stat(DELIVERY_CSV).st_size == 0:
            return []
        df_dr = pd.read_csv(DELIVERY_CSV)
        return [] if df_dr.empty else df_dr.fillna("").to_dict("records")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return []

def save_delivery_requests(requests):
    cols = ["request_id","username","role","location","destination","mode","phone","timestamp"]
    cleaned = [{k: r.get(k, "") for k in cols} for r in requests]
    if not cleaned:
        pd.DataFrame(columns=cols).to_csv(DELIVERY_CSV, index=False)
        return
    pd.DataFrame(cleaned).to_csv(DELIVERY_CSV, index=False)

USERS_CSV = "users.csv"

def load_users():
    cols = ["username","password","role","name","phone"]
    try:
        if not os.path.exists(USERS_CSV) or os.stat(USERS_CSV).st_size == 0:
            df_e = pd.DataFrame(columns=cols); df_e.to_csv(USERS_CSV, index=False); return df_e
        df = pd.read_csv(USERS_CSV)
        for c in cols:
            if c not in df.columns: df[c] = ""
        return df[cols].fillna("")
    except pd.errors.EmptyDataError:
        df_e = pd.DataFrame(columns=cols); df_e.to_csv(USERS_CSV, index=False); return df_e

def save_user(username, password, role, name, phone):
    df = load_users()
    if username in df["username"].values:
        return False
    new_row = {"username": username, "password": hash_password(password),
               "role": role, "name": name, "phone": phone}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(USERS_CSV, index=False)
    return True

def authenticate(username, raw_password):
    df = load_users()
    user = df[df["username"] == username]
    if user.empty:
        return None
    if verify_password(raw_password, str(user.iloc[0]["password"])):
        return {k: user.iloc[0][k] for k in ["username","role","name","phone"]}
    return None

if not st.session_state["farmer_posts"]:
    st.session_state["farmer_posts"] = load_farmer_posts()
if not st.session_state["delivery_requests"]:
    st.session_state["delivery_requests"] = load_delivery_requests()

# ----------------------------------------
# Auth UI
# ----------------------------------------
def signup_page():
    st.markdown("""
    <h1 style='text-align:center; font-weight:bold; color:#1B5E20; padding:12px;'>
        Welcome to Agrolytics🌿
    </h1>
    <p style='text-align:center; font-size:0.9rem; color:#555;'>Where Innovation Meets Agriculture☘️</p>
    <h2 style='text-align:center; margin-bottom:6px;'>Create New Account</h2>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Choose Username", key="su_username")
        password = st.text_input("Choose Password", type="password", key="su_password")
    with col2:
        name = st.text_input("Full Name", key="su_name")
        phone = st.text_input("Phone Number", key="su_phone")
    role = st.selectbox("Select Role", ["farmer","buyer","delivery"], key="su_role")
    if st.button("Create Account", key="su_create"):
        if not all([username, password, name, phone]):
            st.error("All fields are required.")
            return
        ok = save_user(username.strip(), password, role, name.strip(), phone.strip())
        if ok:
            st.success("Account created! You can now log in.")
        else:
            st.error("Username already exists!")
def login_page():
    st.markdown("""
    <h1 style='text-align:center; font-weight:bold; color:#1B5E20; padding:12px;'>
        Welcome to Agrolytics🌿
    </h1>
    <p style='text-align:center; font-size:0.9rem; color:#555;'>Where Innovation Meets Agriculture☘️</p>
    <h2 style='text-align:center; margin-bottom:6px;'>Login</h2>
    """, unsafe_allow_html=True)
    username = st.text_input("Username", key="li_username")
    password = st.text_input("Password", type="password", key="li_password")
    if st.button("Login", key="li_login"):
        user = authenticate(username.strip(), password)
        if user:
            for k, v in user.items():
                st.session_state[k] = v
            st.session_state["login_success"] = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

if not st.session_state["login_success"]:
    left, right = st.columns([2, 1])
    with left:
        if st.session_state["show_signup"]:
            signup_page()
            if st.button("Back to Login", key="su_back"):
                st.session_state["show_signup"] = False; st.rerun()
        else:
            login_page()
            if st.button("New user? Sign up", key="li_new"):
                st.session_state["show_signup"] = True; st.rerun()
    with right:
        st.markdown(f"<div style='background:rgba(93,64,55,0.06); padding:12px; border-radius:8px;'>"
                    f"<h3 style='color:{PRIMARY}; margin:0 0 6px 0'>Quick tips</h3>"
                    "<ul style='margin:0; padding-left:18px; color:#4b4b4b;'>"
                    "<li>Roles: farmer, buyer, delivery</li>"
                    "<li>Signup then login.</li></ul></div>", unsafe_allow_html=True)
    st.stop()

def logout():
    if st.button("Logout", key="logout_btn"):
        for k in ["login_success","role","username","name","phone"]:
            st.session_state[k] = False if k == "login_success" else None
        st.rerun()

col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown("<h1 style='text-align:center;margin:6px 0 8px 0;'>Multi-Crop Price Predictor</h1>", unsafe_allow_html=True)
with col_h2:
    st.markdown(f"<div style='text-align:right; font-size:0.95rem; color:#6b6b6b;'>Logged in as<br><strong>{st.session_state.get('username') or '—'}</strong></div>", unsafe_allow_html=True)
    logout()

role_selection = st.session_state["role"]

# ----------------------------------------
# Filters for Farmer & Buyer
# ----------------------------------------
if role_selection in ["farmer", "buyer"]:
    required_cols = ["State","District","Crop","Modal Price","Date"]
    if not all(c in df.columns for c in required_cols):
        st.error("dataset.csv is missing required columns."); st.stop()

    st.sidebar.title("Filters & Prediction")
    state_list = ["All"] + sorted(df["State"].dropna().unique().tolist())
    state = st.sidebar.selectbox("State", state_list, index=0)
    temp_df = df[df["State"] == state] if state != "All" else df
    dist_list = ["All"] + sorted(temp_df["District"].dropna().unique().tolist())
    district = st.sidebar.selectbox("District", dist_list, index=0)
    temp_df2 = temp_df[temp_df["District"] == district] if district != "All" else temp_df
    crop_list = ["Select crop"] + sorted(temp_df2["Crop"].dropna().unique().tolist())
    crop = st.sidebar.selectbox("Crop", crop_list, index=0)
    future_date = st.sidebar.date_input("Future Date", value=pd.Timestamp.now().date())

    # Algorithm selector in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 ML Algorithm")
    algo_choice = st.sidebar.radio(
        "Select prediction algorithm:",
        ["🌲 Random Forest", "⚡ Gradient Boosting", "📐 Polynomial Regression"],
        index=0
    )
    st.sidebar.markdown("""
    <div style='font-size:0.8rem; color:#666; margin-top:4px;'>
    <b>Random Forest</b>: Ensemble of decision trees, robust to noise.<br><br>
    <b>Gradient Boosting</b>: Sequential boosting, high accuracy on tabular data.<br><br>
    <b>Polynomial Regression</b>: Captures non-linear seasonal trends.
    </div>
    """, unsafe_allow_html=True)

    # Filter data
    filtered = df.copy()
    if state != "All":
        filtered = filtered[filtered["State"] == state]
    if district != "All":
        filtered = filtered[filtered["District"] == district]
    if crop == "Select crop":
        st.info("Please select a Crop from the sidebar to enable prediction.")
        st.stop()
    filtered = filtered[filtered["Crop"] == crop]
    hist = filtered.sort_values("Date")
    if hist.empty:
        st.warning("No historical records found for this selection."); st.stop()

    # Show historical data
    st.markdown(f"<h2>{crop} — Historical Prices in {district if district != 'All' else 'All Districts'}</h2>", unsafe_allow_html=True)
    st.dataframe(hist.head(10))
    st.line_chart(hist.set_index("Date")["Modal Price"])

    # Build features
    X, y, t0, _ = build_features(hist)

    # Cache key for algo caching (crop+state+district)
    crop_key = f"{state}_{district}_{crop}"
    target_date_str = str(future_date)
    X_tuple = tuple(map(tuple, X))
    y_tuple = tuple(y)
    t0_str = str(t0)

    # Run selected algorithm
    with st.spinner("Training model and predicting..."):
        if "Random Forest" in algo_choice:
            pred_price, mae, r2 = predict_random_forest(crop_key, target_date_str, X_tuple, y_tuple, t0_str)
            algo_name = "Random Forest Regressor"
            algo_desc = "Ensemble of 200 decision trees with time, year, and seasonal features."
        elif "Gradient Boosting" in algo_choice:
            pred_price, mae, r2 = predict_gradient_boosting(crop_key, target_date_str, X_tuple, y_tuple, t0_str)
            algo_name = "Gradient Boosting Regressor"
            algo_desc = "300 boosted trees with learning rate 0.05. Best for capturing complex price patterns."
        else:
            pred_price, mae, r2 = predict_polynomial(crop_key, target_date_str, X_tuple, y_tuple, t0_str)
            algo_name = "Polynomial Regression (degree 3) + Ridge"
            algo_desc = "Degree-3 polynomial features with Ridge regularization for seasonal curve fitting."

    if pred_price is None or np.isnan(pred_price):
        st.error("Not enough data to train the model (need at least 5 data points)."); st.stop()

    # Recommendation
    def get_recommendation(last_price, predicted_price, role):
        perc = ((predicted_price - last_price) / last_price) * 100
        if perc > 3:
            rec = "HOLD / WAIT" if role in ["farmer","buyer"] else f"Monitor (+{perc:.2f}%)"
        elif perc < -3:
            rec = "SELL / ACT NOW" if role == "farmer" else "BUY / ACT NOW" if role == "buyer" else f"Monitor ({perc:.2f}%)"
        else:
            rec = f"Monitor (small change {perc:.2f}%)"
        return rec, perc

    last_known_price = hist["Modal Price"].iloc[-1]
    recommendation, perc_change = get_recommendation(last_known_price, pred_price, role_selection)

    # ----------------------------------------
    # Farmer Dashboard
    # ----------------------------------------
    if role_selection == "farmer":
        st.markdown("<h1 style='text-align:center; margin:6px 0 8px 0;'>🌾 Farmer Dashboard</h1>", unsafe_allow_html=True)
    
        # Prediction results
        st.markdown("<div style='background:#f0e6d2; padding:16px; border-radius:10px;'>", unsafe_allow_html=True)
        
        # Algorithm info card
        st.markdown(f"""
        <div class='algo-card'>
            <div class='algo-title'>🤖 {algo_name}</div>
            <div class='small'>{algo_desc}</div>
        </div>
        """, unsafe_allow_html=True)
    
        st.success(f"Predicted **{crop}** price on **{future_date}**: ₹{pred_price:,.2f}")
        st.info(f"Recommendation: **{recommendation}**")
        
        # Model metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Price", f"₹{pred_price:,.2f}")
        with col2:
            st.metric("Price Change", f"{perc_change:+.2f}%")
        with col3:
            if mae is not None:
                st.metric("Model MAE (test set)", f"₹{mae:,.2f}")
        
        if r2 is not None:
            st.caption(f"Model R² Score on test data: {r2:.3f} | Algorithm: {algo_name}")
    
        # Compare all 3 algorithms
        if st.checkbox("📊 Compare all 3 algorithms side by side"):
            with st.spinner("Running all 3 models..."):
                rf_p, rf_mae, rf_r2 = predict_random_forest(crop_key, target_date_str, X_tuple, y_tuple, t0_str)
                gb_p, gb_mae, gb_r2 = predict_gradient_boosting(crop_key, target_date_str, X_tuple, y_tuple, t0_str)
                poly_p, poly_mae, poly_r2 = predict_polynomial(crop_key, target_date_str, X_tuple, y_tuple, t0_str)
    
            comparison = pd.DataFrame([
                {"Algorithm": "🌲 Random Forest", "Predicted Price (₹)": f"{rf_p:,.2f}" if rf_p else "N/A",
                 "Test MAE (₹)": f"{rf_mae:,.2f}" if rf_mae else "N/A", "Test R²": f"{rf_r2:.3f}" if rf_r2 else "N/A"},
                {"Algorithm": "⚡ Gradient Boosting", "Predicted Price (₹)": f"{gb_p:,.2f}" if gb_p else "N/A",
                 "Test MAE (₹)": f"{gb_mae:,.2f}" if gb_mae else "N/A", "Test R²": f"{gb_r2:.3f}" if gb_r2 else "N/A"},
                {"Algorithm": "📐 Polynomial Regression", "Predicted Price (₹)": f"{poly_p:,.2f}" if poly_p else "N/A",
                 "Test MAE (₹)": f"{poly_mae:,.2f}" if poly_mae else "N/A", "Test R²": f"{poly_r2:.3f}" if poly_r2 else "N/A"},
            ])
            st.dataframe(comparison, use_container_width=True)
            st.caption("Lower MAE = better accuracy | Higher R² = better fit (max 1.0)")
    
        out_df = pd.DataFrame([{
            "State": state, "District": district, "Crop": crop,
            "date_of_prediction": future_date, "predicted_price": pred_price,
            "algorithm": algo_name, "model_mae": mae, "model_r2": r2
        }])
        st.download_button("Download prediction CSV",
                           StringIO(out_df.to_csv(index=False)).getvalue(),
                           file_name=f"{crop}_prediction_{future_date}.csv", mime="text/csv")
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Post Crop
        st.markdown("<div style='background:#fff2e6; padding:12px; border-radius:8px; margin-top:16px;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:#5D4037;'>Post your Crop for Sale</h3>", unsafe_allow_html=True)
        crop_name = st.text_input("Crop Name", key="post_crop_name")
        quantity = st.number_input("Quantity (kg)", min_value=1, key="post_quantity")
        phone_number = st.text_input("Phone Number", value=st.session_state.get("phone",""), key="post_phone")
        crop_image = st.file_uploader("Upload Crop Image (Optional)", type=["jpg","jpeg","png"], key="post_image")
        if st.button("Post Crop", key="post_button"):
            if crop_name and quantity and phone_number:
                image_path = ""
                if crop_image:
                    ts = int(datetime.now().timestamp())
                    fname = f"{st.session_state['username']}_{crop_name}_{quantity}_{ts}.jpg"
                    image_path = os.path.join("images", fname)
                    with open(image_path, "wb") as f:
                        f.write(crop_image.getbuffer())
                post_data = {
                    "farmer_id": st.session_state["username"],
                    "farmer_name": st.session_state.get("name","Farmer"),
                    "location": f"{state}, {district}",
                    "crop_name": crop_name, "quantity": quantity,
                    "phone_number": phone_number, "image": image_path
                }
                st.session_state["farmer_posts"].append(post_data)
                save_farmer_posts(st.session_state["farmer_posts"])
                st.success(f"{crop_name} posted successfully!")
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Your posts
        st.markdown("<div style='background:#e6f7ff; padding:12px; border-radius:8px; margin-top:16px;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:#5D4037;'>Your Posted Crops</h3>", unsafe_allow_html=True)
        st.session_state["farmer_posts"] = load_farmer_posts()
        own_posts = [p for p in st.session_state["farmer_posts"] if p.get("farmer_id") == st.session_state["username"]]
        if own_posts:
            for idx, post in enumerate(own_posts):
                st.markdown("<div style='background:var(--card); padding:10px; border-radius:8px; margin-bottom:8px;'>", unsafe_allow_html=True)
                st.write(f"**Crop:** {post.get('crop_name','')} | **Qty:** {post.get('quantity','')} kg | **Phone:** {post.get('phone_number','N/A')} | **Loc:** {post.get('location','')}")
                img = post.get("image","")
                if isinstance(img, str) and img and os.path.exists(img):
                try:
                    st.image(img, width=200)
                except Exception:
                    st.write("_Image not available_")
                if st.button("Remove Post", key=f"remove_post_{idx}"):
                    st.session_state["farmer_posts"].remove(post)
                    save_farmer_posts(st.session_state["farmer_posts"])
                    st.success("Post removed."); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("You have not posted any crops yet.")
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Delivery
        st.markdown("<div style='background:#f9e6ff; padding:12px; border-radius:8px; margin-top:16px;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center; color:#3a2b22;'>Delivery Made Easy</h2>", unsafe_allow_html=True)
        if st.button("Need Delivery", key="need_delivery_farmer"):
            st.session_state["show_delivery_form_farmer"] = True
        if st.session_state["show_delivery_form_farmer"]:
            with st.form("delivery_form_farmer", clear_on_submit=False):
                loc = st.text_input("Your location", key="df_loc")
                dest = st.text_input("Destination", key="df_dest")
                mode = st.selectbox("Mode of transport", ["Bike","Auto","Tractor","Tempo","Lorry"], key="df_mode")
                phone = st.text_input("Phone number", value=st.session_state.get("phone",""), key="df_phone")
                submitted = st.form_submit_button("Submit Delivery Request")
                cancel = st.form_submit_button("Cancel")
                if submitted:
                    req = {
                        "request_id": f"{st.session_state['username']}_{int(datetime.now().timestamp())}",
                        "username": st.session_state["username"], "role": st.session_state["role"],
                        "location": loc, "destination": dest, "mode": mode,
                        "phone": phone, "timestamp": datetime.now().isoformat()
                    }
                    cur = load_delivery_requests(); cur.append(req)
                    st.session_state["delivery_requests"] = cur
                    save_delivery_requests(cur)
                    st.success("Delivery request submitted.")
                    st.session_state["show_delivery_form_farmer"] = False; st.rerun()
                if cancel:
                    st.session_state["show_delivery_form_farmer"] = False; st.rerun()
        st.markdown("<h3 style='text-align:center; color:#5D4037;'>Your delivery requests</h3>", unsafe_allow_html=True)
        st.session_state["delivery_requests"] = load_delivery_requests()
        my_reqs = [r for r in st.session_state["delivery_requests"] if r.get("username") == st.session_state["username"]]
        for r in my_reqs:
            st.markdown("<div style='background:var(--card); padding:10px; border-radius:8px; margin-bottom:8px;'>", unsafe_allow_html=True)
            st.write(f"**{r.get('request_id')}** | {r.get('location')} → {r.get('destination')} | {r.get('mode')} | {r.get('phone')}")
            if st.button("Remove Request", key=f"remove_del_{r.get('request_id')}"):
                cur = [x for x in load_delivery_requests() if x.get("request_id") != r.get("request_id")]
                st.session_state["delivery_requests"] = cur; save_delivery_requests(cur)
                st.success("Removed."); st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        if not my_reqs:
            st.info("No delivery requests yet.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ----------------------------------------
    # Buyer Dashboard
    # ----------------------------------------
    elif role_selection == "buyer":
        st.markdown("<h1 style='text-align:center; margin:6px 0 8px 0;'>🛒 Buyer Dashboard</h1>", unsafe_allow_html=True)
    
        st.markdown("<div style='background:#fff4e6; padding:16px; border-radius:10px;'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='algo-card'>
            <div class='algo-title'>🤖 {algo_name}</div>
            <div class='small'>{algo_desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.success(f"Predicted **{crop}** price on **{future_date}**: ₹{pred_price:,.2f}")
        st.info(f"Recommendation: **{recommendation}**")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Predicted Price", f"₹{pred_price:,.2f}")
        with col2: st.metric("Price Change", f"{perc_change:+.2f}%")
        with col3:
            if mae is not None: st.metric("Model MAE", f"₹{mae:,.2f}")
        if r2 is not None:
            st.caption(f"R² Score: {r2:.3f} | Algorithm: {algo_name}")
    
        if st.checkbox("📊 Compare all 3 algorithms"):
            with st.spinner("Running all 3 models..."):
                rf_p, rf_mae, rf_r2 = predict_random_forest(crop_key, target_date_str, X_tuple, y_tuple, t0_str)
                gb_p, gb_mae, gb_r2 = predict_gradient_boosting(crop_key, target_date_str, X_tuple, y_tuple, t0_str)
                poly_p, poly_mae, poly_r2 = predict_polynomial(crop_key, target_date_str, X_tuple, y_tuple, t0_str)
            comparison = pd.DataFrame([
                {"Algorithm": "🌲 Random Forest", "Predicted Price (₹)": f"{rf_p:,.2f}" if rf_p else "N/A",
                 "Test MAE (₹)": f"{rf_mae:,.2f}" if rf_mae else "N/A", "Test R²": f"{rf_r2:.3f}" if rf_r2 else "N/A"},
                {"Algorithm": "⚡ Gradient Boosting", "Predicted Price (₹)": f"{gb_p:,.2f}" if gb_p else "N/A",
                 "Test MAE (₹)": f"{gb_mae:,.2f}" if gb_mae else "N/A", "Test R²": f"{gb_r2:.3f}" if gb_r2 else "N/A"},
                {"Algorithm": "📐 Polynomial Regression", "Predicted Price (₹)": f"{poly_p:,.2f}" if poly_p else "N/A",
                 "Test MAE (₹)": f"{poly_mae:,.2f}" if poly_mae else "N/A", "Test R²": f"{poly_r2:.3f}" if poly_r2 else "N/A"},
            ])
            st.dataframe(comparison, use_container_width=True)
    
        out_df = pd.DataFrame([{
            "State": state, "District": district, "Crop": crop,
            "date_of_prediction": future_date, "predicted_price": pred_price,
            "algorithm": algo_name
        }])
        st.download_button("Download prediction CSV",
                           StringIO(out_df.to_csv(index=False)).getvalue(),
                           file_name=f"{crop}_prediction_{future_date}.csv", mime="text/csv")
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Available crops
        st.markdown("<div style='background:#f0f0ff; padding:12px; border-radius:8px; margin-top:16px;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:#5D4037;'>Available Crops for Purchase</h3>", unsafe_allow_html=True)
        st.session_state["farmer_posts"] = load_farmer_posts()
        if st.session_state["farmer_posts"]:
            for post in st.session_state["farmer_posts"]:
                st.markdown("<div style='background:var(--card); padding:10px; border-radius:8px; margin-bottom:8px;'>", unsafe_allow_html=True)
                st.write(f"**Crop:** {post.get('crop_name','')} | **Qty:** {post.get('quantity','')} kg | **Location:** {post.get('location','')} | **Phone:** {post.get('phone_number','N/A')}")
                img = post.get("image","")
                if isinstance(img, str) and img and os.path.exists(img):
                try:
                    st.image(img, width=200)
                except Exception:
                    st.write("_Image not available_")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No crops available currently.")
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Delivery
        st.markdown("<div style='background:#f9e6ff; padding:12px; border-radius:8px; margin-top:16px;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center; color:#3a2b22;'>Delivery Made Easy</h2>", unsafe_allow_html=True)
        if st.button("Need Delivery", key="need_delivery_buyer"):
            st.session_state["show_delivery_form_buyer"] = True
        if st.session_state["show_delivery_form_buyer"]:
            with st.form("delivery_form_buyer", clear_on_submit=False):
                loc = st.text_input("Your location", key="db_loc")
                dest = st.text_input("Destination", key="db_dest")
                mode = st.selectbox("Mode of transport", ["Bike","Auto","Tractor","Tempo","Lorry"], key="db_mode")
                phone = st.text_input("Phone number", value=st.session_state.get("phone",""), key="db_phone")
                submitted = st.form_submit_button("Submit Delivery Request")
                cancel = st.form_submit_button("Cancel")
                if submitted:
                    req = {
                        "request_id": f"{st.session_state['username']}_{int(datetime.now().timestamp())}",
                        "username": st.session_state["username"], "role": st.session_state["role"],
                        "location": loc, "destination": dest, "mode": mode,
                        "phone": phone, "timestamp": datetime.now().isoformat()
                    }
                    cur = load_delivery_requests(); cur.append(req)
                    st.session_state["delivery_requests"] = cur; save_delivery_requests(cur)
                    st.success("Delivery request submitted.")
                    st.session_state["show_delivery_form_buyer"] = False; st.rerun()
                if cancel:
                    st.session_state["show_delivery_form_buyer"] = False; st.rerun()
        my_reqs = [r for r in load_delivery_requests() if r.get("username") == st.session_state["username"]]
        for r in my_reqs:
            st.markdown("<div style='background:var(--card); padding:10px; border-radius:8px; margin-bottom:8px;'>", unsafe_allow_html=True)
            st.write(f"**{r.get('request_id')}** | {r.get('location')} → {r.get('destination')} | {r.get('mode')}")
            if st.button("Remove Request", key=f"remove_del_buyer_{r.get('request_id')}"):
                cur = [x for x in load_delivery_requests() if x.get("request_id") != r.get("request_id")]
                st.session_state["delivery_requests"] = cur; save_delivery_requests(cur)
                st.success("Removed."); st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        if not my_reqs:
            st.info("No delivery requests yet.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ----------------------------------------
# Delivery Dashboard
# ----------------------------------------
elif role_selection == "delivery":
    st.markdown("<h1 style='text-align:center; margin:6px 0 8px 0;'>🚚 Delivery Dashboard</h1>", unsafe_allow_html=True)
    st.info("All delivery requests (farmers & buyers)")
    st.session_state["delivery_requests"] = load_delivery_requests()
    if st.session_state["delivery_requests"]:
        for r in st.session_state["delivery_requests"]:
            st.markdown("<div style='background:var(--card); padding:10px; border-radius:8px; margin-bottom:8px;'>", unsafe_allow_html=True)
            st.write(f"**{r.get('request_id')}** | User: {r.get('username')} ({r.get('role')})")
            st.write(f"From: {r.get('location')} → To: {r.get('destination')} | Mode: {r.get('mode')} | Phone: {r.get('phone')}")
            st.write(f"Time: {r.get('timestamp')}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No delivery requests yet.")
