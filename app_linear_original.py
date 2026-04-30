# app.py — Full single-file Streamlit app with "Warm Cocoa" (Choco-Truffle) UI theme (Option B)
# NOTE: I did NOT change any of your app logic — only added visual improvements (CSS + small layout tweaks).
# Everything else (CSV handling, auth, posting, delivery logic) is preserved from your working version.

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from io import StringIO
from PIL import Image
import hashlib

# ----------------------------------------
# Theme colors (Choco-Truffle — Warm Cocoa)
# ----------------------------------------
PRIMARY = "#5D4037"      # Warm Cocoa (main)
ACCENT = "#FFB74D"       # Caramel Gold (accent)
BG = "#F6EEE6"           # Soft warm background
CARD_BG = "#FFFFFF"      # Card background (white for contrast)
TEXT = "#2E2E2E"         # Main text color

# page config
st.set_page_config(page_title="Multi-Crop Price Predictor", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------
# Inject CSS for UI theming (won't change logic)
# ----------------------------------------
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
    /* App background */
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(180deg, var(--bg) 0%, #fffaf6 100%);
        color: var(--text);
    }}
    /* Main content container (card-like) */
    .block-container {{
        padding: 1.25rem 2rem;
    }}
    /* Headers */
    h1, h2, h3 {{
        color: var(--primary) !important;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }}
    h1 {{
        font-size: 2.1rem;
    }}
    h2 {{
        font-size: 1.5rem;
    }}
    /* Cards (download/prediction areas) */
    .stDownloadButton>button, .stButton>button {{
        background-color: var(--primary) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        box-shadow: none !important;
        border: none !important;
    }}
    /* Secondary buttons (danger/neutral) */
    .remove-btn > button {{
        background-color: #A14D3A !important;
        color: white !important;
        border-radius: 8px !important;
    }}
    /* Sidebar tweaks */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #fffaf6, var(--bg));
        border-right: 1px solid rgba(0,0,0,0.06);
    }}
    .css-1d391kg, .css-1v3fvcr {{ /* Some Streamlit auto class names differ; this is harmless */
        color: var(--text);
    }}
    /* Info / success boxes */
    .stAlert {{
        border-left: 4px solid var(--primary) !important;
        background-color: rgba(93,64,55,0.04) !important;
    }}
    /* Make tables look cleaner */
    .stDataFrame table {{
        background: var(--card);
        border-radius: 8px;
        padding: 8px;
    }}
    /* Small helpers */
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------
# Utility: password hashing (sha256)
# ----------------------------------------
def hash_password(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def verify_password(raw: str, hashed: str) -> bool:
    return hash_password(raw) == hashed

# ----------------------------------------
# Session state initialization
# ----------------------------------------
if "login_success" not in st.session_state:
    st.session_state["login_success"] = False
if "role" not in st.session_state:
    st.session_state["role"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "name" not in st.session_state:
    st.session_state["name"] = None
if "phone" not in st.session_state:
    st.session_state["phone"] = None
if "farmer_posts" not in st.session_state:
    st.session_state["farmer_posts"] = []
if "farmer_notifications" not in st.session_state:
    st.session_state["farmer_notifications"] = []
if "delivery_requests" not in st.session_state:
    st.session_state["delivery_requests"] = []
if "show_signup" not in st.session_state:
    st.session_state["show_signup"] = False
# separate toggles for farmer/buyer delivery forms so they don't collide
if "show_delivery_form_farmer" not in st.session_state:
    st.session_state["show_delivery_form_farmer"] = False
if "show_delivery_form_buyer" not in st.session_state:
    st.session_state["show_delivery_form_buyer"] = False

# ----------------------------------------
# Ensure folders & files exist
# ----------------------------------------
os.makedirs("images", exist_ok=True)
if not os.path.exists("crops.csv"):
    pd.DataFrame(columns=[
        "farmer_id", "farmer_name", "location",
        "crop_name", "quantity", "phone_number", "image"
    ]).to_csv("crops.csv", index=False)
if not os.path.exists("users.csv"):
    pd.DataFrame(columns=["username", "password", "role", "name", "phone"]).to_csv("users.csv", index=False)
# delivery.csv: ensure header if missing
if not os.path.exists("delivery.csv"):
    pd.DataFrame(columns=[
        "request_id", "username", "role",
        "location", "destination", "mode",
        "phone", "timestamp"
    ]).to_csv("delivery.csv", index=False)

# ----------------------------------------
# Load dataset (predictions)
# ----------------------------------------
@st.cache_data
def load_dataset(path="dataset.csv"):
    if not os.path.exists(path):
        st.error(f"Dataset file not found at: {path}")
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_dataset("dataset.csv")
if df is None:
    st.stop()

# ----------------------------------------
# Crop posts persistence (safe)
# ----------------------------------------
def load_farmer_posts():
    try:
        if not os.path.exists("crops.csv"):
            return []
        if os.stat("crops.csv").st_size == 0:
            return []
        df_posts = pd.read_csv("crops.csv")
        if df_posts.empty:
            return []
        # normalize empty values to empty string to avoid floats/nan
        df_posts = df_posts.fillna("")
        return df_posts.to_dict("records")
    except pd.errors.EmptyDataError:
        return []

def save_farmer_posts(posts):
    # normalize posts so image is always string ("" when missing)
    cleaned = []
    for p in posts:
        p2 = p.copy()
        if "image" not in p2 or p2["image"] is None or (isinstance(p2["image"], float) and np.isnan(p2["image"])):
            p2["image"] = ""
        cleaned.append(p2)
    if not cleaned:
        df_empty = pd.DataFrame(columns=[
            "farmer_id", "farmer_name", "location",
            "crop_name", "quantity", "phone_number", "image"
        ])
        df_empty.to_csv("crops.csv", index=False)
        return
    df_posts = pd.DataFrame(cleaned)
    df_posts.to_csv("crops.csv", index=False)

# Load persistent posts into session state (only if not already loaded)
if not st.session_state["farmer_posts"]:
    st.session_state["farmer_posts"] = load_farmer_posts()

# ----------------------------------------
# Delivery requests persistence (safe)
# ----------------------------------------
DELIVERY_CSV = "delivery.csv"

def load_delivery_requests():
    try:
        if not os.path.exists(DELIVERY_CSV):
            return []
        if os.stat(DELIVERY_CSV).st_size == 0:
            return []
        df_dr = pd.read_csv(DELIVERY_CSV)
        if df_dr.empty:
            return []
        df_dr = df_dr.fillna("")  # avoid NaN floats in fields
        return df_dr.to_dict("records")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pd.DataFrame(columns=[
            "request_id", "username", "role",
            "location", "destination", "mode",
            "phone", "timestamp"
        ]).to_csv(DELIVERY_CSV, index=False)
        return []

def save_delivery_requests(requests):
    # normalize
    cleaned = []
    for r in requests:
        r2 = r.copy()
        for k in ["location","destination","mode","phone","timestamp","username","role","request_id"]:
            if k not in r2 or r2[k] is None:
                r2[k] = ""
        cleaned.append(r2)
    if not cleaned:
        df_empty = pd.DataFrame(columns=[
            "request_id", "username", "role",
            "location", "destination", "mode",
            "phone", "timestamp"
        ])
        df_empty.to_csv(DELIVERY_CSV, index=False)
        return
    df = pd.DataFrame(cleaned)
    df.to_csv(DELIVERY_CSV, index=False)

# Load into session state if not present
if not st.session_state["delivery_requests"]:
    st.session_state["delivery_requests"] = load_delivery_requests()

# ----------------------------------------
# Users persistence (signup/login) (safe)
# ----------------------------------------
USERS_CSV = "users.csv"

def load_users():
    if not os.path.exists(USERS_CSV):
        df_empty = pd.DataFrame(columns=["username", "password", "role", "name", "phone"])
        df_empty.to_csv(USERS_CSV, index=False)
        return df_empty
    if os.stat(USERS_CSV).st_size == 0:
        df_empty = pd.DataFrame(columns=["username", "password", "role", "name", "phone"])
        df_empty.to_csv(USERS_CSV, index=False)
        return df_empty
    try:
        df = pd.read_csv(USERS_CSV)
        required_cols = ["username", "password", "role", "name", "phone"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""
        df = df[required_cols].fillna("")  # normalize
        return df
    except pd.errors.EmptyDataError:
        df_empty = pd.DataFrame(columns=["username", "password", "role", "name", "phone"])
        df_empty.to_csv(USERS_CSV, index=False)
        return df_empty

def save_user(username, password, role, name, phone):
    df = load_users()
    if username in df["username"].values:
        return False
    # store hashed password
    hashed = hash_password(password)
    new_row = {
        "username": username,
        "password": hashed,
        "role": role,
        "name": name,
        "phone": phone
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(USERS_CSV, index=False)
    return True

def authenticate(username, raw_password):
    df = load_users()
    user = df[df["username"] == username]
    if user.empty:
        return None
    stored_hash = str(user.iloc[0]["password"])
    if verify_password(raw_password, stored_hash):
        return {
            "username": user.iloc[0]["username"],
            "role": user.iloc[0]["role"],
            "name": user.iloc[0]["name"],
            "phone": user.iloc[0]["phone"]
        }
    return None

# ----------------------------------------
# Auth UI: Signup & Login
# ----------------------------------------
def signup_page():
    # App name at top (centered and highlighted)
    st.markdown("""
    <h1 style='text-align:center; font-weight:bold; color:#1B5E20; padding:12px;'>
        Welcome to Agrolytics🌿
    </h1>
    <p style='text-align:center; font-size:0.9rem; color:#555;'>Where Innovation Meets Agriculture☘️</p>
""", unsafe_allow_html=True)

    # Create New Account heading (centered)
    st.markdown("""
        <h2 style='text-align:center; margin-bottom:6px;'>
            Create New Account <span class='small muted'>(Register)</span>
        </h2>
    """, unsafe_allow_html=True)

    # Form container
    with st.container():
        col1, col2 = st.columns([1,1])
        with col1:
            username = st.text_input("Choose Username", key="su_username")
            password = st.text_input("Choose Password", type="password", key="su_password")
        with col2:
            name = st.text_input("Full Name", key="su_name")
            phone = st.text_input("Phone Number", key="su_phone")
        role = st.selectbox("Select Role", ["farmer", "buyer", "delivery"], key="su_role")

        if st.button("Create Account", key="su_create"):
            if not username or not password or not name or not phone:
                st.error("All fields are required.")
                return
            ok = save_user(username.strip(), password, role, name.strip(), phone.strip())
            if ok:
                st.success("Account created successfully! You can now log in.")
            else:
                st.error("Username already exists! Choose a different one.")

def login_page():
    # App name at top (centered and highlighted)
    st.markdown("""
    <h1 style='text-align:center; font-weight:bold; color:#1B5E20; padding:12px;'>
        Welcome to Agrolytics🌿
    </h1>
    <p style='text-align:center; font-size:0.9rem; color:#555;'>Where Innovation Meets Agriculture☘️</p>
""", unsafe_allow_html=True)

    # Login heading (centered)
    st.markdown("""
        <h2 style='text-align:center; margin-bottom:6px;'>
            Login<span class='small muted'>(Enter credentials)</span>
        </h2>
    """, unsafe_allow_html=True)
    with st.container():
        username = st.text_input("Username", key="li_username")
        password = st.text_input("Password", type="password", key="li_password")

        if st.button("Login", key="li_login"):
            user = authenticate(username.strip(), password)
            if user:
                st.session_state["login_success"] = True
                st.session_state["username"] = user["username"]
                st.session_state["role"] = user["role"]
                st.session_state["name"] = user["name"]
                st.session_state["phone"] = user["phone"]
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

# If not logged in: show signup/login UI and stop
if not st.session_state["login_success"]:
    left, right = st.columns([2,1])
    with left:
        if st.session_state["show_signup"]:
            signup_page()
            if st.button("Back to Login", key="su_back"):
                st.session_state["show_signup"] = False
                st.rerun()
        else:
            login_page()
            if st.button("New user? Sign up", key="li_new"):
                st.session_state["show_signup"] = True
                st.rerun()
    # some helpful sidebar hints
    with right:
        st.markdown("<div style='background:rgba(93,64,55,0.06); padding:12px; border-radius:8px;'>"
                    f"<h3 style='color:{PRIMARY}; margin:0 0 6px 0'>Quick tips</h3>"
                    "<ul style='margin:0; padding-left:18px; color:#4b4b4b;'>"
                    "<li>Use the usernames in <code>users.csv</code> to login.</li>"
                    "<li>Create new accounts with role 'farmer', 'buyer', or 'delivery'.</li>"
                    "</ul></div>", unsafe_allow_html=True)
    st.stop()

# After login -> show logout button
def logout():
    if st.button("Logout", key="logout_btn"):
        st.session_state["login_success"] = False
        st.session_state["role"] = None
        st.session_state["username"] = None
        st.session_state["name"] = None
        st.session_state["phone"] = None
        st.success("Logged out successfully!")
        st.rerun()

# top header area (nice layout)
col_h1, col_h2 = st.columns([4,1])
with col_h1:
    st.markdown(f"<h1 style='text-align:center;margin:6px 0 8px 0;'>Multi-Crop Price Predictor</h1>", unsafe_allow_html=True)
with col_h2: 
    st.markdown(f"<div style='text-align:right; font-size:0.95rem; color:#6b6b6b;'>Logged in as<br><strong>{st.session_state.get('username') or '—'}</strong></div>", unsafe_allow_html=True) 
    logout()


role_selection = st.session_state["role"]

# ----------------------------------------
# Filters (Farmer/Buyer) — same logic, slightly nicer sidebar labels
# ----------------------------------------
if role_selection in ["farmer", "buyer"]:
    st.sidebar.title(" Filters & Prediction")
    # protect against missing columns
    if "State" not in df.columns or "District" not in df.columns or "Crop" not in df.columns or "Modal Price" not in df.columns:
        st.error("dataset.csv missing required columns (State, District, Crop, Modal Price, Date etc.). Please check dataset.")
        st.stop()

    state_list = ["All"] + sorted(df["State"].dropna().unique().tolist())
    state = st.sidebar.selectbox("State", state_list, index=0)
    if state != "All":
        dist_list = ["All"] + sorted(df[df["State"] == state]["District"].dropna().unique().tolist())
    else:
        dist_list = ["All"] + sorted(df["District"].dropna().unique().tolist())
    district = st.sidebar.selectbox("District", dist_list, index=0)
    temp = df.copy()
    if state != "All":
        temp = temp[temp["State"] == state]
    if district != "All":
        temp = temp[temp["District"] == district]
    crop_list = ["Select crop"] + sorted(temp["Crop"].dropna().unique().tolist())
    crop = st.sidebar.selectbox("Crop", crop_list, index=0)
    future_date = st.sidebar.date_input("Future Date", value=pd.Timestamp.now().date())

    # Filtered Data
    filtered = df.copy()
    if state != "All":
        filtered = filtered[filtered["State"] == state]
    if district != "All":
        filtered = filtered[filtered["District"] == district]
    if crop != "Select crop":
        filtered = filtered[filtered["Crop"] == crop]
    if crop == "Select crop":
        st.info("Please select a Crop from the sidebar to enable prediction.")
        st.stop()
    hist = filtered.sort_values("Date")
    if hist.empty:
        st.warning("No historical records found for this selection.")
        st.stop()

    # Show historical data
    st.markdown("<div style='display:flex; gap:12px; align-items:center;'>"
                f"<h2 style='margin:0'>{crop} Prices <br> Historical data in {district}</h2>"
                f"<div style='margin-left:10px;'><span class='small muted'>Showing preview</span></div>"
                "</div>", unsafe_allow_html=True)
    st.dataframe(filtered.sort_values("Date").head(10))
    st.markdown("<div style='display:flex; gap:12px; align-items:center;'>"
                f"<h2 style='margin:0'>{crop} Prices <br> [Data from all states]</h2>"
                f"<div style='margin-center:10px;'><span class='small muted'>Showing preview</span></div>"
                "</div>", unsafe_allow_html=True)
    st.line_chart(hist.set_index("Date")["Modal Price"])

    # Prediction functions (unchanged)
    def find_closest_pred(filtered_df, target_date):
        if "Predicted Price" not in filtered_df.columns:
            return None, None
        d = filtered_df.dropna(subset=["Predicted Price"]).copy()
        if d.empty:
            return None, None
        d = d.sort_values("Date")
        d["diff_days"] = (d["Date"] - pd.to_datetime(target_date)).abs().dt.days
        row = d.loc[d["diff_days"].idxmin()]
        return float(row["Predicted Price"]), row["Date"]

    def linear_trend_predict(filtered_df, target_date):
        temp = filtered_df.sort_values("Date").copy()
        temp = temp.dropna(subset=["Modal Price"])
        if temp.shape[0] < 3:
            last = temp["Modal Price"].iloc[-1] if temp.shape[0] >= 1 else np.nan
            return float(last) if not np.isnan(last) else None
        t0 = temp["Date"].min()
        temp["t"] = (temp["Date"] - t0).dt.days.astype(float)
        x = temp["t"].values
        y = temp["Modal Price"].values
        m, c = np.polyfit(x, y, 1)
        future_t = (pd.to_datetime(target_date) - t0).days
        return float(m * future_t + c)

    pred_price, closest_date = find_closest_pred(filtered, future_date)
    if pred_price is None:
        pred_price = linear_trend_predict(filtered, future_date)
        pred_source = "Linear trend based on historical Modal Price"
    else:
        pred_source = f"From dataset 'Predicted Price' (closest date: {closest_date.date()})"
    if pred_price is None or np.isnan(pred_price):
        st.error("Could not compute prediction (insufficient data).")
        st.stop()

    # Recommendation (unchanged)
    def get_recommendation(last_price, predicted_price, role):
        perc_change = ((predicted_price - last_price) / last_price) * 100
        if perc_change > 3:
            rec = "HOLD / WAIT" if role in ["farmer","buyer"] else f"Monitor ({perc_change:.2f}%)"
        elif perc_change < -3:
            rec = "SELL / ACT NOW" if role=="farmer" else "BUY / ACT NOW" if role=="buyer" else f"Monitor ({perc_change:.2f}%)"
        else:
            rec = f"Monitor (small change {perc_change:.2f}%)"
        return rec, perc_change

    last_known_price = hist["Modal Price"].iloc[-1]
    recommendation, perc_change = get_recommendation(last_known_price, pred_price, role_selection)

# ----------------------------------------
# Farmer Dashboard
# ----------------------------------------
if role_selection == "farmer":
    st.markdown(f"<h1 style='text-align:center; margin:6px 0 8px 0;'>Farmer Dashboard</h1>", unsafe_allow_html=True)

    # Predictions (unchanged) — wrap in a colored container (Segment 1: Before 'Post your Crop for Sale')
    st.markdown("<div style='background:#f0e6d2; padding:12px; border-radius:8px;'>", unsafe_allow_html=True)
    st.success(f"Predicted {crop} price on {future_date}: ₹{pred_price:,.2f}")
    st.info(f"Recommendation: {recommendation}")
    st.caption(pred_source)
    st.subheader("Predicted Price for Selected Date")
    st.write(f"Predicted Price: ₹{pred_price:,.2f} | Change: {perc_change:.2f}%")
    out_df = pd.DataFrame([{
        "State": state, "District": district, "Crop": crop,
        "date_of_prediction": future_date, "predicted_price": pred_price
    }])
    st.download_button("Download prediction CSV", StringIO(out_df.to_csv(index=False)).getvalue(),
                       file_name=f"{crop}_prediction_{future_date}.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

    # Post Crop (Segment 2: 'Post your Crop for Sale' form)
    st.markdown("<div style='background:#fff2e6; padding:12px; border-radius:8px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#5D4037; font-family:Helvetica Neue;'>Post your Crop for Sale</h3>", unsafe_allow_html=True)
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
                "farmer_name": st.session_state.get("name", "Farmer Name"),
                "location": f"{state}, {district}",
                "crop_name": crop_name,
                "quantity": quantity,
                "phone_number": phone_number,
                "image": image_path
            }
            st.session_state["farmer_posts"].append(post_data)
            save_farmer_posts(st.session_state["farmer_posts"])
            st.success(f"{crop_name} posted successfully!")
    st.markdown("</div>", unsafe_allow_html=True)

    # Display farmer's own posts with remove option (Segment 3: 'Your posted crops')
    st.markdown("<div style='background:#e6f7ff; padding:12px; border-radius:8px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#5D4037; font-family:Helvetica Neue;'>Your posted crops</h3>", unsafe_allow_html=True)
    st.session_state["farmer_posts"] = load_farmer_posts()
    own_posts = [p for p in st.session_state["farmer_posts"] if p.get("farmer_id") == st.session_state["username"]]
    if own_posts:
        for idx, post in enumerate(own_posts):
            st.markdown("<div style='background:var(--card); padding:10px; border-radius:8px; margin-bottom:8px;'>", unsafe_allow_html=True)
            st.write(f"**Crop:** {post.get('crop_name','')}")
            st.write(f"**Quantity:** {post.get('quantity','')} kg")
            st.write(f"**Phone Number:** {post.get('phone_number','N/A')}")
            st.write(f"**Location:** {post.get('location','')}")
            img = post.get('image', "")
            if isinstance(img, str) and img:
                if os.path.exists(img):
                    st.image(img, width=200)
                else:
                    st.write("_Image file missing_")
            remove_col1, remove_col2 = st.columns([4,1])
            with remove_col2:
                if st.button("Remove Post", key=f"remove_post_{idx}", help="Remove this crop post"):
                    st.session_state["farmer_posts"].remove(post)
                    save_farmer_posts(st.session_state["farmer_posts"])
                    st.success("Post removed.")
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("You have not posted any crops yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Delivery Made Easy (Segment 4: Delivery section)
    st.markdown("<div style='background:#f9e6ff; padding:12px; border-radius:8px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; margin:6px 0 8px 0; color:#3a2b22;'>Delivery Made Easy</h2>", unsafe_allow_html=True)
    if st.button("Need Delivery", key="need_delivery_farmer"):
        st.session_state["show_delivery_form_farmer"] = True

    if st.session_state["show_delivery_form_farmer"]:
        with st.form("delivery_form_farmer", clear_on_submit=False):
            st.write("Enter delivery request details")
            loc = st.text_input("Your location", key="df_loc")
            dest = st.text_input("Destination", key="df_dest")
            mode = st.selectbox("Mode of transport", ["Bike", "Auto", "Tractor", "Tempo", "Lorry"], key="df_mode")
            phone = st.text_input("Phone number", value=st.session_state.get("phone",""), key="df_phone")
            submitted = st.form_submit_button("Submit Delivery Request", key="df_submit")
            cancel = st.form_submit_button("Cancel", key="df_cancel")
            if submitted:
                ts = datetime.now().isoformat()
                request_id = f"{st.session_state['username']}_{int(datetime.now().timestamp())}"
                req = {
                    "request_id": request_id,
                    "username": st.session_state["username"],
                    "role": st.session_state["role"],
                    "location": loc,
                    "destination": dest,
                    "mode": mode,
                    "phone": phone,
                    "timestamp": ts
                }
                current = load_delivery_requests()
                current.append(req)
                st.session_state["delivery_requests"] = current
                save_delivery_requests(st.session_state["delivery_requests"])
                st.success("Delivery request submitted.")
                st.session_state["show_delivery_form_farmer"] = False
                st.rerun()
            if cancel:
                st.session_state["show_delivery_form_farmer"] = False
                st.rerun()

    st.markdown("<h3 style='text-align:center; color:#5D4037; font-family:Helvetica Neue;'>Your delivery requests</h3>", unsafe_allow_html=True)
    st.session_state["delivery_requests"] = load_delivery_requests()
    my_reqs = [r for r in st.session_state["delivery_requests"] if r.get("username") == st.session_state["username"]]
    if my_reqs:
        for r in my_reqs:
            st.markdown("<div style='background:var(--card); padding:10px; border-radius:8px; margin-bottom:8px;'>", unsafe_allow_html=True)
            st.write(f"Request ID: {r.get('request_id')}")
            st.write(f"From: {r.get('location')} → To: {r.get('destination')} | Mode: {r.get('mode')}")
            st.write(f"Phone: {r.get('phone')} | Time: {r.get('timestamp')}")
            if st.button("Remove Request", key=f"remove_del_{r.get('request_id')}"):
                cur = load_delivery_requests()
                cur = [x for x in cur if x.get("request_id") != r.get("request_id")]
                st.session_state["delivery_requests"] = cur
                save_delivery_requests(st.session_state["delivery_requests"])
                st.success("Delivery request removed.")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No delivery requests yet.")
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------
# Buyer Dashboard (view-only + Need Delivery)
# ----------------------------------------
elif role_selection == "buyer":
    # Segment 1: Buyer Dashboard main area
    st.markdown("<div style='background:#fff4e6; padding:12px; border-radius:8px;'>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align:center; margin:6px 0 8px 0;'>Buyer Dashboard</h1>", unsafe_allow_html=True)
    st.success(f"Predicted {crop} price on {future_date}: ₹{pred_price:,.2f}")
    st.info(f"Recommendation: {recommendation}")
    st.caption(pred_source)
    st.subheader("Predicted Price for Selected Date")
    st.write(f"Predicted Price: ₹{pred_price:,.2f} | Change: {perc_change:.2f}%")
    out_df = pd.DataFrame([{
        "State": state, "District": district, "Crop": crop,
        "date_of_prediction": future_date, "predicted_price": pred_price
    }])
    st.download_button("Download prediction CSV", StringIO(out_df.to_csv(index=False)).getvalue(),
                       file_name=f"{crop}_prediction_{future_date}.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

    # Segment 2: Available Crops for Purchase
    st.markdown("<div style='background:#f0f0ff; padding:12px; border-radius:8px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#5D4037; font-family:Helvetica Neue;'>Available Crops for Purchase</h3>", unsafe_allow_html=True)
    st.session_state["farmer_posts"] = load_farmer_posts()
    if st.session_state["farmer_posts"]:
        for idx, post in enumerate(st.session_state["farmer_posts"]):
            st.markdown("<div style='background:var(--card); padding:10px; border-radius:8px; margin-bottom:8px;'>", unsafe_allow_html=True)
            st.write(f"**Crop:** {post.get('crop_name','')}")  
            st.write(f"**Quantity:** {post.get('quantity','')} kg")
            st.write(f"**Farmer Location:** {post.get('location','')}")  
            st.write(f"**Phone Number:** {post.get('phone_number','N/A')}")  
            img = post.get('image', "")
            if isinstance(img, str) and img:
                if os.path.exists(img):
                    st.image(img, width=200)
                else:
                    st.write("_Image file missing_")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No crops available currently.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Segment 3: Delivery Made Easy
    st.markdown("<div style='background:#f9e6ff; padding:12px; border-radius:8px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; margin:6px 0 8px 0; color:#3a2b22;'>Delivery Made Easy</h2>", unsafe_allow_html=True)
    if st.button("Need Delivery", key="need_delivery_buyer"):
        st.session_state["show_delivery_form_buyer"] = True

    if st.session_state["show_delivery_form_buyer"]:
        with st.form("delivery_form_buyer", clear_on_submit=False):
            st.write("Enter delivery request details")
            loc = st.text_input("Your location", key="db_loc")
            dest = st.text_input("Destination", key="db_dest")
            mode = st.selectbox("Mode of transport", ["Bike", "Auto", "Tractor", "Tempo", "Lorry"], key="db_mode")
            phone = st.text_input("Phone number", value=st.session_state.get("phone",""), key="db_phone")
            submitted = st.form_submit_button("Submit Delivery Request", key="db_submit")
            cancel = st.form_submit_button("Cancel", key="db_cancel")
            if submitted:
                ts = datetime.now().isoformat()
                request_id = f"{st.session_state['username']}_{int(datetime.now().timestamp())}"
                req = {
                    "request_id": request_id,
                    "username": st.session_state["username"],
                    "role": st.session_state["role"],
                    "location": loc,
                    "destination": dest,
                    "mode": mode,
                    "phone": phone,
                    "timestamp": ts
                }
                cur = load_delivery_requests()
                cur.append(req)
                st.session_state["delivery_requests"] = cur
                save_delivery_requests(st.session_state["delivery_requests"])
                st.success("Delivery request submitted.")
                st.session_state["show_delivery_form_buyer"] = False
                st.rerun()
            if cancel:
                st.session_state["show_delivery_form_buyer"] = False
                st.rerun()

    st.markdown("<h3 style='text-align:center; color:#5D4037; font-family:Helvetica Neue;'>Your delivery requests</h3>", unsafe_allow_html=True)
    st.session_state["delivery_requests"] = load_delivery_requests()
    my_reqs = [r for r in st.session_state["delivery_requests"] if r.get("username") == st.session_state["username"]]
    if my_reqs:
        for r in my_reqs:
            st.markdown("<div style='background:var(--card); padding:10px; border-radius:8px; margin-bottom:8px;'>", unsafe_allow_html=True)
            st.write(f"Request ID: {r.get('request_id')}")
            st.write(f"From: {r.get('location')} → To: {r.get('destination')} | Mode: {r.get('mode')}")
            st.write(f"Phone: {r.get('phone')} | Time: {r.get('timestamp')}")
            if st.button("Remove Request", key=f"remove_del_buyer_{r.get('request_id')}"):
                cur = load_delivery_requests()
                cur = [x for x in cur if x.get("request_id") != r.get("request_id")]
                st.session_state["delivery_requests"] = cur
                save_delivery_requests(st.session_state["delivery_requests"])
                st.success("Delivery request removed.")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No delivery requests yet.")
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------
# Delivery Dashboard: sees all requests
# ----------------------------------------
elif role_selection == "delivery":
    st.markdown(f"<h1 style='text-align:center; margin:6px 0 8px 0;'>Delivery Dashboard</h1>", unsafe_allow_html=True)
    st.info("All delivery requests (farmers & buyers)")

    # Reload from disk to ensure latest (in case other users added)
    st.session_state["delivery_requests"] = load_delivery_requests()

    if st.session_state["delivery_requests"]:
        for idx, r in enumerate(st.session_state["delivery_requests"]):
            st.markdown("<div style='background:var(--card); padding:10px; border-radius:8px; margin-bottom:8px;'>", unsafe_allow_html=True)
            st.write(f"**Request ID:** {r.get('request_id')}")
            st.write(f"User: {r.get('username')} ({r.get('role')})")
            st.write(f"From: {r.get('location')} → To: {r.get('destination')} | Mode: {r.get('mode')}")
            st.write(f"Phone: {r.get('phone')} | Time: {r.get('timestamp')}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No delivery requests yet.")
