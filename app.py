import streamlit as st
import pandas as pd
from db import init_db, save_search, get_history
from data_engine import generate_mock_data
from ml_model import train_and_predict_best_deal
from vision import identify_product_from_image
import os
from dotenv import load_dotenv

# --- PAGE CONFIG ---
st.set_page_config(page_title="PriceLens", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

load_dotenv(override=True)
gemini_api_key = os.environ.get("GEMINI_API_KEY")
init_db()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Reset & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .stApp {
        background-color: #f8fafe;
        font-family: 'Inter', sans-serif;
    }
    
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 4rem !important;
        max-width: 1050px;
    }

    /* --- SIDEBAR --- */
    section[data-testid="stSidebar"] {
        background-color: #f7f9fc !important;
        border-right: none !important;
        box-shadow: 2px 0 20px rgba(0,0,0,0.02);
    }
    [data-testid="stSidebarNav"] {display: none;}
    
    /* History Items */
    .history-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 18px;
        margin: 4px 0;
        border-radius: 12px;
        cursor: pointer;
        transition: background 0.2s;
    }
    .history-item:hover { background: #f0f4ff; }
    .history-item.active { background: #e8ecfa; }
    .history-title {
        font-size: 0.9em; font-weight: 500; color: #444;
        display: flex; align-items: center; gap: 12px;
    }
    .history-time { font-size: 0.75em; color: #999; }

    /* --- MAIN CONTENT --- */
    /* Search Bar Group */
    div[data-testid="stTextInput"] * {
        border-color: transparent !important;
    }
    div[data-testid="stTextInput"] [data-baseweb="base-input"] {
        background: white !important;
        border-radius: 30px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
        overflow: hidden !important;
    }
    div[data-testid="stTextInput"] input {
        background: transparent !important;
        padding: 15px 60px 15px 30px !important;
        font-size: 1.05em !important;
        height: 60px !important;
    }
    
    /* Absolute position for the blue search button inside input */
    div[data-testid="stTextInput"] {
        position: relative !important;
    }
    div[data-testid="stTextInput"]::after {
        content: "🔍" !important;
        position: absolute !important;
        right: 8px !important;
        top: 8px !important;
        background: linear-gradient(135deg, #7c3aed, #5d5fef) !important; /* Purple-blue gradient */
        color: white !important;
        width: 44px !important;
        height: 44px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.2em !important;
        z-index: 10 !important;
        box-shadow: 0 4px 10px rgba(124, 58, 237, 0.3) !important;
        pointer-events: none !important;
    }

    /* Scan Button Hack */
    [data-testid="stFileUploader"] {
        margin: 0 !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0 !important;
        height: 60px !important;
        min-height: 60px !important;
        position: relative !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04) !important;
        overflow: hidden !important;
    }
    /* Hide the default button visually but keep it clickable */
    [data-testid="stFileUploadDropzone"] button {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        z-index: 50 !important;
        opacity: 0 !important;
        cursor: pointer !important;
    }
    /* Use ::before to cover everything else in the dropzone with our custom design */
    [data-testid="stFileUploadDropzone"]::before {
        content: "📷 Scan Image" !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: white !important;
        color: #5d5fef !important;
        font-weight: 600 !important;
        font-size: 1.05em !important;
        border-radius: 30px !important;
        z-index: 10 !important;
        pointer-events: none !important;
    }
    
    /* Popular Searches Pills */
    .pop-pill {
        background: white;
        padding: 10px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        font-weight: 500;
        font-size: 0.9em;
        color: #444;
        display: flex;
        align-items: center;
        gap: 10px;
        cursor: pointer;
    }
    .pop-logo {
        width: 22px; height: 22px; border-radius: 4px; display:flex; align-items:center; justify-content:center; color:white; font-size:10px; font-weight:bold;
    }

    /* Bottom Features */
    .feature-box {
        display: flex;
        align-items: flex-start;
        gap: 15px;
    }
    .feature-icon {
        background: #eef2ff;
        color: #5d5fef;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1em;
        flex-shrink: 0;
    }
    
    /* Results Cards */
    .card-container {
        border-radius: 12px !important;
        background: white !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03) !important;
        border: 1px solid #f5f5f5 !important;
    }
    
    /* Clear history button */
    .stButton[data-testid="stSidebar"] button {
        background: white !important;
        border: 1px solid #eee !important;
        border-radius: 8px !important;
        color: #666 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR UI ---
st.sidebar.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 30px; margin-top: 10px;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="background: transparent; color: #5d5fef; font-size: 26px;">🔍</div>
        <h2 style="margin: 0; color: #5d5fef; font-weight: 800; font-size: 1.6em; letter-spacing: -0.5px;">priceLens</h2>
    </div>
    <div style="color: #888; font-size: 1.2em;">«</div>
</div>
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px; padding-left: 5px;">
    <span style="font-size: 1.2em; color: #555;">🕒</span>
    <span style="font-weight: 700; color: #222; font-size: 1.05em;">Search History</span>
</div>
""", unsafe_allow_html=True)

if 'history_loaded' not in st.session_state:
    st.session_state.history = get_history()
    st.session_state.history_loaded = True

if not st.session_state.history.empty:
    for idx, row in st.session_state.history.iterrows():
        bg_class = "active" if idx == 0 else ""
        st.sidebar.markdown(f"""
        <div class="history-item {bg_class}">
            <div class="history-title">
                <span style="color:#aaa; font-size:1.1em;">🕒</span> {row['product_name'][:18]}
            </div>
            <div class="history-time">{row['best_platform']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    if st.sidebar.button("🗑️ Clear History", use_container_width=True):
        import sqlite3
        conn = sqlite3.connect("pricelens.db")
        conn.cursor().execute("DELETE FROM search_history")
        conn.commit()
        st.session_state.history = get_history()
        st.rerun()
else:
    st.sidebar.markdown("<div style='padding-left:10px; color:#888;'>No history yet.</div>", unsafe_allow_html=True)


# --- MAIN HEADER ---
st.markdown("""
<div style="text-align: center; margin-bottom: 40px;">
    <h1 style="font-size: 3.5em; font-weight: 800; color: #111; margin-bottom: 10px; letter-spacing:-1px;">
        Discover the <span style="color: #5d5fef;">Best Prices</span>
    </h1>
    <p style="color: #777; font-size: 1.1em;">Search, compare, and save on millions of products</p>
</div>
""", unsafe_allow_html=True)


# --- SEARCH AREA ---
col1, col2 = st.columns([7, 2])
search_query = ""
scanned_image_b64 = None

with col1:
    search_input = st.text_input("Search", placeholder="🔍 Search for any product...", label_visibility="collapsed")

with col2:
    uploaded_file = st.file_uploader("Scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# --- POPULAR SEARCHES ---
st.markdown("""
<div style="margin-top: 25px; margin-bottom: 40px;">
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        <span style="font-size: 1.2em;">🔥</span>
        <span style="font-weight: 600; color: #555; font-size:0.95em;">Popular Searches</span>
    </div>
    <div style="display: flex; gap: 15px; flex-wrap: wrap; align-items: center;">
        <div class="pop-pill"><div class="pop-logo" style="background:#0a3e9c;">S</div> Samsung</div>
        <div class="pop-pill"><div class="pop-logo" style="background:#4169e1;">V</div> Vivo</div>
        <div class="pop-pill"><div class="pop-logo" style="background:#ffcc00; color:#333;">R</div> Realme</div>
        <div class="pop-pill"><div class="pop-logo" style="background:#000;"></div> iPhone 16</div>
        <div class="pop-pill"><div class="pop-logo" style="background:#888;">💻</div> MacBook Air</div>
        <div style="background:white; width:35px; height:35px; border-radius:50%; display:flex; align-items:center; justify-content:center; color:#5d5fef; font-weight:bold; box-shadow:0 2px 10px rgba(0,0,0,0.05); cursor:pointer;">❯</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Image Scan Processing
if uploaded_file is not None:
    if gemini_api_key:
        with st.spinner("Analyzing image with Gemini AI..."):
            image_bytes = uploaded_file.read()
            detected_product = identify_product_from_image(image_bytes, gemini_api_key)
            if "Error:" not in detected_product:
                search_query = detected_product
                import base64
                scanned_image_b64 = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
            else:
                st.error(f"Error details: {detected_product}")
    else:
        st.error("No Gemini API Key found!")

if search_input and not search_query:
    search_query = search_input

# --- RESULTS OR BANNERS ---
if search_query:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.spinner("Fetching live data & running ML model..."):
        results_df = generate_mock_data(search_query)
        best_deal, model = train_and_predict_best_deal(results_df)
        save_search(search_query, best_deal['Platform'], best_deal['Price (₹)'])
        
        st.subheader("📊 Top Picks For You")
        
        display_df = results_df.drop(columns=['Deal_Score'])
        
        cols = st.columns(3)
        for idx, row in display_df.iterrows():
            with cols[idx % 3]:
                platform_color = "#ff9900" if row['Platform'] == "Amazon" else ("#2874f0" if row['Platform'] == "Flipkart" else "#ea4335")
                best_badge = '<div style="position: absolute; top: -12px; right: -12px; background: #28a745; color: white; padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 0.9em; box-shadow: 0 4px 6px rgba(0,0,0,0.2); z-index: 10;">🏆 Best Deal</div>' if row['Platform'] == best_deal['Platform'] else ''
                final_image = scanned_image_b64 if scanned_image_b64 else row['Image_URL']

                card_html = f"""<div class="card-container" style="position: relative; overflow: hidden; padding: 15px; margin-bottom: 20px;">
{best_badge}
<div style="height: 180px; display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
<img src="{final_image}" style="max-height: 100%; max-width: 100%; object-fit: contain;">
</div>
<div style="font-size: 0.95em; color: #333; height: 2.8em; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; margin-bottom: 8px;">
{row['Product']}
</div>
<div style="background: #388e3c; color: white; display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold; margin-bottom:15px;">
{row['Rating']} ★
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-top:1px solid #f0f0f0; padding-top:15px;">
<div>
<div style="color: {platform_color}; font-weight: 900; font-style: italic; font-size: 0.8em; margin-bottom:2px;">{row['Platform']}</div>
<div style="color: #111; font-size: 1.2em; font-weight: 800;">₹{int(row['Price (₹)']):,}</div>
</div>
<a href="{row['Link']}" target="_blank" style="background: #5d5fef; color: white; padding: 8px 16px; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 0.9em;">Buy</a>
</div>
</div>"""
                st.markdown(card_html, unsafe_allow_html=True)

# --- PROMO BANNERS ---
st.markdown("""
<div style="background: #f8f9ff; border-radius: 12px; padding: 25px 30px; margin-top: 50px; display: flex; align-items: center; gap: 20px;">
    <div style="background: #5d5fef; width: 55px; height: 55px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 4px 10px rgba(93,95,239,0.2);">
        <span style="display: inline-block; transform: rotate(-45deg);">🏷️</span>
    </div>
    <div>
        <h3 style="margin: 0 0 5px 0; color: #111; font-weight: 700; font-size: 1.25em;">Find the Best Deals</h3>
        <p style="margin: 0; color: #777; font-size: 0.95em;">Compare prices from trusted stores and get the best offers</p>
    </div>
</div>

<hr style="border: 0; border-top: 1px solid #f0f0f0; margin: 40px 0;">

<div style="display: flex; justify-content: space-around; padding: 0 20px 40px 20px; text-align: center;">
    <div style="flex: 1;">
        <div style="background: #f0f4ff; color: #5d5fef; width: 45px; height: 45px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; margin: 0 auto 15px auto;">🔒</div>
        <h4 style="margin: 0 0 8px 0; font-weight: 700; color: #222; font-size: 1.1em;">100% Secure</h4>
        <p style="color: #888; font-size: 0.85em; margin: 0;">Your data is safe with us</p>
    </div>
    <div style="flex: 1;">
        <div style="background: #f0f4ff; color: #5d5fef; width: 45px; height: 45px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; margin: 0 auto 15px auto;">⏱️</div>
        <h4 style="margin: 0 0 8px 0; font-weight: 700; color: #222; font-size: 1.1em;">Real-time Prices</h4>
        <p style="color: #888; font-size: 0.85em; margin: 0;">Always get the latest prices</p>
    </div>
    <div style="flex: 1;">
        <div style="background: #f0f4ff; color: #5d5fef; width: 45px; height: 45px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; margin: 0 auto 15px auto;">🛡️</div>
        <h4 style="margin: 0 0 8px 0; font-weight: 700; color: #222; font-size: 1.1em;">Trusted Sources</h4>
        <p style="color: #888; font-size: 0.85em; margin: 0;">From verified sellers only</p>
    </div>
</div>
""", unsafe_allow_html=True)
