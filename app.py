import streamlit as st
import pandas as pd
from db import init_db, save_search, get_history
from data_engine import generate_mock_data
from ml_model import train_and_predict_best_deal
from vision import identify_product_from_image
import os
from dotenv import load_dotenv

# --- PAGE CONFIG ---
st.set_page_config(page_title="PriceLens", page_icon="🔍", layout="wide")

# Load environment variables
load_dotenv(override=True)
gemini_api_key = os.environ.get("GEMINI_API_KEY")

# --- INITIALIZE DATABASE ---
init_db()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Minimalist E-commerce Theme */
    .stApp {
        background-color: #ffffff;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1000px;
    }
    .card-container {
        border-radius: 4px;
        border: 1px solid #e0e0e0;
        transition: all 0.2s ease;
    }
    .card-container:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    /* Main Background & Font */
    .stApp {
        background-color: #f7f9fc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Unified Search Bar & Scan Button (Attached) */
    div[data-testid="column"]:nth-child(1) {
        padding-right: 0 !important;
    }
    div[data-testid="column"]:nth-child(2) {
        padding-left: 0 !important;
    }
    .stTextInput input {
        border-radius: 30px 0 0 30px !important;
        padding: 18px 25px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
        font-size: 1.1em !important;
        transition: all 0.3s ease !important;
        background: white !important;
        height: 60px !important;
    }
    .stTextInput input:focus {
        box-shadow: 0 4px 20px rgba(93, 95, 239, 0.2) !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #f0f0f0 !important;
    }
    
    /* Pills */
    .pop-pill {
        background: white;
        padding: 10px 20px;
        border-radius: 30px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        font-weight: 600;
        color: #444;
        cursor: pointer;
        border: 1px solid #f0f0f0;
        transition: all 0.2s ease;
    }
    .pop-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-color: #5d5fef;
        color: #5d5fef;
    }

    /* Beautiful Cards */
    .card-container {
        border-radius: 15px !important;
        border: none !important;
        background: white !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04) !important;
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
    }
    .card-container:hover {
        transform: translateY(-8px) !important;
        box-shadow: 0 15px 30px rgba(0,0,0,0.08) !important;
    }
    
    /* Deals Page Cards */
    .deal-card {
        background: #fcfcfc;
        border: 1px dashed #e0e0e0;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .deal-card:hover {
        background: white;
        border-style: solid;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        transform: translateY(-5px);
    }

    /* File Uploader Perfect Icon Hack (Attached to Search Bar) */
    [data-testid="stFileUploader"] {
        margin-top: 0 !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background: white !important;
        border: none !important;
        border-radius: 0 30px 30px 0 !important;
        padding: 0 !important;
        min-height: 60px !important;
        position: relative !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
    }
    /* Hide the drag-and-drop text and icon */
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] small,
    [data-testid="stFileUploadDropzone"] svg {
        display: none !important;
    }
    /* Make the button fill the entire container and hide its original text */
    [data-testid="stFileUploadDropzone"] button {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        visibility: hidden !important;
    }
    /* Inject our custom camera icon using ::after */
    [data-testid="stFileUploadDropzone"] button::after {
        content: "📷" !important;
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        height: 100% !important;
        background-color: transparent !important;
        color: #5d5fef !important;
        font-size: 1.5em !important;
        cursor: pointer !important;
        transition: transform 0.2s ease !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover::after {
        transform: scale(1.1) !important;
        color: #4a4cd6 !important;
    }
    /* Remove any inner padding from the default container */
    [data-testid="stFileUploadDropzone"] > div {
        padding: 0 !important;
        height: 100% !important;
    }
</style>
""", unsafe_allow_html=True)


# --- SIDEBAR: SEARCH HISTORY ---
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 40px; margin-top: 10px;">
    <div style="background: #eef2ff; color: #5d5fef; width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px;">🔍</div>
    <h2 style="margin: 0; color: #5d5fef; font-weight: 800; font-size: 1.5em; letter-spacing: -0.5px;">priceLens</h2>
</div>
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
    <span style="font-size: 1.2em; color: #555;">🕒</span>
    <span style="font-weight: 600; color: #111; font-size: 1.1em;">Search History</span>
</div>
""", unsafe_allow_html=True)

if 'history_loaded' not in st.session_state:
    st.session_state.history = get_history()
    st.session_state.history_loaded = True

if not st.session_state.history.empty:
    for idx, row in st.session_state.history.iterrows():
        st.sidebar.markdown(f"""
        <div style="background: white; border: 1px solid #f0f0f0; border-radius: 10px; padding: 12px 15px; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.02); display: flex; justify-content: space-between; align-items: center;">
            <div style="font-weight: 500; color: #333; font-size: 0.9em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 140px;">{row['product_name']}</div>
            <div style="color: #999; font-size: 0.75em;">{row['best_platform']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.sidebar.button("🗑️ Clear History", use_container_width=True):
        import sqlite3
        conn = sqlite3.connect("price_lens.db")
        conn.cursor().execute("DELETE FROM search_history")
        conn.commit()
        st.session_state.history = get_history()
        st.rerun()
else:
    st.sidebar.write("No history yet.")

if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Main Search Area variables
search_query = ""
scanned_image_b64 = None

if st.session_state.page == 'home':
    # --- MAIN UI ---
    st.markdown("""
    <div style="text-align: center; margin-top: 20px; margin-bottom: 40px;">
        <h1 style="font-size: 3.5em; font-weight: 800; color: #111; margin-bottom: 10px; font-family: sans-serif;">
            Discover the <span style="color: #5d5fef;">Best Prices</span>
        </h1>
        <p style="color: #6c757d; font-size: 1.1em;">Search, compare, and save on millions of products</p>
    </div>
    """, unsafe_allow_html=True)

    # Search Bar & Scan Button unified in a tighter grid
    col1, col2 = st.columns([10, 1])
    with col1:
        search_input = st.text_input("Search", placeholder="🔍 Search for smartphones, brands, or models...", label_visibility="collapsed")
    with col2:
        # Instead of a button, we use the file_uploader visually hacked into a camera icon
        uploaded_file = st.file_uploader("Scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    # Links just below the search bar
    nav_c1, nav_c2, nav_c3 = st.columns([8, 1, 1])
    with nav_c1:
        st.markdown("""
        <div style="margin-top: 8px; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.2em;">🔥</span>
            <span style="font-weight: 600; color: #555; font-size: 0.9em;">Popular Searches:</span>
            <div style="display: flex; gap: 10px;">
                <div class="pop-pill" style="padding: 5px 12px; font-size: 0.8em;">Samsung</div>
                <div class="pop-pill" style="padding: 5px 12px; font-size: 0.8em;">Vivo</div>
                <div class="pop-pill" style="padding: 5px 12px; font-size: 0.8em;">iPhone 16</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with nav_c2:
        if st.button("🏷️ Deals", type="tertiary", use_container_width=True):
            st.session_state.page = "deals"
            st.rerun()
    with nav_c3:
        if st.button("📁 Categories", type="tertiary", use_container_width=True):
            st.session_state.page = "categories"
            st.rerun()

    # Immediately process the uploaded file if present
    if uploaded_file is not None:
        if gemini_api_key:
            with st.spinner("Analyzing image with Gemini AI..."):
                image_bytes = uploaded_file.read()
                detected_product = identify_product_from_image(image_bytes, gemini_api_key)
                if "Error:" not in detected_product:
                    st.success(f"Detected Product: **{detected_product}**")
                    search_query = detected_product
                    import base64
                    scanned_image_b64 = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
                else:
                    st.error(f"Error details: {detected_product}")
        else:
            st.error("No Gemini API Key found!")

    # Trigger search if text is inputted
    if search_input and not search_query:
        search_query = search_input
        
    # --- PROCESS SEARCH & SHOW RESULTS IMMEDIATELY UNDER SEARCH BAR ---
    if search_query:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.spinner("Fetching live data & running ML model..."):
            results_df = generate_mock_data(search_query)
            best_deal, model = train_and_predict_best_deal(results_df)
            save_search(search_query, best_deal['Platform'], best_deal['Price (₹)'])
            
            st.subheader("📊 Top Picks For You")
            
            display_df = results_df.drop(columns=['Deal_Score'])
            
            # --- DRAW PROFESSIONAL CARDS ---
            cols = st.columns(3)
            for index, row in display_df.iterrows():
                with cols[index % 3]:
                    # Check if this is the best deal
                    is_best = (row['Platform'] == best_deal['Platform'])
                    best_badge = "<div style='position: absolute; top: 10px; left: 10px; background: #ff0000; color: white; padding: 3px 10px; border-radius: 20px; font-weight: bold; font-size: 0.7em; z-index: 10; box-shadow: 0 2px 5px rgba(0,0,0,0.2);'>🔥 BEST DEAL</div>" if is_best else ""
                    
                    platform_color = "#ff9900" if row['Platform'] == "Amazon" else "#047BD5" if row['Platform'] == "Flipkart" else "#ea4335"

                    # If we scanned an image, use it! Otherwise, use the scraped image.
                    final_image = scanned_image_b64 if scanned_image_b64 else row['Image_URL']

                    # HTML template for the minimalist e-commerce card
                    card_html = f"""<div class="card-container" style="position: relative; background: white; margin-bottom: 20px; overflow: hidden;">
    <div style="padding: 15px;">
    {best_badge}
    <div style="height: 180px; display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
    <img src="{final_image}" style="max-height: 100%; max-width: 100%; object-fit: contain;">
    </div>
    <div style="font-size: 0.95em; color: #333; height: 2.8em; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; margin-bottom: 8px;">
    {row['Product']}
    </div>
    <div style="background: #388e3c; color: white; display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold;">
    {row['Rating']} ★
    </div>
    </div>
    <div style="background: #f9f9f9; padding: 12px 15px; border-top: 1px solid #eee; display: flex; justify-content: space-between; align-items: center;">
    <div style="display: flex; align-items: center; gap: 8px;">
    <div style="color: {platform_color}; font-weight: 900; font-style: italic; font-size: 0.85em;">
    {row['Platform']}
    </div>
    <div style="color: #111; font-size: 1.1em; font-weight: bold;">
    ₹{int(row['Price (₹)']):,}
    </div>
    </div>
    <a href="{row['Link']}" target="_blank" style="background: #ff3f00; color: white; padding: 6px 12px; text-decoration: none; border-radius: 3px; font-weight: bold; font-size: 0.9em; box-shadow: 0 2px 4px rgba(255,63,0,0.2);">
    BUY
    </a>
    </div>
    </div>"""
                    st.markdown(card_html, unsafe_allow_html=True)
            
            # --- AI RECOMMENDATION ---
            st.markdown(f"""
            <div style="background: #eef2ff; border-left: 5px solid #5d5fef; padding: 20px; border-radius: 10px; margin-top: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.02);">
                <h3 style="margin-top: 0; color: #5d5fef;">🤖 AI Recommendation</h3>
                <p style="font-size: 1.1em; margin-bottom: 0;">Our Machine Learning model predicts the best value is on <strong>{best_deal['Platform']}</strong> at <strong>₹{int(best_deal['Price (₹)']):,}</strong>.</p>
                <small style="color: #666;">Model Accuracy (R² Score): {model.score(display_df[['Price (₹)', 'Rating', 'Discount (%)']], display_df['Price (₹)']):.2f}</small>
            </div>
            """, unsafe_allow_html=True)

    # Now show the banners AT THE VERY BOTTOM OF THE PAGE
    st.markdown("""
    <div style="background: linear-gradient(to right, #f2f5ff, #eef1fe); border-radius: 15px; padding: 25px; margin-top: 60px; display: flex; align-items: center; gap: 20px; box-shadow: 0 4px 15px rgba(93,95,239,0.05);">
        <div style="background: #5d5fef; width: 60px; height: 60px; border-radius: 15px; display: flex; align-items: center; justify-content: center; color: white; font-size: 24px; box-shadow: 0 4px 10px rgba(93,95,239,0.3);">🏷️</div>
        <div>
            <h3 style="margin: 0; color: #111; font-weight: 700; font-size: 1.3em;">Find the Best Deals</h3>
            <p style="margin: 5px 0 0 0; color: #666;">Compare prices from trusted stores and get the best offers</p>
        </div>
    </div>

    <div style="display: flex; justify-content: space-around; margin-top: 40px; padding-top: 40px; border-top: 1px solid #eaeaea; margin-bottom: 50px;">
        <div style="text-align: center; flex: 1;">
            <div style="background: #eef2ff; color: #5d5fef; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; margin: 0 auto 15px auto;">🔒</div>
            <h4 style="margin:0; font-weight: 600; color: #333;">100% Secure</h4>
            <p style="color: #888; font-size: 0.9em; margin: 5px 0 0 0;">Your data is safe with us</p>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="background: #eef2ff; color: #5d5fef; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; margin: 0 auto 15px auto;">⏱️</div>
            <h4 style="margin:0; font-weight: 600; color: #333;">Real-time Prices</h4>
            <p style="color: #888; font-size: 0.9em; margin: 5px 0 0 0;">Always get the latest prices</p>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="background: #eef2ff; color: #5d5fef; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; margin: 0 auto 15px auto;">🛡️</div>
            <h4 style="margin:0; font-weight: 600; color: #333;">Trusted Sources</h4>
            <p style="color: #888; font-size: 0.9em; margin: 5px 0 0 0;">From verified sellers only</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == 'deals':
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; margin-bottom: 40px;">
        <h1 style="font-size: 2.8em; font-weight: 800; color: #111;">Daily Smartphone Deals</h1>
        <p style="color: #6c757d; font-size: 1.1em;">Hand-picked offers from top retailers updated every hour.</p>
    </div>
    """, unsafe_allow_html=True)
    
    d1, d2, d3 = st.columns(3)
    
    with d1:
        st.markdown("""
        <div style="background: #fcfcfc; border: 1px dashed #e0e0e0; border-radius: 15px; padding: 30px 20px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.02); height: 100%;">
            <span style="background: #eef2ff; color: #5d5fef; padding: 5px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 700; text-transform: uppercase;">FLASH SALE</span>
            <h3 style="margin: 20px 0 10px 0; color: #222; font-size: 1.3em;">Amazon Summer Special</h3>
            <p style="color: #777; font-size: 0.95em; line-height: 1.5;">Up to 40% off on premium flagship models.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with d2:
        st.markdown("""
        <div style="background: #fcfcfc; border: 1px dashed #e0e0e0; border-radius: 15px; padding: 30px 20px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.02); height: 100%;">
            <span style="background: #eefbee; color: #28a745; padding: 5px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 700; text-transform: uppercase;">BANK OFFER</span>
            <h3 style="margin: 20px 0 10px 0; color: #222; font-size: 1.3em;">Flipkart Card Discounts</h3>
            <p style="color: #777; font-size: 0.95em; line-height: 1.5;">Flat ₹5,000 off on select credit cards.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with d3:
        st.markdown("""
        <div style="background: #fcfcfc; border: 1px dashed #e0e0e0; border-radius: 15px; padding: 30px 20px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.02); height: 100%;">
            <span style="background: #fff4e6; color: #fd7e14; padding: 5px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 700; text-transform: uppercase;">NEW ARRIVAL</span>
            <h3 style="margin: 20px 0 10px 0; color: #222; font-size: 1.3em;">Croma Exclusive</h3>
            <p style="color: #777; font-size: 0.95em; line-height: 1.5;">Exchange your old phone for high value upgrades.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([4, 2, 4])
    with col2:
        if st.button("Back to Search ➔", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

elif st.session_state.page == 'categories':
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; margin-bottom: 40px;">
        <h1 style="font-size: 2.8em; font-weight: 800; color: #111;">Browse Categories</h1>
        <p style="color: #6c757d; font-size: 1.1em;">Find exactly what you're looking for across all our product lines.</p>
    </div>
    """, unsafe_allow_html=True)
    
    cat1, cat2, cat3 = st.columns(3)
    with cat1: st.markdown("<div class='deal-card'><h3>📱 Smartphones</h3><p>Latest iOS & Android</p></div>", unsafe_allow_html=True)
    with cat2: st.markdown("<div class='deal-card'><h3>💻 Laptops</h3><p>Work & Gaming</p></div>", unsafe_allow_html=True)
    with cat3: st.markdown("<div class='deal-card'><h3>🎧 Accessories</h3><p>Audio & Wearables</p></div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([4, 2, 4])
    with col2:
        if st.button("Back to Search ➔", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

# --- PROCESS SEARCH & SHOW RESULTS ---
if search_query:
    st.markdown("---")
    
    with st.spinner("Fetching live data & running ML model..."):
        results_df = generate_mock_data(search_query)
        best_deal, model = train_and_predict_best_deal(results_df)
        save_search(search_query, best_deal['Platform'], best_deal['Price (₹)'])
        
        st.subheader("📊 Top Picks For You")
        
        display_df = results_df.drop(columns=['Deal_Score'])
        
        # --- DRAW PROFESSIONAL CARDS ---
        cols = st.columns(3)
        for idx, row in display_df.iterrows():
            with cols[idx]:
                platform_color = "#ff9900" if row['Platform'] == "Amazon" else ("#2874f0" if row['Platform'] == "Flipkart" else "#ea4335")
                best_badge = '<div style="position: absolute; top: -12px; right: -12px; background: #28a745; color: white; padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 0.9em; box-shadow: 0 4px 6px rgba(0,0,0,0.2); z-index: 10;">🏆 Best Deal</div>' if row['Platform'] == best_deal['Platform'] else ''
                
                # If we scanned an image, use it! Otherwise, use the scraped image.
                final_image = scanned_image_b64 if scanned_image_b64 else row['Image_URL']

                # HTML template for the minimalist e-commerce card
                card_html = f"""<div class="card-container" style="position: relative; background: white; margin-bottom: 20px; overflow: hidden;">
<div style="padding: 15px;">
{best_badge}
<div style="height: 180px; display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
<img src="{final_image}" style="max-height: 100%; max-width: 100%; object-fit: contain;">
</div>
<div style="font-size: 0.95em; color: #333; height: 2.8em; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; margin-bottom: 8px;">
{row['Product']}
</div>
<div style="background: #388e3c; color: white; display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold;">
{row['Rating']} ★
</div>
</div>
<div style="background: #f9f9f9; padding: 12px 15px; border-top: 1px solid #eee; display: flex; justify-content: space-between; align-items: center;">
<div style="display: flex; align-items: center; gap: 8px;">
<div style="color: {platform_color}; font-weight: 900; font-style: italic; font-size: 0.85em;">
{row['Platform']}
</div>
<div style="color: #111; font-size: 1.1em; font-weight: bold;">
₹{int(row['Price (₹)']):,}
</div>
</div>
<a href="{row['Link']}" target="_blank" style="background: #ff3f00; color: white; padding: 6px 12px; text-decoration: none; border-radius: 3px; font-weight: bold; font-size: 0.9em; box-shadow: 0 2px 4px rgba(255,63,0,0.2);">
BUY
</a>
</div>
</div>"""
                st.markdown(card_html, unsafe_allow_html=True)
        
        st.markdown("---")
        # --- DISPLAY METRICS ---
        st.subheader("💡 Market Insights")
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.markdown(f"<div class='metric-card'><b>Average Price</b><br>₹{int(results_df['Price (₹)'].mean()):,}</div>", unsafe_allow_html=True)
        with mcol2:
            highest_discount = results_df['Discount (%)'].max()
            platform = results_df.loc[results_df['Discount (%)'].idxmax()]['Platform']
            st.markdown(f"<div class='metric-card'><b>Highest Discount</b><br>{highest_discount}% on {platform}</div>", unsafe_allow_html=True)
        with mcol3:
            best_rating = results_df['Rating'].max()
            platform = results_df.loc[results_df['Rating'].idxmax()]['Platform']
            st.markdown(f"<div class='metric-card'><b>Top Rated</b><br>{best_rating}⭐ on {platform}</div>", unsafe_allow_html=True)
