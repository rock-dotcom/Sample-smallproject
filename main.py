from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import os
import pandas as pd
import base64
import json
from db import init_db, save_search, get_history
from data_engine import generate_mock_data
from ml_model import train_and_predict_best_deal
from vision import identify_product_from_image
from dotenv import load_dotenv

load_dotenv(override=True)
gemini_api_key = os.environ.get("GEMINI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/history")
async def fetch_history():
    df = get_history()
    return df.to_dict(orient="records")

@app.post("/api/search")
async def search(query: str = Form(...)):
    results_df = generate_mock_data(query)
    best_deal, model = train_and_predict_best_deal(results_df)
    
    # Extract values safely to python native types
    best_platform = str(best_deal['Platform'])
    best_price = float(best_deal['Price (₹)'])
    
    save_search(query, best_platform, best_price)
    
    return {
        "results": json.loads(results_df.to_json(orient="records")),
        "best_deal": json.loads(best_deal.to_json())
    }

@app.post("/api/scan")
async def scan(file: UploadFile = File(...)):
    contents = await file.read()
    detected_product = identify_product_from_image(contents, gemini_api_key)
    
    if "Error:" in detected_product:
        return {"error": detected_product}
    
    results_df = generate_mock_data(detected_product)
    best_deal, model = train_and_predict_best_deal(results_df)
    
    # Extract values safely
    best_platform = str(best_deal['Platform'])
    best_price = float(best_deal['Price (₹)'])
    
    save_search(detected_product, best_platform, best_price)
    
    return {
        "detected_product": detected_product,
        "results": json.loads(results_df.to_json(orient="records")),
        "best_deal": json.loads(best_deal.to_json())
    }

@app.post("/api/clear-history")
async def clear_history():
    import sqlite3
    conn = sqlite3.connect("pricelens.db")
    conn.cursor().execute("DELETE FROM search_history")
    conn.commit()
    return {"status": "success"}

if __name__ == "__main__":
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"
        
    print("PriceLens starting at:")
    print(" - Local:   http://localhost:8000")
    print(f" - Network: http://{local_ip}:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
