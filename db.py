import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "pricelens.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT,
            best_platform TEXT,
            best_price REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_search(product_name, best_platform, best_price):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO search_history (product_name, best_platform, best_price, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (product_name, best_platform, best_price, timestamp))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_NAME)
    # Check if table exists before querying
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='search_history'")
    if not c.fetchone():
        conn.close()
        return pd.DataFrame()
        
    df = pd.read_sql_query("SELECT * FROM search_history ORDER BY id DESC", conn)
    conn.close()
    return df
