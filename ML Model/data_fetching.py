# data_fetching.py

import sqlite3
import pandas as pd

def fetch_data(database_file):
    conn = sqlite3.connect(database_file)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(query, conn)
    
    dataframes = {}
    for table in tables['name']:
        dataframes[table] = pd.read_sql(f"SELECT * FROM {table}", conn)
    
    df = pd.merge(dataframes['id_text'], dataframes['id_dialect'], on='id')
    return df
