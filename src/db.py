# src/db.py
import sqlite3

def get_db_connection(db_path='face_recognition.db'):
    """Create and return a database connection"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    return conn