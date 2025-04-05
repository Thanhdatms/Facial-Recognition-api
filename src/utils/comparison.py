import numpy as np
from flask import Flask, jsonify, request
import sqlite3
import requests


def compare_embedding(query_embedding, reference_embeddings, threshold=0.5):
    """So sánh embedding với các vector cụ thể và trả về tên người dùng nếu khớp."""
    
    for username, embedding in reference_embeddings.items():
        distance = np.linalg.norm(query_embedding - embedding)
        print(f"Distance to {username}: {distance}")
        
        if distance < threshold:
            return username  # Trả về tên người dùng nếu khoảng cách nhỏ hơn ngưỡng
    
    return None 
        
