import numpy as np
from flask import Flask, jsonify, request
import sqlite3
import requests

def compare_embedding(query_embedding, reference_embedding, threshold=0.5):
    """So sánh embedding với vector cụ thể."""
    distance = np.linalg.norm(query_embedding - reference_embedding)
    print(f"Distance to specific embedding: {distance}")
    if distance < threshold:
        return "Match"
    else:
        return "No Match"
    
