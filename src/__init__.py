# src/__init__.py
from flask import Flask, jsonify, request
from src.config import config
from src.routes.camera_recognition import face_recognition_bp
from src.routes.register_face import register_bp
from src.db import get_db_connection
import requests
import sqlite3

def create_app(testing=False):
    app = Flask(__name__)
    # app.config.from_object(config)
    
    # Database configuration
    DATABASE = 'test_face_recognition.db' if testing else 'face_recognition.db'

    def init_db():
        """Initialize the database with required tables"""
        try:
            conn = get_db_connection(DATABASE)
            cursor = conn.cursor()

            # Create tbl_account table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tbl_account (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255),
                    role TEXT CHECK(role IN ('admin', 'user')),
                    name VARCHAR(100),
                    email VARCHAR(100)
                )
            ''')

            # Create tbl_register_faces table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tbl_register_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_image VARCHAR(255),
                    account_id INTEGER,
                    image_vector JSON,
                    image_vector_process JSON,
                    face_image_process VARCHAR(255),
                    FOREIGN KEY (account_id) REFERENCES tbl_account(id) ON DELETE CASCADE
                )
            ''')

            # Create tbl_enter_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tbl_enter_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    enter_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    account_id INTEGER,
                    face_image VARCHAR(255),
                    FOREIGN KEY (account_id) REFERENCES tbl_account(id) ON DELETE CASCADE
                )
            ''')

            conn.commit()
            print(f"Database {'test_' if testing else ''}initialized successfully")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            conn.close()

    # Initialize database
    with app.app_context():
        print(f"Creating {'test ' if testing else ''}database")
        init_db()

    # Example API endpoints
    @app.route('/api/accounts', methods=['GET'])
    def get_accounts():
        try:
            conn = get_db_connection(DATABASE)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tbl_account")
            accounts = [dict(row) for row in cursor.fetchall()]
            return jsonify(accounts)
        except sqlite3.Error as e:
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()

    @app.route('/api/accounts', methods=['POST'])
    def create_account():
        try:
            data = request.get_json()
            conn = get_db_connection(DATABASE)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tbl_account (username, password, role, name, email)
                VALUES (?, ?, ?, ?, ?)
            """, (
                data['username'],
                data['password'],
                data['role'],
                data.get('name'),
                data.get('email')
            ))
            conn.commit()
            return jsonify({"message": "Account created", "id": cursor.lastrowid}), 201
        except sqlite3.Error as e:
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()

    @app.route('/api/test-db', methods=['GET'])
    def test_db():
        try:
            conn = get_db_connection(DATABASE)
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version();")
            version = cursor.fetchone()
            return jsonify({"sqlite_version": version[0], "status": "Database connected"})
        except sqlite3.Error as e:
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()

    # Register blueprints
    app.register_blueprint(face_recognition_bp, url_prefix='/api')
    app.register_blueprint(register_bp, url_prefix='/api')
    
    return app

__all__ = ['create_app', 'get_db_connection']