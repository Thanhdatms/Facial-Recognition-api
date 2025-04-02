from flask import Flask, jsonify, request
from src.config import config
from src.routes.camera_recognition import face_recognition_bp
import sqlite3
from src import create_app 

def create_database():
    app = Flask(__name__)
    # app.config.from_object(config)
    # Database configuration
    DATABASE = 'face_recognition.db'

    def get_db_connection():
        """Create and return a database connection"""
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row  # This allows accessing columns by name
        return conn

    def init_db():
        """Initialize the database with required tables"""
        try:
            conn = get_db_connection()
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
            print("Database initialized successfully")
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            conn.close()

    # Call init_db() when the app is created
    with app.app_context():
        print('database create')
        init_db()

create_database()

app = create_app()  # Initialize Flask app

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)