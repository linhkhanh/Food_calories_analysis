# db_utils.py
import sqlite3
import os

def clear_processed_nutrition(db_path: str = "database/nutrition_data.db") -> bool:
    """
    Clears all rows from the processed_nutrition table in SQLite.
    Returns True if successful, False if the database or table doesn't exist.
    """
    if not os.path.exists(db_path):
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM processed_nutrition")
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Error clearing table: {e}")
        return False
