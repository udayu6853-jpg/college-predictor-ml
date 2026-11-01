import os
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row

load_dotenv("c:/Users/udayu/OneDrive/Desktop/COLLEGE-PREDICTOR/.env")
db_url = os.getenv("DATABASE_URL")

print("=" * 60)
print("SMART COLLEGE PREDICTOR - DATABASE VIEWER")
print("=" * 60)

with psycopg.connect(db_url, row_factory=dict_row) as conn:
    with conn.cursor() as cur:
        # 1. Users Table
        print("\n[TABLE 1: users]")
        print("-" * 60)
        cur.execute("SELECT id, name, email, google_id, created_at FROM users ORDER BY id ASC")
        users = cur.fetchall()
        for u in users:
            print(f"ID: {u['id']} | Name: {u['name']} | Email: {u['email']} | Google ID: {u['google_id'] or 'None'} | Created: {u['created_at']}")
        
        # 2. Prediction History Table
        print("\n[TABLE 2: prediction_history (Recent Searches)]")
        print("-" * 60)
        cur.execute("""
            SELECT p.id, p.user_id, u.name as user_name, u.email as user_email,
                   p.exam, p.rank, p.branch, p.category, p.created_at
            FROM prediction_history p
            LEFT JOIN users u ON p.user_id = u.id
            ORDER BY p.id DESC
            LIMIT 10
        """)
        searches = cur.fetchall()
        if not searches:
            print("No prediction searches recorded yet.")
        for s in searches:
            print(f"Search #{s['id']} | User: {s['user_name']} ({s['user_email']}) | Exam: {s['exam']} | Rank: {s['rank']} | Branch: {s['branch']} | Category: {s['category']} | Date: {s['created_at']}")

        # 3. Saved Colleges Table
        print("\n[TABLE 3: saved_colleges (Bookmarked Colleges)]")
        print("-" * 60)
        cur.execute("""
            SELECT s.id, s.user_id, u.name, s.exam, s.college_index, s.created_at
            FROM saved_colleges s
            LEFT JOIN users u ON s.user_id = u.id
            ORDER BY s.id DESC
            LIMIT 10
        """)
        saved = cur.fetchall()
        if not saved:
            print("No saved colleges bookmarked yet.")
        for sc in saved:
            print(f"Saved #{sc['id']} | User: {sc['name']} | Exam: {sc['exam']} | College Index: {sc['college_index']} | Date: {sc['created_at']}")

print("\n" + "=" * 60)
