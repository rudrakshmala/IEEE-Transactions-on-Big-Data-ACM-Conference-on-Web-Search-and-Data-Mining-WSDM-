from dotenv import load_dotenv
from pathlib import Path
import os
import psycopg2

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    sslmode="require"
)

cur = conn.cursor()

cur.execute("""
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public';
""")

tables = cur.fetchall()

print("Tables in database:")

for t in tables:
    print("-", t[0])

cur.close()
conn.close()