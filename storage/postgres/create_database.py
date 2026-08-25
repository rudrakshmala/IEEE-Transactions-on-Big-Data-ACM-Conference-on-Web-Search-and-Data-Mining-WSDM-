from dotenv import load_dotenv
from pathlib import Path
import os
import psycopg2

# Load .env from project root
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")

try:
    conn = psycopg2.connect(
        host=HOST,
        port=PORT,
        dbname="postgres",
        user=USER,
        password=PASSWORD,
        sslmode="require"
    )

    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE DATABASE aliexpressdb;")

    print("Database 'aliexpressdb' created successfully!")

    cur.close()
    conn.close()

except psycopg2.errors.DuplicateDatabase:
    print("Database already exists.")

except Exception as e:
    print("Failed to create database.")
    print(e)