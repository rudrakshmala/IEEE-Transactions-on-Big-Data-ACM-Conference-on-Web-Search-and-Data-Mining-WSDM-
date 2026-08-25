from dotenv import load_dotenv
from pathlib import Path
import os
import psycopg2

# Load .env
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
DATABASE = os.getenv("DB_NAME")
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")

# Read schema file
schema_path = Path(__file__).resolve().parents[1] / "schemas" / "schema.sql"

with open(schema_path, "r") as f:
    schema_sql = f.read()

try:
    conn = psycopg2.connect(
        host=HOST,
        port=PORT,
        dbname=DATABASE,
        user=USER,
        password=PASSWORD,
        sslmode="require"
    )

    cur = conn.cursor()

    cur.execute(schema_sql)

    conn.commit()

    print("Schema deployed successfully!")

    cur.close()
    conn.close()

except Exception as e:
    print("Schema deployment failed.")
    print(e)