from pathlib import Path
from dotenv import load_dotenv
import os
import psycopg2

# Load .env
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

_conn = None


def get_connection():
    global _conn

    if _conn is None or _conn.closed != 0:
        _conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode="require",
        )

    return _conn


def insert_product(product):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO products (
                product_id,
                seller_id,
                category,
                title,
                price,
                rating,
                review_count,
                orders,
                url,
                first_seen,
                last_seen
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW())
            ON CONFLICT (product_id)
            DO UPDATE SET
                price = EXCLUDED.price,
                rating = EXCLUDED.rating,
                review_count = EXCLUDED.review_count,
                orders = EXCLUDED.orders,
                last_seen = NOW();
            """,
            (
                product["product_id"],
                product["seller_id"],
                product["category"],
                product["title"],
                product["price"],
                product["rating"],
                product["review_count"],
                product["orders"],
                product["url"],
            ),
        )

        conn.commit()

    except Exception as e:
        conn.rollback()
        print("DATABASE ERROR:", repr(e))
        raise

    finally:
        cur.close()