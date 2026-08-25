import os
import time
from pathlib import Path
from typing import Optional
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)


class PostgresBulkLoader:
    """
    High-performance batch loader for Azure Database for PostgreSQL.
    """

    def __init__(self):
        self.host = os.getenv("DB_HOST")
        self.port = os.getenv("DB_PORT", "5432")
        self.dbname = os.getenv("DB_NAME")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")

    def get_connection(self):
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            sslmode="require",
        )

    def load_sellers(self, df_sellers: pd.DataFrame, page_size: int = 1000) -> int:
        """
        Inserts or updates seller records in bulk.
        """
        records = [
            (
                int(row["seller_id"]),
                str(row["store_name"]),
                str(row.get("country", "")),
                int(row.get("followers", 0)),
                float(row.get("store_rating", 0.0)),
                int(row.get("total_products", 0)),
                row["first_seen"],
                row["last_seen"],
            )
            for _, row in df_sellers.iterrows()
        ]

        query = """
            INSERT INTO sellers (
                seller_id, store_name, country, followers, store_rating, total_products, first_seen, last_seen
            ) VALUES %s
            ON CONFLICT (seller_id) DO UPDATE SET
                followers = EXCLUDED.followers,
                store_rating = EXCLUDED.store_rating,
                total_products = EXCLUDED.total_products,
                last_seen = EXCLUDED.last_seen;
        """

        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                execute_values(cur, query, records, page_size=page_size)
            conn.commit()
            print(f"[PostgresBulkLoader] Successfully upserted {len(records)} sellers.")
            return len(records)
        except Exception as e:
            conn.rollback()
            print(f"[PostgresBulkLoader] Error inserting sellers: {e}")
            raise
        finally:
            conn.close()

    def load_products(self, df_products: pd.DataFrame, page_size: int = 1000) -> int:
        """
        Inserts or updates product records in bulk.
        """
        records = [
            (
                int(row["product_id"]),
                int(row["seller_id"]),
                str(row.get("category", "")),
                str(row.get("title", "")),
                float(row.get("price", 0.0)),
                float(row.get("rating", 0.0)),
                int(row.get("review_count", 0)),
                int(row.get("orders", 0)),
                str(row.get("url", "")),
                row["first_seen"],
                row["last_seen"],
            )
            for _, row in df_products.iterrows()
        ]

        query = """
            INSERT INTO products (
                product_id, seller_id, category, title, price, rating, review_count, orders, url, first_seen, last_seen
            ) VALUES %s
            ON CONFLICT (product_id) DO UPDATE SET
                price = EXCLUDED.price,
                rating = EXCLUDED.rating,
                review_count = EXCLUDED.review_count,
                orders = EXCLUDED.orders,
                last_seen = EXCLUDED.last_seen;
        """

        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                execute_values(cur, query, records, page_size=page_size)
            conn.commit()
            print(f"[PostgresBulkLoader] Successfully upserted {len(records)} products.")
            return len(records)
        except Exception as e:
            conn.rollback()
            print(f"[PostgresBulkLoader] Error inserting products: {e}")
            raise
        finally:
            conn.close()

    def load_reviews(self, df_reviews: pd.DataFrame, page_size: int = 2000) -> int:
        """
        Inserts review records in bulk.
        """
        records = [
            (
                str(row["review_id"]),
                int(row["product_id"]),
                int(row["seller_id"]),
                str(row["reviewer_id"]),
                int(row["rating"]),
                str(row.get("review_text", "")),
                row["review_date"],
                str(row.get("country", "")),
                int(row.get("photo_count", 0)),
                int(row.get("helpful_votes", 0)),
                row["crawl_time"],
            )
            for _, row in df_reviews.iterrows()
        ]

        query = """
            INSERT INTO reviews (
                review_id, product_id, seller_id, reviewer_id, rating, review_text, review_date, country, photo_count, helpful_votes, crawl_time
            ) VALUES %s
            ON CONFLICT (review_id) DO NOTHING;
        """

        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                execute_values(cur, query, records, page_size=page_size)
            conn.commit()
            print(f"[PostgresBulkLoader] Successfully inserted {len(records)} reviews.")
            return len(records)
        except Exception as e:
            conn.rollback()
            print(f"[PostgresBulkLoader] Error inserting reviews: {e}")
            raise
        finally:
            conn.close()

    def load_all(self, df_products: pd.DataFrame, df_sellers: pd.DataFrame, df_reviews: pd.DataFrame):
        t0 = time.time()
        print("[PostgresBulkLoader] Starting database bulk ingestion...")
        self.load_sellers(df_sellers)
        self.load_products(df_products)
        self.load_reviews(df_reviews)
        print(f"[PostgresBulkLoader] Bulk ingestion completed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    from pipeline.dataset_loader import DatasetLoader

    loader = DatasetLoader()
    print("Generating sample data for bulk ingestion...")
    df_p, df_s, df_r = loader.generate_benchmark_dataset(num_products=100, num_sellers=20, num_reviewers=500, num_reviews=1500)
    
    bulk_loader = PostgresBulkLoader()
    try:
        bulk_loader.load_all(df_p, df_s, df_r)
    except Exception as e:
        print(f"Skipping live DB test if credentials not active: {e}")
