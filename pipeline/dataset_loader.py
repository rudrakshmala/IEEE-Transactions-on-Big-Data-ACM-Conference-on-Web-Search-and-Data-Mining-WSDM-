import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class DatasetLoader:
    """
    Loads open e-commerce research datasets or generates realistic benchmark
    data simulating AliExpress products, sellers, and reviewer networks for graph analytics.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.root_dir = Path(__file__).resolve().parents[1]
        self.output_dir = Path(output_dir) if output_dir else self.root_dir / "storage" / "parquet"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_benchmark_dataset(
        self,
        num_products: int = 1000,
        num_sellers: int = 150,
        num_reviewers: int = 5000,
        num_reviews: int = 20000,
        anomaly_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generates realistic e-commerce network data with both organic behavior and
        injected anomalous review rings (coordinated review manipulation).
        """
        np.random.seed(random_seed)
        random.seed(random_seed)

        categories = [
            "Consumer Electronics",
            "Phones & Telecommunications",
            "Computer & Office",
            "Home & Garden",
            "Automotive",
            "Tools & Home Improvement",
            "Sports & Entertainment",
            "Jewelry & Accessories",
        ]

        countries = ["US", "ES", "FR", "BR", "RU", "PL", "DE", "IT", "KR", "MX"]

        # 1. Generate Sellers
        seller_ids = [1000000 + i for i in range(num_sellers)]
        sellers_data = []
        now = datetime.utcnow()

        for s_id in seller_ids:
            first_seen = now - timedelta(days=random.randint(60, 1000))
            sellers_data.append(
                {
                    "seller_id": s_id,
                    "store_name": f"Store_{s_id}_Official",
                    "country": random.choice(countries),
                    "followers": int(np.random.exponential(scale=5000) + 50),
                    "store_rating": round(float(np.clip(np.random.normal(4.6, 0.3), 3.0, 5.0)), 2),
                    "total_products": random.randint(10, 300),
                    "first_seen": first_seen,
                    "last_seen": now,
                }
            )
        df_sellers = pd.DataFrame(sellers_data)

        # 2. Generate Products
        product_ids = [20000000 + i for i in range(num_products)]
        products_data = []

        for p_id in product_ids:
            seller_id = random.choice(seller_ids)
            category = random.choice(categories)
            price = round(float(np.random.exponential(scale=25.0) + 1.99), 2)
            first_seen = now - timedelta(days=random.randint(30, 800))

            products_data.append(
                {
                    "product_id": p_id,
                    "seller_id": seller_id,
                    "category": category,
                    "title": f"{category} Premium Item {p_id}",
                    "price": price,
                    "rating": round(float(np.clip(np.random.normal(4.5, 0.4), 1.0, 5.0)), 2),
                    "review_count": 0,
                    "orders": int(np.random.exponential(scale=800) + 10),
                    "url": f"https://www.aliexpress.com/item/{p_id}.html",
                    "first_seen": first_seen,
                    "last_seen": now,
                }
            )
        df_products = pd.DataFrame(products_data)

        # 3. Reviewers & Review Generation (Organic + Injected Sybil/Spam Rings)
        reviewer_ids = [f"usr_{300000 + i}" for i in range(num_reviewers)]
        
        num_sybil_reviewers = int(num_reviewers * anomaly_ratio)
        sybil_reviewers = reviewer_ids[:num_sybil_reviewers]
        organic_reviewers = reviewer_ids[num_sybil_reviewers:]

        num_target_products = int(num_products * anomaly_ratio)
        target_products = product_ids[:num_target_products]
        organic_products = product_ids[num_target_products:]

        prod_seller_map = dict(zip(df_products["product_id"], df_products["seller_id"]))

        reviews_data = []
        review_counter = 1

        num_organic_reviews = int(num_reviews * (1 - anomaly_ratio))
        organic_prod_weights = np.random.pareto(a=1.5, size=len(organic_products))
        organic_prod_weights /= organic_prod_weights.sum()

        for _ in range(num_organic_reviews):
            p_id = int(np.random.choice(organic_products, p=organic_prod_weights))
            u_id = random.choice(organic_reviewers)
            rating = int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.05, 0.10, 0.30, 0.50]))
            days_ago = random.randint(1, 180)
            r_date = (now - timedelta(days=days_ago)).date()

            reviews_data.append(
                {
                    "review_id": f"rev_{review_counter}",
                    "product_id": p_id,
                    "seller_id": int(prod_seller_map[p_id]),
                    "reviewer_id": u_id,
                    "rating": rating,
                    "review_text": f"Sample verified review for product {p_id}. Good product.",
                    "review_date": r_date,
                    "country": random.choice(countries),
                    "photo_count": random.choice([0, 0, 0, 1, 2, 3]),
                    "helpful_votes": random.choice([0, 0, 0, 1, 5]),
                    "crawl_time": now,
                }
            )
            review_counter += 1

        num_sybil_reviews = num_reviews - num_organic_reviews
        num_clusters = min(5, max(2, len(target_products) // 10))
        prod_clusters = np.array_split(target_products, num_clusters)
        user_clusters = np.array_split(sybil_reviewers, num_clusters)

        for c_idx in range(num_clusters):
            c_prods = prod_clusters[c_idx]
            c_users = user_clusters[c_idx]
            if len(c_prods) == 0 or len(c_users) == 0:
                continue

            cluster_reviews_count = num_sybil_reviews // num_clusters
            burst_date = now - timedelta(days=random.randint(5, 45))

            for _ in range(cluster_reviews_count):
                p_id = int(random.choice(c_prods))
                u_id = random.choice(c_users)
                rating = int(np.random.choice([4, 5], p=[0.08, 0.92]))
                time_delta_days = random.randint(0, 4)
                r_date = (burst_date + timedelta(days=time_delta_days)).date()

                reviews_data.append(
                    {
                        "review_id": f"rev_{review_counter}",
                        "product_id": p_id,
                        "seller_id": int(prod_seller_map[p_id]),
                        "reviewer_id": u_id,
                        "rating": rating,
                        "review_text": f"Outstanding product {p_id}! Fast shipping, highly recommend.",
                        "review_date": r_date,
                        "country": random.choice(countries),
                        "photo_count": random.choice([0, 1, 2]),
                        "helpful_votes": random.randint(0, 3),
                        "crawl_time": now,
                    }
                )
                review_counter += 1

        df_reviews = pd.DataFrame(reviews_data)
        df_reviews = df_reviews.drop_duplicates(subset=["product_id", "reviewer_id"])

        review_stats = df_reviews.groupby("product_id").agg(
            review_count=("review_id", "count"),
            avg_rating=("rating", "mean")
        ).reset_index()

        df_products = df_products.merge(review_stats, on="product_id", how="left")
        if "review_count_y" in df_products.columns:
            df_products["review_count"] = df_products["review_count_y"].fillna(0).astype(int)
        if "avg_rating" in df_products.columns:
            df_products["rating"] = df_products["avg_rating"].fillna(df_products["rating"]).round(2)
        df_products = df_products.drop(columns=["review_count_x", "review_count_y", "avg_rating"], errors="ignore")

        return df_products, df_sellers, df_reviews

    def save_to_parquet(
        self,
        df_products: pd.DataFrame,
        df_sellers: pd.DataFrame,
        df_reviews: pd.DataFrame,
        prefix: str = "snapshot",
    ) -> Dict[str, Path]:
        """
        Saves DataFrames into compressed Parquet files for big data queries and Spark.
        """
        paths = {
            "products": self.output_dir / f"{prefix}_products.parquet",
            "sellers": self.output_dir / f"{prefix}_sellers.parquet",
            "reviews": self.output_dir / f"{prefix}_reviews.parquet",
        }

        df_products.to_parquet(paths["products"], index=False, engine="pyarrow", compression="snappy")
        df_sellers.to_parquet(paths["sellers"], index=False, engine="pyarrow", compression="snappy")
        df_reviews.to_parquet(paths["reviews"], index=False, engine="pyarrow", compression="snappy")

        print(f"[DatasetLoader] Saved Parquet datasets to {self.output_dir}")
        for k, p in paths.items():
            print(f"  -> {k}: {p.name} ({p.stat().st_size / 1024:.1f} KB)")

        return paths

    def load_from_parquet(self, prefix: str = "snapshot") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads cached datasets from Parquet.
        """
        p_prod = self.output_dir / f"{prefix}_products.parquet"
        p_sell = self.output_dir / f"{prefix}_sellers.parquet"
        p_rev = self.output_dir / f"{prefix}_reviews.parquet"

        if not (p_prod.exists() and p_sell.exists() and p_rev.exists()):
            raise FileNotFoundError(f"Parquet files with prefix '{prefix}' not found in {self.output_dir}")

        df_products = pd.read_parquet(p_prod)
        df_sellers = pd.read_parquet(p_sell)
        df_reviews = pd.read_parquet(p_rev)

        return df_products, df_sellers, df_reviews


if __name__ == "__main__":
    loader = DatasetLoader()
    print("Generating benchmark e-commerce dataset for graph research...")
    df_p, df_s, df_r = loader.generate_benchmark_dataset(
        num_products=500,
        num_sellers=80,
        num_reviewers=2000,
        num_reviews=8000,
        anomaly_ratio=0.2,
    )
    print(f"Generated: {len(df_p)} products, {len(df_s)} sellers, {len(df_r)} reviews")
    loader.save_to_parquet(df_p, df_s, df_r)
