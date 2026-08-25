from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from project root
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "sslmode": "require"
}

# Initial seed categories
CATEGORY_URLS = [
    "https://www.aliexpress.com/category/100003109/women-clothing.html",
    "https://www.aliexpress.com/category/200000297/consumer-electronics.html",
    "https://www.aliexpress.com/category/1509/home-garden.html"
]

HEADLESS = True
REQUEST_DELAY = 2