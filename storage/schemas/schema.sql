CREATE TABLE products (
product_id BIGINT PRIMARY KEY,
seller_id BIGINT,
category TEXT,
title TEXT,
price NUMERIC,
rating FLOAT,
review_count INT,
orders INT,
url TEXT,
first_seen TIMESTAMP,
last_seen TIMESTAMP
);

CREATE TABLE sellers (
seller_id BIGINT PRIMARY KEY,
store_name TEXT,
country TEXT,
followers INT,
store_rating FLOAT,
total_products INT,
first_seen TIMESTAMP,
last_seen TIMESTAMP
);

CREATE TABLE reviews (
review_id TEXT PRIMARY KEY,
product_id BIGINT,
seller_id BIGINT,
reviewer_id TEXT,
rating INT,
review_text TEXT,
review_date DATE,
country TEXT,
photo_count INT,
helpful_votes INT,
crawl_time TIMESTAMP
);
