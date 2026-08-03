# System Architecture

AliExpress Public Pages
|
v
Azure Container App (Crawler)
|
+------------------+
|                  |
v                  v
Azure Blob Storage     Azure PostgreSQL
(raw JSON/HTML)        (structured data)
|                  |
+--------+---------+
|
v
Graph Builder
(NetworkX / Neo4j)
|
v
Anomaly Detection
(Isolation Forest, Louvain, PageRank)
