import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from pipeline.dataset_loader import DatasetLoader
from pipeline.regulatory_safety_loader import RegulatorySafetyLoader
from pipeline.graph_builder.bipartite_graph import BipartiteGraphBuilder
from graph.communities.community_detection import CommunityDetector
from graph.anomalies.anomaly_detector import AnomalyDetector
from graph.risk_propagation import GraphRiskPropagator


def run_pipeline(args):
    print("=" * 70)
    print("  ALIEXPRESS NETWORK RESEARCH PLATFORM - END-TO-END PIPELINE")
    print("=" * 70)

    start_total_time = time.time()

    # Step 1: Data Ingestion / Generation
    print("\n[Step 1/5] Ingesting Research Dataset...")
    loader = DatasetLoader(output_dir=str(root_dir / "storage" / "parquet"))
    
    if args.load_cached:
        print(f"Loading cached dataset from storage/parquet/ (prefix: {args.prefix})...")
        df_products, df_sellers, df_reviews = loader.load_from_parquet(prefix=args.prefix)
    else:
        print(f"Generating benchmark research dataset (Products: {args.num_products}, Sellers: {args.num_sellers}, Reviews: {args.num_reviews}, Anomaly Ratio: {args.anomaly_ratio})...")
        df_products, df_sellers, df_reviews = loader.generate_benchmark_dataset(
            num_products=args.num_products,
            num_sellers=args.num_sellers,
            num_reviewers=args.num_reviewers,
            num_reviews=args.num_reviews,
            anomaly_ratio=args.anomaly_ratio,
        )

    # Optional EU Regulatory Risk Ingestion
    if args.enable_eu_safety:
        print("\n[Regulatory Risk] Ingesting EU Safety Gate (RAPEX) / DSA Ground-Truth Fines...")
        safety_loader = RegulatorySafetyLoader(seed=42)
        df_products, df_sellers = safety_loader.annotate_products_with_regulatory_risk(df_products, df_sellers)

    print(f"Dataset Summary: {len(df_products):,} Products | {len(df_sellers):,} Sellers | {len(df_reviews):,} Reviews")

    # Step 2: Parquet Storage
    if args.save_parquet:
        print("\n[Step 2/5] Persisting datasets to Parquet format...")
        loader.save_to_parquet(df_products, df_sellers, df_reviews, prefix=args.prefix)

    # Optional Step: PostgreSQL Sync
    if args.sync_db:
        print("\n[Database Sync] Syncing records to Azure Database for PostgreSQL...")
        try:
            from storage.postgres.bulk_loader import PostgresBulkLoader
            db_loader = PostgresBulkLoader()
            db_loader.load_all(df_products, df_sellers, df_reviews)
        except Exception as e:
            print(f"[Warning] PostgreSQL sync skipped or encountered an error: {e}")

    # Step 3: Graph Construction
    print("\n[Step 3/5] Building Bipartite Reviewer-Product Graph & Co-Review Projection...")
    builder = BipartiteGraphBuilder(output_dir=str(root_dir / "storage" / "graphs"))
    B = builder.build_bipartite_graph(df_reviews, df_products)
    P = builder.project_product_network(B, min_shared_reviewers=args.min_shared)

    if args.export_graph:
        graphml_path = builder.export_graph(P, f"{args.prefix}_product_network.graphml")

    # Optional Step: EU Risk Diffusion
    if args.enable_eu_safety:
        print("\n[Graph Risk Diffusion] Propagating Regulatory Hazard Index across Network...")
        propagator = GraphRiskPropagator()
        G_risk = propagator.build_augmented_risk_graph(P, df_products)
        df_products = propagator.compute_product_hazard_index(G_risk, df_products)

    # Step 4: Community Detection
    print("\n[Step 4/5] Running Louvain Modularity Community Detection...")
    detector = CommunityDetector(resolution=args.resolution)
    node_to_comm, df_communities = detector.detect_communities(P)

    # Step 5: Feature Extraction & Anomaly Detection
    print("\n[Step 5/5] Extracting Graph Topological Features & Running Isolation Forest...")
    anomaly_detector = AnomalyDetector(contamination=args.contamination)
    df_features = anomaly_detector.extract_graph_features(P, node_to_comm, df_communities)
    
    # Merge Product Hazard Index if computed
    if args.enable_eu_safety and "product_hazard_index" in df_products.columns:
        df_features = df_features.merge(
            df_products[["product_id", "hazard_code", "is_known_eu_sanction", "is_true_dangerous", "product_hazard_index", "risk_tier"]],
            on="product_id",
            how="left"
        )

    df_scored = anomaly_detector.detect_anomalies(df_features)

    # Export Research Report
    reports_dir = root_dir / "storage" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_csv = reports_dir / f"{args.prefix}_anomaly_results.csv"
    df_scored.to_csv(report_csv, index=False)

    top_suspicious_sellers = df_scored[df_scored["is_suspicious"]]["seller_id"].value_counts().head(5).to_dict()

    summary_json = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_products": len(df_products),
        "total_sellers": len(df_sellers),
        "total_reviews": len(df_reviews),
        "graph_product_nodes": P.number_of_nodes(),
        "graph_shared_review_edges": P.number_of_edges(),
        "total_communities_detected": len(df_communities),
        "total_suspicious_products_flagged": int(df_scored["is_suspicious"].sum()),
        "top_suspicious_seller_clusters": top_suspicious_sellers,
        "eu_safety_enabled": args.enable_eu_safety,
        "results_file": str(report_csv),
    }

    if args.enable_eu_safety and "risk_tier" in df_scored.columns:
        summary_json["eu_risk_tiers"] = df_scored["risk_tier"].value_counts().to_dict()

    report_summary_file = reports_dir / f"{args.prefix}_summary.json"
    with open(report_summary_file, "w") as f:
        json.dump(summary_json, f, indent=2)

    print("\n" + "=" * 70)
    print("  RESEARCH PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"Total Execution Time: {time.time() - start_total_time:.2f} seconds")
    print(f"Results CSV: {report_csv}")
    print(f"Summary JSON: {report_summary_file}")
    print("\nTop 5 Flagged Suspicious Products:")
    cols_to_print = ["product_id", "seller_id", "rating", "degree", "max_overlap", "mean_jaccard", "anomaly_score"]
    if args.enable_eu_safety and "product_hazard_index" in df_scored.columns:
        cols_to_print += ["product_hazard_index", "risk_tier"]
    print(df_scored[cols_to_print].head(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="AliExpress Network Research Data & Graph Analytics Pipeline")
    parser.add_argument("--num-products", type=int, default=1000, help="Number of products to generate/simulate")
    parser.add_argument("--num-sellers", type=int, default=150, help="Number of sellers")
    parser.add_argument("--num-reviewers", type=int, default=4000, help="Number of unique reviewers")
    parser.add_argument("--num-reviews", type=int, default=15000, help="Total number of reviews")
    parser.add_argument("--anomaly-ratio", type=float, default=0.15, help="Proportion of products in coordinated review rings")
    parser.add_argument("--min-shared", type=int, default=1, help="Minimum shared reviewers to form an edge")
    parser.add_argument("--resolution", type=float, default=1.0, help="Louvain community resolution parameter")
    parser.add_argument("--contamination", type=float, default=0.15, help="Isolation Forest anomaly contamination rate")
    parser.add_argument("--prefix", type=str, default="research_run", help="Dataset prefix name")
    parser.add_argument("--save-parquet", action="store_true", default=True, help="Save Parquet files")
    parser.add_argument("--load-cached", action="store_true", help="Load cached Parquet instead of regenerating")
    parser.add_argument("--sync-db", action="store_true", help="Bulk upsert data into Azure PostgreSQL")
    parser.add_argument("--export-graph", action="store_true", default=True, help="Export GraphML graph representation")
    parser.add_argument("--enable-eu-safety", action="store_true", default=True, help="Enable EU Safety Gate risk diffusion")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

