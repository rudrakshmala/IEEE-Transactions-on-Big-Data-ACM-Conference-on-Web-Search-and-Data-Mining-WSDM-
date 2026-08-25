import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

# Ensure root directory in sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from pipeline.dataset_loader import DatasetLoader
from pipeline.regulatory_safety_loader import RegulatorySafetyLoader
from pipeline.graph_builder.bipartite_graph import BipartiteGraphBuilder
from graph.risk_propagation import GraphRiskPropagator


def evaluate_compliance_models(
    num_products: int = 1000,
    num_sellers: int = 120,
    num_reviewers: int = 4000,
    num_reviews: int = 15000,
    random_seed: int = 42,
):
    print("=" * 75)
    print("  EU REGULATORY COMPLIANCE & DANGEROUS PRODUCT DETECTION BENCHMARK")
    print("=" * 75)

    # 1. Generate & Annotate Data
    print("\n[1/4] Loading E-Commerce Marketplace & EU Safety Gate Ground Truth...")
    loader = DatasetLoader()
    df_p, df_s, df_r = loader.generate_benchmark_dataset(
        num_products=num_products,
        num_sellers=num_sellers,
        num_reviewers=num_reviewers,
        num_reviews=num_reviews,
        random_seed=random_seed,
    )

    safety_loader = RegulatorySafetyLoader(seed=random_seed)
    df_p, df_s = safety_loader.annotate_products_with_regulatory_risk(df_p, df_s)

    # Target ground-truth (Exclude known seeds to evaluate generalization to unflagged listings)
    unseeded_mask = ~df_p["is_known_eu_sanction"]
    y_true = df_p.loc[unseeded_mask, "is_true_dangerous"].astype(int).values

    # 2. Graph Construction & Risk Propagation
    print("\n[2/4] Constructing Multi-Layer Network & Running Graph Risk Diffusion...")
    builder = BipartiteGraphBuilder()
    B = builder.build_bipartite_graph(df_r, df_p)
    P = builder.project_product_network(B)

    propagator = GraphRiskPropagator()
    G_risk = propagator.build_augmented_risk_graph(P, df_p)
    df_scored = propagator.compute_product_hazard_index(G_risk, df_p)

    # 3. Model Predictions & Scores
    # Method A: Baseline Keyword Detection
    score_keyword = df_scored.loc[unseeded_mask, "keyword_risk_flag"].astype(float).values

    # Method B: Baseline Metadata Anomaly (Price dump + extreme rating variance)
    norm_price = (df_scored["price"] - df_scored["price"].mean()) / df_scored["price"].std()
    score_metadata = (df_scored["rating"] < 3.8).astype(float) + (norm_price < -0.8).astype(float)
    score_metadata = score_metadata.loc[unseeded_mask].values

    # Method C: Proposed Graph Risk Diffusion (PHI)
    score_graph_phi = df_scored.loc[unseeded_mask, "product_hazard_index"].values

    # 4. Compute Benchmark Metrics
    print("\n[3/4] Evaluating Detection Performance Metrics...")

    def compute_metrics(y_true, y_score, threshold=0.35):
        auc_roc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
        pr_auc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0

        # Binary thresholded metrics
        y_pred = (y_score >= threshold).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        # Precision@K
        k_values = [10, 25, 50]
        p_at_k = {}
        sorted_indices = np.argsort(-y_score)
        for k in k_values:
            top_k_indices = sorted_indices[:k]
            p_at_k[f"P@{k}"] = round(float(y_true[top_k_indices].mean()), 4)

        return {
            "ROC-AUC": round(float(auc_roc), 4),
            "PR-AUC": round(float(pr_auc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            **p_at_k,
        }

    metrics_keyword = compute_metrics(y_true, score_keyword, threshold=0.5)
    metrics_metadata = compute_metrics(y_true, score_metadata, threshold=1.0)
    metrics_graph = compute_metrics(y_true, score_graph_phi, threshold=0.25)

    # Stealth Detection Rate (% of dangerous products with NO keywords caught by Graph PHI)
    stealth_mask = unseeded_mask & df_p["is_true_dangerous"] & (~df_p["keyword_risk_flag"])
    stealth_caught = (df_scored.loc[stealth_mask, "product_hazard_index"] >= 0.25).sum()
    stealth_total = stealth_mask.sum()
    stealth_rate = (stealth_caught / stealth_total * 100.0) if stealth_total > 0 else 0.0

    # Build Comparison Table
    comparison_df = pd.DataFrame(
        [
            {"Model": "Baseline: Title Keyword Filter", **metrics_keyword},
            {"Model": "Baseline: Price/Rating Metadata", **metrics_metadata},
            {"Model": "Proposed: Graph Risk Diffusion (PHI)", **metrics_graph},
        ]
    )

    print("\n" + "=" * 75)
    print("  EXPERIMENT BENCHMARK RESULTS")
    print("=" * 75)
    print(comparison_df.to_string(index=False))
    print(f"\nStealth Hazard Discovery Rate (No keywords present): {stealth_caught}/{stealth_total} ({stealth_rate:.1f}%)")

    # Export results
    reports_dir = root_dir / "storage" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "eu_compliance_benchmark.json"

    benchmark_data = {
        "dataset_size": len(df_p),
        "total_true_dangerous": int(df_p["is_true_dangerous"].sum()),
        "known_seed_sanctions": int(df_p["is_known_eu_sanction"].sum()),
        "stealth_hazard_discovery_rate_pct": round(stealth_rate, 2),
        "models": comparison_df.to_dict(orient="records"),
    }

    with open(report_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\nBenchmark results saved to: {report_file}")
    return comparison_df


if __name__ == "__main__":
    evaluate_compliance_models()
