import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Ensure root directory in sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from pipeline.dataset_loader import DatasetLoader
from pipeline.regulatory_safety_loader import RegulatorySafetyLoader
from pipeline.graph_builder.bipartite_graph import BipartiteGraphBuilder
from graph.communities.community_detection import CommunityDetector
from graph.risk_propagation import GraphRiskPropagator
from graph.anomalies.anomaly_detector import AnomalyDetector


class NetworkVisualizer:
    """
    Generates interactive HTML dashboards and publication-quality figures
    for the EU Regulatory Compliance and Graph Risk Diffusion research platform.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else root_dir / "storage" / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_interactive_network_graph(
        self,
        G_risk: nx.Graph,
        df_scored: pd.DataFrame,
        filename: str = "interactive_network_graph.html",
    ) -> Path:
        """
        Creates an interactive 2D Force-Directed Network Graph where node colors
        represent EU Product Hazard Index (PHI) and risk tiers.
        """
        print("[Visualizer] Computing 2D Spring Layout for network graph...")
        # Use spring layout with random seed for reproducibility
        pos = nx.spring_layout(G_risk, k=0.18, iterations=60, seed=42)

        # Build edge traces
        edge_x = []
        edge_y = []
        for u, v in G_risk.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.7, color="#d3d3d3"),
            hoverinfo="none",
            mode="lines",
            name="Shared Reviewer / Co-listing Edge",
        )

        # Map node attributes
        if "node_id" not in df_scored.columns:
            if "node" in df_scored.columns:
                df_scored["node_id"] = df_scored["node"]
            else:
                df_scored["node_id"] = [f"prod_{pid}" for pid in df_scored["product_id"]]

        node_lookup = df_scored.set_index("node_id").to_dict(orient="index")

        node_x = []
        node_y = []
        node_color = []
        node_size = []
        node_text = []

        color_map = {
            "CRITICAL_HAZARD": "#d90429",   # Vibrant Crimson Red
            "HIGH_RISK": "#f77f00",         # Orange
            "MODERATE_RISK": "#fcbf49",     # Amber Yellow
            "LOW_RISK": "#2a9d8f",          # Teal Green
        }

        for node in G_risk.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                info = node_lookup.get(node, {})
                phi = info.get("product_hazard_index", 0.0)
                tier = info.get("risk_tier", "LOW_RISK")
                pid = info.get("product_id", node)
                sid = info.get("seller_id", "N/A")
                cat = info.get("category", "N/A")
                deg = G_risk.degree(node)
                is_seed = info.get("is_known_eu_sanction", False)

                node_color.append(color_map.get(tier, "#2a9d8f"))
                node_size.append(max(6, min(24, int(deg ** 0.5 * 3 + 4))))

                seed_str = "<b>[EU SAFETY GATE SANCTION SEED]</b><br>" if is_seed else ""
                hover_label = (
                    f"{seed_str}"
                    f"<b>Product ID:</b> {pid}<br>"
                    f"<b>Seller ID:</b> {sid}<br>"
                    f"<b>Category:</b> {cat}<br>"
                    f"<b>Risk Tier:</b> {tier}<br>"
                    f"<b>Product Hazard Index (PHI):</b> {phi:.4f}<br>"
                    f"<b>Shared Review Degree:</b> {deg}"
                )
                node_text.append(hover_label)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(
                color=node_color,
                size=node_size,
                line=dict(width=1, color="#1e1e1e"),
            ),
            name="Products",
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="<b>Interactive Product Network & EU Risk Diffusion Radar</b><br><sup>Nodes colored by Product Hazard Index (PHI) | Red = Critical Hazard, Orange = High Risk, Green = Low Risk</sup>",
                titlefont_size=18,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=15, r=15, t=50),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8f9fa",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        out_path = self.output_dir / filename
        fig.write_html(str(out_path))
        print(f"[Visualizer] Saved interactive network graph to: {out_path}")
        return out_path

    def plot_evaluation_curves(
        self,
        df_scored: pd.DataFrame,
        filename: str = "compliance_evaluation_curves.html",
    ) -> Path:
        """
        Plots ROC Curves and Precision-Recall Curves comparing Graph Risk Diffusion vs. Baselines.
        """
        unseeded = df_scored[~df_scored["is_known_eu_sanction"]].copy()
        y_true = unseeded["is_true_dangerous"].astype(int).values

        # Scores
        scores = {
            "Graph Risk Diffusion (PHI)": unseeded["product_hazard_index"].values,
            "Baseline: Title Keyword Filter": unseeded.get("keyword_risk_flag", pd.Series([False]*len(unseeded))).astype(float).values,
            "Baseline: Price & Rating Anomaly": ((unseeded["rating"] < 3.8).astype(float) + (unseeded["price"] < unseeded["price"].median()).astype(float)).values,
        }

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "<b>Receiver Operating Characteristic (ROC) Curve</b>",
                "<b>Precision-Recall (PR) Curve</b>",
            ),
        )

        colors = ["#d90429", "#3a86ff", "#8338ec"]

        for (name, y_score), color in zip(scores.items(), colors):
            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC = {roc_auc:.3f})", line=dict(color=color, width=2.5)),
                row=1, col=1,
            )

            # Precision-Recall
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(rec, prec)
            fig.add_trace(
                go.Scatter(x=rec, y=prec, mode="lines", name=f"{name} (PR-AUC = {pr_auc:.3f})", line=dict(color=color, width=2.5, dash="dot" if "Baseline" in name else "solid"), showlegend=False),
                row=1, col=2,
            )

        # Baseline random line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(color="gray", dash="dash")),
            row=1, col=1,
        )

        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)

        fig.update_layout(
            title_text="<b>EU Regulatory Compliance Model Benchmark: Dangerous Product Detection</b>",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8f9fa",
            height=500,
            legend=dict(x=0.02, y=0.05, bgcolor="rgba(255,255,255,0.8)"),
        )

        out_path = self.output_dir / filename
        fig.write_html(str(out_path))
        print(f"[Visualizer] Saved evaluation curves to: {out_path}")
        return out_path

    def plot_risk_distribution_dashboard(
        self,
        df_scored: pd.DataFrame,
        filename: str = "risk_distribution_dashboard.html",
    ) -> Path:
        """
        Plots a multi-panel analytics dashboard showing risk tier breakdown,
        seller concentration, and reviewer overlap vs. hazard score.
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "domain"}],
            ],
            subplot_titles=(
                "<b>Product Hazard Index (PHI) Distribution by Risk Tier</b>",
                "<b>Top Rogue Seller Entities Flagged by Network Radar</b>",
                "<b>Reviewer Overlap vs. Product Hazard Index</b>",
                "<b>Hazard Code Breakdown (EU Safety Gate Taxonomies)</b>",
            ),
        )

        # Panel 1: Risk Tiers Bar Chart
        tier_counts = df_scored["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["risk_tier", "count"]
        color_discrete_map = {
            "CRITICAL_HAZARD": "#d90429",
            "HIGH_RISK": "#f77f00",
            "MODERATE_RISK": "#fcbf49",
            "LOW_RISK": "#2a9d8f",
        }
        fig.add_trace(
            go.Bar(
                x=tier_counts["risk_tier"],
                y=tier_counts["count"],
                marker_color=[color_discrete_map.get(t, "#2a9d8f") for t in tier_counts["risk_tier"]],
                text=tier_counts["count"],
                textposition="auto",
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Panel 2: Top Suspicious Sellers
        suspicious_sellers = df_scored[df_scored["risk_tier"].isin(["CRITICAL_HAZARD", "HIGH_RISK"])]
        seller_counts = suspicious_sellers["seller_id"].value_counts().head(8).reset_index()
        seller_counts.columns = ["seller_id", "high_risk_products"]
        fig.add_trace(
            go.Bar(
                x=[f"Store_{sid}" for sid in seller_counts["seller_id"]],
                y=seller_counts["high_risk_products"],
                marker_color="#e63946",
                text=seller_counts["high_risk_products"],
                textposition="auto",
                showlegend=False,
            ),
            row=1, col=2,
        )

        # Panel 3: Reviewer Overlap vs. PHI Scatter
        fig.add_trace(
            go.Scatter(
                x=df_scored["degree"],
                y=df_scored["product_hazard_index"],
                mode="markers",
                marker=dict(
                    size=7,
                    color=df_scored["product_hazard_index"],
                    colorscale="Viridis",
                    showscale=False,
                ),
                text=[f"Product: {pid}<br>Seller: {sid}" for pid, sid in zip(df_scored["product_id"], df_scored["seller_id"])],
                showlegend=False,
            ),
            row=2, col=1,
        )

        # Panel 4: Hazard Code Breakdown
        hazard_counts = df_scored[df_scored["hazard_code"] != "NONE"]["hazard_code"].value_counts().reset_index()
        hazard_counts.columns = ["hazard_code", "count"]
        fig.add_trace(
            go.Pie(
                labels=hazard_counts["hazard_code"],
                values=hazard_counts["count"],
                hole=0.4,
                showlegend=True,
            ),
            row=2, col=2,
        )

        fig.update_xaxes(title_text="Risk Classification Tier", row=1, col=1)
        fig.update_yaxes(title_text="Number of Listings", row=1, col=1)
        fig.update_xaxes(title_text="Seller Entity", row=1, col=2)
        fig.update_yaxes(title_text="Hazardous Product Count", row=1, col=2)
        fig.update_xaxes(title_text="Shared Reviewer Degree in Graph", row=2, col=1)
        fig.update_yaxes(title_text="Product Hazard Index (PHI)", row=2, col=1)

        fig.update_layout(
            title_text="<b>AliExpress EU Regulatory Compliance & Product Safety Dashboard</b>",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8f9fa",
            height=750,
        )

        out_path = self.output_dir / filename
        fig.write_html(str(out_path))
        print(f"[Visualizer] Saved risk distribution dashboard to: {out_path}")
        return out_path


def generate_all_visualizations(
    num_products: int = 500,
    num_sellers: int = 50,
    num_reviewers: int = 1500,
    num_reviews: int = 6000,
):
    print("=" * 70)
    print("  GENERATING INTERACTIVE RESEARCH VISUALIZATIONS & DASHBOARDS")
    print("=" * 70)

    loader = DatasetLoader()
    df_p, df_s, df_r = loader.generate_benchmark_dataset(
        num_products=num_products,
        num_sellers=num_sellers,
        num_reviewers=num_reviewers,
        num_reviews=num_reviews,
        random_seed=42,
    )

    safety_loader = RegulatorySafetyLoader(seed=42)
    df_p, df_s = safety_loader.annotate_products_with_regulatory_risk(df_p, df_s)

    builder = BipartiteGraphBuilder()
    B = builder.build_bipartite_graph(df_r, df_p)
    P = builder.project_product_network(B)

    propagator = GraphRiskPropagator()
    G_risk = propagator.build_augmented_risk_graph(P, df_p)
    df_scored = propagator.compute_product_hazard_index(G_risk, df_p)

    detector = CommunityDetector()
    node_to_comm, df_comm = detector.detect_communities(P)

    anomaly_detector = AnomalyDetector()
    df_features = anomaly_detector.extract_graph_features(P, node_to_comm, df_comm)
    cols_to_merge = [c for c in ["product_id", "price", "hazard_code", "is_known_eu_sanction", "is_true_dangerous", "keyword_risk_flag", "product_hazard_index", "risk_tier"] if c in df_scored.columns]
    df_full = df_features.merge(
        df_scored[cols_to_merge],
        on="product_id",
        how="left"
    )

    visualizer = NetworkVisualizer()
    p1 = visualizer.plot_interactive_network_graph(G_risk, df_full)
    p2 = visualizer.plot_evaluation_curves(df_full)
    p3 = visualizer.plot_risk_distribution_dashboard(df_full)

    print("\n" + "=" * 70)
    print("  ALL RESEARCH DASHBOARDS GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"1. Interactive Network Map: {p1}")
    print(f"2. Model Evaluation Curves: {p2}")
    print(f"3. Compliance Dashboard:    {p3}")


if __name__ == "__main__":
    generate_all_visualizations()
