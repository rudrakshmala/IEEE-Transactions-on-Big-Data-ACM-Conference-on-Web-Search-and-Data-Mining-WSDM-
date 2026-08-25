from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd


class GraphRiskPropagator:
    """
    Implements Personalized PageRank (PPR) and multi-hop graph risk diffusion
    to propagate regulatory hazard scores from known EU-sanctioned products
    across shared reviewer rings and seller co-listing networks.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        max_iter: int = 200,
        tol: float = 1e-6,
        seller_link_weight: float = 2.5,
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.seller_link_weight = seller_link_weight

    def build_augmented_risk_graph(
        self,
        P: nx.Graph,
        df_products: pd.DataFrame,
    ) -> nx.Graph:
        """
        Augments the co-review product graph with seller co-listing edges,
        connecting products listed by the same merchant to model shell accounts.
        """
        G_risk = P.copy()

        # Group products by seller
        seller_to_prods = df_products.groupby("seller_id")["product_id"].apply(list).to_dict()

        added_seller_edges = 0
        for s_id, prods in seller_to_prods.items():
            if len(prods) > 1:
                node_list = [f"prod_{p}" for p in prods if f"prod_{p}" in G_risk]
                for i in range(len(node_list)):
                    for j in range(i + 1, len(node_list)):
                        u, v = node_list[i], node_list[j]
                        if G_risk.has_edge(u, v):
                            # Boost existing edge weight
                            G_risk[u][v]["weight"] = G_risk[u][v].get("weight", 1.0) + self.seller_link_weight
                        else:
                            G_risk.add_edge(u, v, weight=self.seller_link_weight, edge_type="seller_co_listing")
                            added_seller_edges += 1

        print(f"[GraphRiskPropagator] Augmented Graph for Risk Diffusion:")
        print(f"  -> Total Nodes: {G_risk.number_of_nodes()}")
        print(f"  -> Total Risk Diffusion Edges: {G_risk.number_of_edges()} (added {added_seller_edges} seller co-listing edges)")

        return G_risk

    def compute_product_hazard_index(
        self,
        G_risk: nx.Graph,
        df_products: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Runs Personalized PageRank seeded by known EU regulatory sanctions
        to compute the Product Hazard Index (PHI) for every listing.
        """
        # Identify seed nodes (products with confirmed EU sanctions / Safety Gate alerts)
        seed_nodes = set()
        for _, row in df_products.iterrows():
            if row.get("is_known_eu_sanction", False):
                node_name = f"prod_{row['product_id']}"
                if node_name in G_risk:
                    seed_nodes.add(node_name)

        if not seed_nodes:
            print("[GraphRiskPropagator] Warning: No seed sanction nodes found. Uniform risk assumed.")
            personalization = {n: 1.0 / len(G_risk) for n in G_risk.nodes()}
        else:
            personalization = {n: (1.0 / len(seed_nodes) if n in seed_nodes else 0.0) for n in G_risk.nodes()}

        print(f"[GraphRiskPropagator] Propagating risk from {len(seed_nodes)} EU sanction seeds...")

        # Compute Personalized PageRank risk scores
        try:
            ppr_scores = nx.pagerank(
                G_risk,
                alpha=self.alpha,
                personalization=personalization,
                weight="weight",
                max_iter=self.max_iter,
                tol=self.tol,
            )
        except Exception as e:
            print(f"[GraphRiskPropagator] Fallback to uniform due to: {e}")
            ppr_scores = {n: 0.0 for n in G_risk.nodes()}

        # Normalize scores to 0.0 - 1.0 Product Hazard Index (PHI)
        raw_values = np.array(list(ppr_scores.values()))
        min_val, max_val = raw_values.min(), raw_values.max()

        if max_val > min_val:
            # Min-Max scaling with power law scaling to highlight high-risk clusters
            normalized_scores = {
                k: round(float(((v - min_val) / (max_val - min_val)) ** 0.5), 4)
                for k, v in ppr_scores.items()
            }
        else:
            normalized_scores = {k: 0.0 for k in ppr_scores.keys()}

        # Merge PHI back into the products DataFrame
        df_out = df_products.copy()
        df_out["node_id"] = [f"prod_{pid}" for pid in df_out["product_id"]]
        df_out["raw_ppr_score"] = df_out["node_id"].map(ppr_scores).fillna(0.0)
        df_out["product_hazard_index"] = df_out["node_id"].map(normalized_scores).fillna(0.0)

        # Categorize into Regulatory Risk Tiers
        def get_risk_tier(phi):
            if phi >= 0.65:
                return "CRITICAL_HAZARD"
            elif phi >= 0.35:
                return "HIGH_RISK"
            elif phi >= 0.15:
                return "MODERATE_RISK"
            else:
                return "LOW_RISK"

        df_out["risk_tier"] = df_out["product_hazard_index"].apply(get_risk_tier)
        df_out = df_out.sort_values(by="product_hazard_index", ascending=False)

        tier_counts = df_out["risk_tier"].value_counts().to_dict()
        print(f"[GraphRiskPropagator] Risk Diffusion Complete. Risk Tiers: {tier_counts}")

        return df_out


if __name__ == "__main__":
    from pipeline.dataset_loader import DatasetLoader
    from pipeline.regulatory_safety_loader import RegulatorySafetyLoader
    from pipeline.graph_builder.bipartite_graph import BipartiteGraphBuilder

    loader = DatasetLoader()
    df_p, df_s, df_r = loader.generate_benchmark_dataset(num_products=300, num_sellers=40, num_reviewers=1000, num_reviews=4000)

    safety_loader = RegulatorySafetyLoader()
    df_p, df_s = safety_loader.annotate_products_with_regulatory_risk(df_p, df_s)

    builder = BipartiteGraphBuilder()
    B = builder.build_bipartite_graph(df_r, df_p)
    P = builder.project_product_network(B)

    propagator = GraphRiskPropagator()
    G_risk = propagator.build_augmented_risk_graph(P, df_p)
    df_scored = propagator.compute_product_hazard_index(G_risk, df_p)

    print("\nTop 10 High-Risk Listings Identified by Network Diffusion:")
    print(df_scored[["product_id", "seller_id", "category", "hazard_code", "is_known_eu_sanction", "product_hazard_index", "risk_tier"]].head(10))
