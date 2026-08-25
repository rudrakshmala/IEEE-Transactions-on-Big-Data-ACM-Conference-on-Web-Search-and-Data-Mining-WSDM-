from typing import Dict, List, Tuple
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter


class CommunityDetector:
    """
    Detects clusters of tightly-coupled products exhibiting abnormally high reviewer overlap
    using the Louvain modularity optimization algorithm.
    """

    def __init__(self, resolution: float = 1.0, random_state: int = 42):
        self.resolution = resolution
        self.random_state = random_state

    def detect_communities(self, P: nx.Graph) -> Tuple[Dict[str, int], pd.DataFrame]:
        """
        Executes Louvain community detection on the weighted product co-review graph.
        Returns:
            - node_to_community mapping
            - DataFrame with community-level metadata and suspiciousness metrics
        """
        if P.number_of_edges() == 0:
            print("[CommunityDetector] Warning: Graph has 0 edges. Assigning isolated communities.")
            node_to_comm = {node: idx for idx, node in enumerate(P.nodes())}
            df_comm = pd.DataFrame([{"community_id": idx, "size": 1} for idx in node_to_comm.values()])
            return node_to_comm, df_comm

        # Use NetworkX Louvain community detection
        communities = nx.community.louvain_communities(
            P,
            weight="weight",
            resolution=self.resolution,
            seed=self.random_state,
        )

        modularity = nx.community.modularity(P, communities, weight="weight")
        print(f"[CommunityDetector] Detected {len(communities)} communities (Modularity Q = {modularity:.4f})")

        node_to_comm = {}
        comm_stats = []

        for c_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                node_to_comm[node] = c_id

            # Analyze internal density and seller concentration
            subgraph = P.subgraph(comm_nodes)
            sellers = [P.nodes[n].get("seller_id") for n in comm_nodes if "seller_id" in P.nodes[n]]
            seller_counts = Counter(sellers)
            dominant_seller, top_seller_count = seller_counts.most_common(1)[0] if seller_counts else (None, 0)
            seller_concentration = (top_seller_count / len(comm_nodes)) if comm_nodes else 0.0

            # Internal edge weights (shared reviews)
            weights = [d.get("weight", 1) for _, _, d in subgraph.edges(data=True)]
            jaccards = [d.get("jaccard", 0.0) for _, _, d in subgraph.edges(data=True)]

            comm_stats.append(
                {
                    "community_id": c_id,
                    "num_products": len(comm_nodes),
                    "internal_edges": subgraph.number_of_edges(),
                    "density": round(nx.density(subgraph), 4),
                    "mean_shared_reviews": round(float(np.mean(weights)), 2) if weights else 0.0,
                    "max_shared_reviews": int(max(weights)) if weights else 0,
                    "mean_jaccard": round(float(np.mean(jaccards)), 4) if jaccards else 0.0,
                    "num_distinct_sellers": len(seller_counts),
                    "seller_concentration": round(seller_concentration, 3),
                    "dominant_seller_id": dominant_seller,
                }
            )

        df_communities = pd.DataFrame(comm_stats).sort_values(by="density", ascending=False)
        return node_to_comm, df_communities


if __name__ == "__main__":
    from pipeline.dataset_loader import DatasetLoader
    from pipeline.graph_builder.bipartite_graph import BipartiteGraphBuilder

    loader = DatasetLoader()
    df_p, df_s, df_r = loader.generate_benchmark_dataset(num_products=150, num_sellers=20, num_reviewers=600, num_reviews=2500)

    builder = BipartiteGraphBuilder()
    B = builder.build_bipartite_graph(df_r, df_p)
    P = builder.project_product_network(B)

    detector = CommunityDetector()
    node_map, df_comm = detector.detect_communities(P)
    print("\nTop 5 Communities by Density:")
    print(df_comm.head())
