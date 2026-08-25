import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Set
import networkx as nx
import pandas as pd
from itertools import combinations
from collections import defaultdict


class BipartiteGraphBuilder:
    """
    Constructs bipartite reviewer-product networks and projected co-review networks
    for detecting coordinated review manipulation rings.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.output_dir = Path(output_dir) if output_dir else self.root_dir / "storage" / "graphs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_bipartite_graph(
        self,
        df_reviews: pd.DataFrame,
        df_products: Optional[pd.DataFrame] = None,
    ) -> nx.Graph:
        """
        Constructs a bipartite graph G where:
        - Set 0: Reviewers (node IDs: usr_...)
        - Set 1: Products (node IDs: prod_...)
        """
        B = nx.Graph()

        # Add Product nodes with metadata
        if df_products is not None:
            for _, row in df_products.iterrows():
                p_node = f"prod_{row['product_id']}"
                B.add_node(
                    p_node,
                    bipartite=1,
                    type="product",
                    product_id=int(row["product_id"]),
                    seller_id=int(row.get("seller_id", 0)),
                    category=str(row.get("category", "")),
                    rating=float(row.get("rating", 0.0)),
                    review_count=int(row.get("review_count", 0)),
                )

        # Add Reviewer nodes and Review Edges
        for _, row in df_reviews.iterrows():
            u_node = f"usr_{row['reviewer_id']}" if not str(row['reviewer_id']).startswith("usr_") else str(row['reviewer_id'])
            p_node = f"prod_{row['product_id']}"

            if not B.has_node(u_node):
                B.add_node(u_node, bipartite=0, type="reviewer", reviewer_id=str(row['reviewer_id']))

            if not B.has_node(p_node):
                B.add_node(p_node, bipartite=1, type="product", product_id=int(row['product_id']))

            B.add_edge(
                u_node,
                p_node,
                rating=int(row.get("rating", 5)),
                review_date=str(row.get("review_date", "")),
                review_id=str(row.get("review_id", "")),
            )

        print(f"[GraphBuilder] Bipartite Graph built:")
        print(f"  -> Total Nodes: {B.number_of_nodes()} (Reviewers: {sum(1 for _, d in B.nodes(data=True) if d.get('bipartite') == 0)}, Products: {sum(1 for _, d in B.nodes(data=True) if d.get('bipartite') == 1)})")
        print(f"  -> Total Review Edges: {B.number_of_edges()}")

        return B

    def project_product_network(
        self,
        B: nx.Graph,
        min_shared_reviewers: int = 1,
    ) -> nx.Graph:
        """
        Projects bipartite graph B onto the Product set (Product-Product Co-Review Network).
        Edges represent shared reviewers, weighted by count and Jaccard similarity.
        """
        product_nodes = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 1}
        reviewer_to_products = defaultdict(set)

        for u, v in B.edges():
            u_node, p_node = (u, v) if B.nodes[u].get("bipartite") == 0 else (v, u)
            reviewer_to_products[u_node].add(p_node)

        # Count shared reviewers between product pairs
        pair_weights = defaultdict(int)
        for u_node, prods in reviewer_to_products.items():
            if len(prods) > 1:
                for p1, p2 in combinations(sorted(prods), 2):
                    pair_weights[(p1, p2)] += 1

        P = nx.Graph()

        # Add all product nodes with their original attributes
        for p in product_nodes:
            P.add_node(p, **B.nodes[p])

        # Add weighted edges
        for (p1, p2), shared_count in pair_weights.items():
            if shared_count >= min_shared_reviewers:
                deg1 = B.degree(p1)
                deg2 = B.degree(p2)
                union_size = deg1 + deg2 - shared_count
                jaccard = shared_count / union_size if union_size > 0 else 0.0

                P.add_edge(
                    p1,
                    p2,
                    weight=shared_count,
                    jaccard=round(jaccard, 4),
                )

        print(f"[GraphBuilder] Projected Product-Product Co-Review Network:")
        print(f"  -> Product Nodes: {P.number_of_nodes()}")
        print(f"  -> Shared-Reviewer Edges: {P.number_of_edges()}")

        return P

    def export_graph(self, G: nx.Graph, filename: str = "product_network.graphml") -> Path:
        """
        Exports the network to GraphML format for Gephi, Cytoscape, or NetworkX analysis.
        """
        export_path = self.output_dir / filename
        # Clean attributes for GraphML serialization
        G_clean = G.copy()
        for _, data in G_clean.nodes(data=True):
            for k, v in list(data.items()):
                if v is None:
                    data[k] = ""
        
        nx.write_graphml(G_clean, export_path)
        print(f"[GraphBuilder] Exported graph to {export_path}")
        return export_path


if __name__ == "__main__":
    from pipeline.dataset_loader import DatasetLoader

    loader = DatasetLoader()
    df_p, df_s, df_r = loader.generate_benchmark_dataset(num_products=200, num_sellers=30, num_reviewers=800, num_reviews=3000)

    builder = BipartiteGraphBuilder()
    B = builder.build_bipartite_graph(df_r, df_p)
    P = builder.project_product_network(B)
    builder.export_graph(P, "test_product_network.graphml")
