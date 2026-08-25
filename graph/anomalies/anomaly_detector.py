from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Extracts graph topological and metadata features to detect coordinated review manipulation
    and suspicious seller networks using unsupervised Isolation Forest.
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=150,
        )
        self.scaler = StandardScaler()

    def extract_graph_features(
        self,
        P: nx.Graph,
        node_to_comm: Optional[Dict[str, int]] = None,
        df_comm: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Extracts structural network features for every product node in the projected network.
        """
        clustering = nx.clustering(P, weight="weight")
        core_numbers = nx.core_number(P) if P.number_of_edges() > 0 else {n: 0 for n in P.nodes()}
        pagerank = nx.pagerank(P, weight="weight") if P.number_of_edges() > 0 else {n: 1.0 / max(1, len(P)) for n in P.nodes()}

        comm_density_map = {}
        comm_conc_map = {}
        if df_comm is not None:
            comm_density_map = dict(zip(df_comm["community_id"], df_comm["density"]))
            comm_conc_map = dict(zip(df_comm["community_id"], df_comm["seller_concentration"]))

        rows = []
        for node in P.nodes():
            data = P.nodes[node]
            neighbors = list(P.neighbors(node))
            deg = len(neighbors)

            if deg > 0:
                weights = [P[node][nbr].get("weight", 1) for nbr in neighbors]
                jaccards = [P[node][nbr].get("jaccard", 0.0) for nbr in neighbors]
                weighted_deg = sum(weights)
                max_overlap = max(weights)
                mean_jaccard = float(np.mean(jaccards))
                max_jaccard = float(max(jaccards))
            else:
                weighted_deg = 0
                max_overlap = 0
                mean_jaccard = 0.0
                max_jaccard = 0.0

            comm_id = node_to_comm.get(node, -1) if node_to_comm else -1

            rows.append(
                {
                    "node": node,
                    "product_id": data.get("product_id", node),
                    "seller_id": data.get("seller_id", 0),
                    "category": data.get("category", ""),
                    "rating": data.get("rating", 0.0),
                    "review_count": data.get("review_count", 0),
                    "degree": deg,
                    "weighted_degree": weighted_deg,
                    "max_overlap": max_overlap,
                    "mean_jaccard": round(mean_jaccard, 4),
                    "max_jaccard": round(max_jaccard, 4),
                    "clustering_coeff": round(clustering.get(node, 0.0), 4),
                    "core_number": core_numbers.get(node, 0),
                    "pagerank": round(pagerank.get(node, 0.0), 6),
                    "community_id": comm_id,
                    "community_density": comm_density_map.get(comm_id, 0.0),
                    "seller_concentration": comm_conc_map.get(comm_id, 0.0),
                }
            )

        df_features = pd.DataFrame(rows)
        return df_features

    def detect_anomalies(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Trains Isolation Forest on topological graph features and outputs anomaly scores.
        """
        feature_cols = [
            "degree",
            "weighted_degree",
            "max_overlap",
            "mean_jaccard",
            "max_jaccard",
            "clustering_coeff",
            "core_number",
            "pagerank",
            "community_density",
            "seller_concentration",
            "rating",
        ]

        X = df_features[feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        # Isolation Forest prediction: -1 = anomaly, 1 = normal
        predictions = self.model.fit_predict(X_scaled)
        # Decision function: lower values mean more anomalous
        scores = self.model.decision_function(X_scaled)

        df_result = df_features.copy()
        df_result["anomaly_prediction"] = predictions
        df_result["is_suspicious"] = df_result["anomaly_prediction"] == -1
        df_result["anomaly_score"] = np.round(scores, 4)

        # Sort with most suspicious (lowest decision score) first
        df_result = df_result.sort_values(by="anomaly_score", ascending=True)

        num_anomalies = int(df_result["is_suspicious"].sum())
        print(f"[AnomalyDetector] Anomaly detection complete: {num_anomalies}/{len(df_result)} products flagged as suspicious.")

        return df_result


if __name__ == "__main__":
    from pipeline.dataset_loader import DatasetLoader
    from pipeline.graph_builder.bipartite_graph import BipartiteGraphBuilder
    from graph.communities.community_detection import CommunityDetector

    loader = DatasetLoader()
    df_p, df_s, df_r = loader.generate_benchmark_dataset(num_products=200, num_sellers=30, num_reviewers=800, num_reviews=3000, anomaly_ratio=0.2)

    builder = BipartiteGraphBuilder()
    B = builder.build_bipartite_graph(df_r, df_p)
    P = builder.project_product_network(B)

    detector = CommunityDetector()
    node_map, df_comm = detector.detect_communities(P)

    anomaly_detector = AnomalyDetector(contamination=0.15)
    df_features = anomaly_detector.extract_graph_features(P, node_map, df_comm)
    df_scored = anomaly_detector.detect_anomalies(df_features)

    print("\nTop 10 Most Suspicious Products:")
    print(df_scored[["product_id", "seller_id", "rating", "max_overlap", "mean_jaccard", "community_density", "anomaly_score", "is_suspicious"]].head(10))
