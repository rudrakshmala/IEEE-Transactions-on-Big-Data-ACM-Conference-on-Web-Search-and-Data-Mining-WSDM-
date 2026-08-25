import random
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class RegulatorySafetyLoader:
    """
    Simulates and loads EU Safety Gate (RAPEX) and EU Digital Services Act (DSA)
    regulatory violations, product hazard codes, and enforcement ground truth.
    """

    HAZARD_CATEGORIES = {
        "ELEC_SHOCK": {
            "name": "Electric Shock & Fire Hazard",
            "categories": ["Consumer Electronics", "Phones & Telecommunications", "Tools & Home Improvement"],
            "risk_weight": 0.95,
            "keywords": ["charger", "adapter", "power supply", "heating", "laser", "battery"],
        },
        "CHEM_TOXIC": {
            "name": "Chemical & Toxic Substance Hazard",
            "categories": ["Jewelry & Accessories", "Home & Garden", "Beauty & Health"],
            "risk_weight": 0.90,
            "keywords": ["whitening", "slimming", "heavy metal", "lead", "cadmium", "phthalates"],
        },
        "CHOKE_CHILD": {
            "name": "Choking & Child Safety Hazard",
            "categories": ["Toys & Hobbies", "Mother & Kids", "Sports & Entertainment"],
            "risk_weight": 0.85,
            "keywords": ["small parts", "magnetic balls", "balloon", "rattle", "baby"],
        },
        "UNCERTIFIED_MED": {
            "name": "Uncertified Medical & Health Claims",
            "categories": ["Beauty & Health", "Consumer Electronics"],
            "risk_weight": 0.80,
            "keywords": ["cure", "therapy", "diagnose", "miracle", "antiviral", "sterilizer"],
        },
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def annotate_products_with_regulatory_risk(
        self,
        df_products: pd.DataFrame,
        df_sellers: pd.DataFrame,
        known_violation_rate: float = 0.05,
        hidden_violation_rate: float = 0.10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Annotates product and seller datasets with:
        1. Known EU Safety Gate Ground-Truth Fines (Seed Violations)
        2. Hidden/Dormant Dangerous Listings (Linked via rogue seller/reviewer networks)
        3. Baseline text-based risk keywords
        """
        df_p = df_products.copy()
        df_s = df_sellers.copy()

        # Identify rogue seller clusters (sellers operating shell stores with dangerous goods)
        num_rogue_sellers = max(2, int(len(df_s) * 0.12))
        rogue_seller_ids = set(df_s["seller_id"].sample(n=num_rogue_sellers, random_state=self.seed))

        df_s["is_rogue_seller"] = df_s["seller_id"].isin(rogue_seller_ids)

        # Assign product hazard classifications
        hazard_types = list(self.HAZARD_CATEGORIES.keys())
        p_hazard_labels = []
        is_known_sanctioned = []
        is_true_dangerous = []
        text_keyword_flag = []

        for _, row in df_p.iterrows():
            seller_id = row["seller_id"]
            category = row["category"]
            title = str(row["title"]).lower()

            is_rogue = seller_id in rogue_seller_ids

            if is_rogue:
                # Rogue sellers have a 70% probability of listing non-compliant/hazardous items
                is_danger = np.random.rand() < 0.70
            else:
                # Normal sellers have low 3% accidental non-compliance
                is_danger = np.random.rand() < 0.03

            if is_danger:
                hazard_code = random.choice(hazard_types)
                # Subset of dangerous products were already audited/fined by EU Safety Gate (Ground Truth Seeds)
                is_seed_fine = np.random.rand() < (known_violation_rate / (known_violation_rate + hidden_violation_rate))
                
                # Check if title contains obvious keywords (for baseline keyword detector)
                hazard_keywords = self.HAZARD_CATEGORIES[hazard_code]["keywords"]
                has_keyword = any(kw in title for kw in hazard_keywords) or (np.random.rand() < 0.35)
            else:
                hazard_code = "NONE"
                is_seed_fine = False
                has_keyword = np.random.rand() < 0.05  # False positive keyword noise

            p_hazard_labels.append(hazard_code)
            is_known_sanctioned.append(is_seed_fine)
            is_true_dangerous.append(is_danger)
            text_keyword_flag.append(has_keyword)

        df_p["hazard_code"] = p_hazard_labels
        df_p["is_known_eu_sanction"] = is_known_sanctioned  # Seeds for risk propagation
        df_p["is_true_dangerous"] = is_true_dangerous        # Ground truth for evaluation
        df_p["keyword_risk_flag"] = text_keyword_flag        # Baseline comparison

        num_seeds = int(df_p['is_known_eu_sanction'].sum())
        num_true_danger = int(df_p['is_true_dangerous'].sum())
        print(f"[RegulatorySafetyLoader] Regulatory Risk Annotation Complete:")
        print(f"  -> Total Products: {len(df_p)}")
        print(f"  -> Known EU Safety Gate Fines (Seeds): {num_seeds}")
        print(f"  -> Total True Dangerous Products: {num_true_danger} ({num_true_danger - num_seeds} stealth/unflagged)")
        print(f"  -> Rogue Seller Entities Identified: {len(rogue_seller_ids)}")

        return df_p, df_s


if __name__ == "__main__":
    from pipeline.dataset_loader import DatasetLoader

    loader = DatasetLoader()
    df_p, df_s, df_r = loader.generate_benchmark_dataset(num_products=300, num_sellers=40, num_reviewers=1000, num_reviews=4000)

    safety_loader = RegulatorySafetyLoader()
    df_p_annotated, df_s_annotated = safety_loader.annotate_products_with_regulatory_risk(df_p, df_s)
    print("\nSample Annotated Products:")
    print(df_p_annotated[["product_id", "seller_id", "category", "hazard_code", "is_known_eu_sanction", "is_true_dangerous"]].head(10))
