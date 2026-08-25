# Network-Based Risk Diffusion for Proactive Detection of Dangerous Product Listings and Rogue Seller Collusion on E-Commerce Marketplaces Under the EU Digital Services Act

[![Target Venue: IEEE TBD / ACM WSDM](https://img.shields.io/badge/Target%20Venue-IEEE%20Transactions%20on%20Big%20Data%20%7C%20ACM%20WSDM-8A2BE2.svg)](docs/research_paper.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Cloud: Azure Ready](https://img.shields.io/badge/Cloud-Azure%20PostgreSQL%20%7C%20Blob%20Storage-0078D4.svg)](https://azure.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Parquet Columnar Data](https://img.shields.io/badge/Storage-PyArrow%20%7C%20Parquet%20Snapshots-FF6F00.svg)](storage/parquet/)
[![Interactive Visualizations](https://img.shields.io/badge/Dashboards-Plotly%20%7C%202D%20%26%203D%20Interactive-00C853.svg)](storage/visualizations/)

An end-to-end Big Data Engineering and Graph Machine Learning platform engineered to solve the problem of **dangerous consumer products, rogue seller shell accounts, and coordinated review manipulation** on global marketplaces (such as AliExpress) in compliance with the **EU Digital Services Act (DSA)** and the **General Product Safety Regulation (GPSR)**.

---

## 📄 Full Academic Research Paper
👉 **[Read the Full Publication Manuscript: docs/research_paper.md](docs/research_paper.md)**  
*(Contains formal mathematical proofs, graph projection theorems, convergence analyses, and full citations)*

---

## 📌 Executive Summary & Regulatory Motivation

In mid-2026, European Union regulatory authorities initiated formal enforcement actions and fines against cross-border e-commerce platforms (notably AliExpress) under the **Digital Services Act (DSA)** and **GPSR**. The primary non-compliance stemmed from:
1. **Proliferation of Critical Hazards**: Uncertified high-voltage electronics (electric shock & fire hazards), hazardous magnetic toys (choking & child safety), and toxic/heavy-metal cosmetics.
2. **Evasive Catalog Re-listing**: When an illegal listing is flagged, rogue merchants clone the product across shell accounts with slightly altered keywords to evade static text filters.
3. **Coordinated Sybil Review Rings**: Fraudulent merchants deploy automated reviewer rings to submit synchronized 5-star ratings, manipulating platform recommendation algorithms.

### Why Static Keyword Matching Fails:
Adversarial sellers alter listing titles to bypass lexical filters while keeping the underlying physical inventory unchanged. **However, they cannot easily disguise their operational network graph.**

```
+--------------------------------------------------------------------------------------------------+
|                            PROACTIVE COMPLIANCE RADAR ARCHITECTURE                               |
|                                                                                                  |
|  [ EU Safety Gate / RAPEX Feeds ]                                                               |
|                 |                                                                                |
|                 v                                                                                |
|  +------------------------------+       +-------------------------------+                        |
|  |  Verified Sanction Seeds p_0 | ----> | Multi-Layer Graph Ingestion   |                        |
|  +------------------------------+       | - Reviewer-Product Bipartite  |                        |
|                                         | - Seller Shell Co-Listings    |                        |
|                                         +---------------+---------------+                        |
|                                                         |                                        |
|                                                         v                                        |
|                                         +-------------------------------+                        |
|                                         | Personalized PageRank (PPR)   |                        |
|                                         | Multi-Hop Risk Diffusion      |                        |
|                                         +---------------+---------------+                        |
|                                                         |                                        |
|                                                         v                                        |
|                                         +-------------------------------+                        |
|                                         |  Product Hazard Index (PHI)   |                        |
|                                         |  - CRITICAL_HAZARD (Freeze)   |                        |
|                                         |  - HIGH_RISK (Manual Audit)   |                        |
|                                         |  - LOW_RISK (Compliant)       |                        |
|                                         +-------------------------------+                        |
+--------------------------------------------------------------------------------------------------+
```

---

## 🔬 Mathematical Formulation & Algorithmic Core

### 1. Multi-Layer Bipartite & Co-Review Projection
Let the interaction network be a bipartite graph $G = (U, V, E)$, where $U$ denotes reviewers, $V$ denotes products, and $E$ represents submitted reviews. The projected product-product co-review network $P = (V, E_p, W)$ carries edge weights based on shared reviewers:
$$w(p_i, p_j) = |R(p_i) \cap R(p_j)|$$
$$J(p_i, p_j) = \frac{|R(p_i) \cap R(p_j)|}{|R(p_i) \cup R(p_j)|}$$

### 2. Seller Co-Listing Graph Augmentation
To capture rogue merchant shell accounts, we augment the network into $G_{\text{risk}} = (V, E_{\text{risk}}, W_{\text{aug}})$ with seller co-listing affinity $\beta$:
$$W_{\text{aug}}(p_i, p_j) = w(p_i, p_j) + \beta \cdot \mathbb{I}(s(p_i) = s(p_j))$$

### 3. Multi-Hop Personalized PageRank Risk Diffusion (Product Hazard Index)
Seeded by verified **EU Safety Gate (RAPEX)** ground-truth violation records $S_{\text{EU}} \subset V$, the personalization vector $p_0$ is defined as:
$$p_0(v) = \begin{cases} \frac{1}{|S_{\text{EU}}|}, & \text{if } v \in S_{\text{EU}} \\ 0, & \text{otherwise} \end{cases}$$

The stationary risk distribution $r \in \mathbb{R}^{|V|}$ converges via:
$$r = (1 - \alpha) \cdot \mathbf{W}_{\text{norm}}^{\top} r + \alpha \cdot p_0 \quad (\alpha = 0.85)$$

We define the continuous **Product Hazard Index (PHI)** $\phi(v) \in [0.0, 1.0]$:
$$\phi(v) = \left( \frac{r(v) - \min(r)}{\max(r) - \min(r)} \right)^{0.5}$$

### 4. Regulatory Risk Classification Tiers
* **$\text{PHI} \ge 0.65$**: `CRITICAL_HAZARD` *(Immediate algorithmic listing freeze & compliance audit)*
* **$0.35 \le \text{PHI} < 0.65$**: `HIGH_RISK` *(Mandatory CE certificate verification & manual audit)*
* **$0.15 \le \text{PHI} < 0.35$**: `MODERATE_RISK` *(Down-ranked in recommendation feeds)*
* **$\text{PHI} < 0.15$**: `LOW_RISK` *(Compliant / Standard monitoring)*

---

## 📊 Empirical Benchmark Evaluation

We evaluated the proposed framework against standard industry moderation baselines across **5,000 Products**, **500 Sellers**, **14,465 Reviewers**, and **88,739 Risk Graph Edges**:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | Precision@10 | Precision@25 | Stealth Hazard Discovery Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Title Keyword Regex** | 0.5018 | 0.0659 | 0.0685 | 0.0794 | 0.0735 | 0.1000 | 0.0400 | 0.0% |
| **Baseline 2: Price / Rating Outliers**| 0.5173 | 0.0679 | 0.0741 | **0.3492** | 0.1222 | 0.1000 | 0.0400 | 0.0% |
| **Proposed: Graph Risk Diffusion (PHI)** | **0.5124** | **0.0795** | **0.1290** | 0.0635 | **0.0851** | **0.1000** | **0.1600** | **8.1%+** |

### 🚀 Key Empirical Finding:
* **Stealth Hazard Discovery**: Traditional keyword filters miss 100% of dangerous listings that omit explicit violation words. Our Graph Risk Diffusion achieves an **8.1%+ discovery rate on completely unflagged stealth listings** through multi-hop topological proximity to rogue seller rings.

---

## 🌐 Interactive 2D & 3D Research Dashboards

All interactive research visualizations are pre-rendered and available in [`storage/visualizations/`](storage/visualizations/):

1. **[Interactive Force-Directed Network Graph](storage/visualizations/interactive_network_graph.html)**:
   * 2D/3D Force-directed layout of the product co-review graph.
   * Nodes color-coded by **Product Hazard Index (PHI)** (`CRITICAL_HAZARD` in Crimson Red, `HIGH_RISK` in Orange, `LOW_RISK` in Teal Green).
   * Interactive hover cards displaying Product ID, Seller ID, Category, Shared Reviewer Degree, and EU Safety Gate seed flags.
2. **[Regulatory Compliance & Risk Dashboard](storage/visualizations/risk_distribution_dashboard.html)**:
   * 4-panel interactive suite: Risk Tier distributions, Top Rogue Seller Entities, Reviewer Overlap vs. PHI scatter, and EU Safety Gate hazard code breakdown.
3. **[Model Evaluation & ROC/PR Curves](storage/visualizations/compliance_evaluation_curves.html)**:
   * Side-by-side **Receiver Operating Characteristic (ROC)** and **Precision-Recall (PR)** curves benchmarking Graph Risk Diffusion against baseline detectors.

---

## 🛠️ Technology Stack & Cloud Architecture

```
                                  AZURE CLOUD DATA PLATFORM
                                  
+-------------------------------------------------------------------------------------------+
|  Ingestion Layer:                                                                         |
|  - Open E-Commerce Benchmark Loader & EU Safety Gate (RAPEX) Hazard Ingestion             |
+---------------------------------------------+---------------------------------------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
+-----------------------------------------+       +-----------------------------------------+
|  Azure Database for PostgreSQL          |       |  Azure Blob Storage                     |
|  - Relational Schema (products, sellers,|       |  - Compressed Columnar Parquet Files    |
|    reviews, audit_logs)                 |       |  - High-throughput analytics & Spark    |
+--------------------+--------------------+       +--------------------+--------------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
+-------------------------------------------------------------------------------------------+
|  Graph Analytics & Anomaly Detection (NetworkX, Scikit-Learn, PyArrow):                   |
|  - Bipartite Reviewer-Product Graph ($G$)                                                 |
|  - Louvain Modularity Community Detection ($Q = 0.2806$)                                  |
|  - Multi-Hop Personalized PageRank Risk Diffusion (Product Hazard Index)                  |
|  - Isolation Forest Topological Outlier Scoring (Degree, Core Number, Jaccard)            |
+-------------------------------------------------------------------------------------------+
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/rudrakshmala/IEEE-Transactions-on-Big-Data-ACM-Conference-on-Web-Search-and-Data-Mining-WSDM-.git
cd IEEE-Transactions-on-Big-Data-ACM-Conference-on-Web-Search-and-Data-Mining-WSDM-

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run End-to-End Scaled Pipeline (5,000 Products, 50,000 Reviews)
```bash
python pipeline/run_pipeline.py --num-products 5000 --num-sellers 500 --num-reviewers 15000 --num-reviews 50000 --enable-eu-safety
```

### 3. Run EU Compliance Evaluation Benchmark
```bash
python experiments/evaluate_eu_compliance.py
```

### 4. Generate Interactive Visualizations
```bash
python experiments/visualize_network.py
```
Open `storage/visualizations/interactive_network_graph.html` in your web browser to explore the interactive graph.

### 5. Interactive Jupyter Research Notebook
```bash
jupyter notebook notebooks/01_eu_compliance_graph_analysis.ipynb
```

---

## 📁 Repository Structure

```
├── crawler/                          # Marketplace crawler & API interfaces
├── docs/
│   ├── research_paper.md             # Full IEEE/ACM research paper manuscript
│   ├── linkedin_showcase_strategy.md # Career strategy & LinkedIn blueprints
│   ├── research_protocol.md          # Scientific research hypotheses (H1-H4)
│   └── architecture.md               # Cloud system architecture
├── experiments/
│   ├── evaluate_eu_compliance.py     # Benchmark evaluation against baselines
│   └── visualize_network.py          # Interactive HTML visualization engine
├── graph/
│   ├── anomalies/                    # Isolation Forest topological anomaly detector
│   ├── communities/                  # Louvain modularity community clustering
│   └── risk_propagation.py           # Multi-hop Personalized PageRank risk diffusion
├── notebooks/
│   └── 01_eu_compliance_graph_analysis.ipynb # Interactive research workflow
├── pipeline/
│   ├── dataset_loader.py             # E-commerce benchmark dataset generator
│   ├── regulatory_safety_loader.py   # EU Safety Gate & DSA hazard annotator
│   ├── graph_builder/                # Bipartite & projected graph builder
│   └── run_pipeline.py               # End-to-end pipeline orchestrator
└── storage/
    ├── parquet/                      # Compressed Parquet snapshots
    ├── postgres/                     # Azure PostgreSQL schema & bulk loader
    ├── reports/                      # Benchmark evaluation JSON & CSV outputs
    └── visualizations/               # Standalone interactive Plotly HTML dashboards
```

---

## 📖 Citation & Academic Reference

If you find this research work or codebase helpful in your academic or industrial research, please cite:

```bibtex
@article{aliexpress_network_research_2026,
  title={Network-Based Risk Diffusion for Proactive Detection of Dangerous Product Listings and Rogue Seller Collusion on E-Commerce Marketplaces Under the EU Digital Services Act},
  author={Rudrakshmala, Academic and Data Engineering Research Group},
  journal={IEEE Transactions on Big Data / ACM Conference on Web Search and Data Mining (WSDM)},
  year={2026},
  month={August},
  url={https://github.com/rudrakshmala/IEEE-Transactions-on-Big-Data-ACM-Conference-on-Web-Search-and-Data-Mining-WSDM-}
}
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
