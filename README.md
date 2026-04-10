# STAR: Spatial Temporal Automated Risk system

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/React-18+-61DAFB.svg?logo=react" alt="React Platform"/>
  <img src="https://img.shields.io/badge/PyTorch-TGNN-EE4C2C.svg?logo=pytorch" alt="PyTorch Framework"/>
  <img src="https://img.shields.io/badge/Neo4j-Graph_DB-008CC1.svg?logo=neo4j" alt="Neo4j Layer"/>
  <img src="https://img.shields.io/badge/AWS-Bedrock%20LLMs-FF9900.svg?logo=amazonaws" alt="AWS Bedrock Hub"/>
</p>

## 🚀 Overview

The ** Spatial Temporal Automated Risk system(STAR)** system is an advanced Intelligent Fund Flow Tracking (IFFT) platform and Anti-Money Laundering (AML) simulator. It represents a paradigm shift from legacy, siloed, and reactive enterprise compliance systems (e.g., standard rule-based scenarios) to **Contextual Proactive Intelligence**.

By integrating **Temporal Graph Neural Networks (TGNNs)**, real-time graph visualization via **Neo4j**, and an **Agentic AML Orchestrator** powered by **AWS Bedrock / LLMs**, FCCI automatically flags complex typologies—ranging from rapid layering to circular fund routing—while fully resolving evidence and drafting Suspicious Activity Reports (SARs) via AI.

## ✨ Core Features

*   **Agentic Investigation Hub:** An LLM-orchestrated agent loop (utilizing models like Claude via Bedrock or Hugging Face APIs) that dynamically paginates through network evidence, explores fund flows logically, and auto-drafts SAR narratives.
*   **Temporal Graph Processing (TGNN):** A hybrid engine measuring both the spatial network topology (who connects to whom) and temporal velocity (how fast funds are moving), overcoming the blind spots of singular rule-based detection.
*   **Real-Time Explorable Subgraphs:** Uses Neo4j to store relational memory. Transactions are modeled as nodes, letting investigators trace deep multi-hop "fund momentum" pathways visually.
*   **Typology Detection:** Captures rapid layering, round-tripping, smurfing (structuring), and dormant account activation instantly.
*   **High-Fidelity Command Center:** A sleek React/Vite/TypeScript frontend rendering high-performance force-directed graph UI (via WebSockets/Socket.io), enabling proactive intervention.

---

## 🏗️ Architecture

1.  **Tier 1: Unified Data Fabric (Ingestion)**
    Real-time streaming and ingestion of financial transfers, performing live identity resolution to create "Golden Entities."
2.  **Tier 2: Graph Intelligence Engine (TGNN Core)**
    Hybrid LSTM-GNN architecture that tracks "Risk Momentum Scores" across specific paths, assigning spatial and temporal weights to transactions.
3.  **Tier 3: Agentic Detection Hub**
    A Co-Investigator AI using conversational multi-turn workflows to justify alerts, trace poisoned paths, and offer SHAP/LIME-based insights under "Right to Explanation" framework.

---

## 💻 Tech Stack

*   **Backend:** Python, FastAPI, WebSockets (`socket.io`), PyTorch.
*   **Frontend:** React, TypeScript, Vite, D3/Force-Graph.
*   **Database Engine:** Neo4j (Graph logic & pathfinding queries).
*   **AI/Inference:** LLMs (AWS Bedrock / Hugging Face Spaces routing), LangChain/Custom Agents.

---

## ⚙️ Quick Start

### 1. Prerequisites
*   [Neo4j Desktop](https://neo4j.com/download/) (Ensure bolt is running on `bolt://localhost:7687`)
*   Python 3.11+ and Node.js 18+

### 2. Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt # (or using uv ecosystem)

# Generate synthetic transaction patterns (poisoned typologies)
python generate_test_data.py

# Launch the FastAPI orchestrator and Backend Engine
uvicorn server:app --port 8000 --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
*Navigate to `http://localhost:5173` to access the Investigator Command Center.*

---

## 🧪 Validated Topologies (MVB Simulation)

| Typology Pattern | Target State TGNN Approach | Prototype Cypher Validation |
| :--- | :--- | :--- |
| **Circular Flow** | Cycle-Detection Motifs | `MATCH (p)-[*2..5]->(p)` |
| **Rapid Layering** | Temporal Momentum Analysis | `duration.between(in, out) < 30m` |
| **Structuring** | Sequential Clustering | `$50k Vol via <$10k chunks` |
| **Dormant Burst** | LSTM Anomaly Burst | `>180d Gap AND Z-Score > 3.0` |

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
