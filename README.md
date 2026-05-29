# STAR: Spatial Temporal Automated Risk system

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/React-18+-61DAFB.svg?logo=react" alt="React Platform"/>
  <img src="https://img.shields.io/badge/PyTorch-Geometric-EE4C2C.svg?logo=pytorch" alt="PyTorch Framework"/>
  <img src="https://img.shields.io/badge/Neo4j-Aura_Cloud-008CC1.svg?logo=neo4j" alt="Neo4j Layer"/>
</p>

## 🚀 Overview

The **Spatial Temporal Automated Risk (STAR)** system is an advanced Anti-Money Laundering (AML) platform bridging the gap between cutting-edge Graph Machine Learning research and production-ready real-time infrastructure.

By integrating **Temporal Graph Neural Networks (TGNNs)** with **Neo4j Aura** cloud databases, STAR automatically flags complex typologies—ranging from rapid layering to circular fund routing—by evaluating the contextual "subgraph" of every transaction in real time.

This repository contains both the **Model Training Pipeline** used to train the TGNN on the IBM AML dataset, and the **Real-Time Inference App** built to simulate live production traffic.

## 🏗️ Repository Architecture

1. **`/model` (TGNN Training Pipeline)**
   Contains the complete PyTorch Geometric (PyG) pipeline used to train our AI model. 
   - **`training.py` & `models.py`:** Implements the Hybrid Temporal GAT (GATe) architecture, utilizing both spatial message passing and temporal transaction features (time-deltas, ports).
   - **`data_loading.py`:** Handles ingestion and graph batching of the IBM AML Synthetic dataset.
   - **`checkpoint_tgnn_gat_v2.tar`:** The production-ready pre-trained weights yielding high ROC-AUC on fraud detection.

2. **`/backend` (Real-Time Inference Server)**
   A FastAPI application simulating a high-throughput transaction stream.
   - **k-hop Subgraph Extraction:** For every transaction, the backend executes a Cypher query against Neo4j Aura to extract a 2-hop neighborhood, dynamically constructing localized PyG tensors on the fly.
   - **Human-In-The-Loop (HITL) Queue:** Instead of auto-resolving, transactions flagged with >50% fraud probability are persisted to Neo4j as `:Alert` nodes, awaiting analyst review.
   - **`demo_server.py`:** The primary orchestrator handling WebSockets, Neo4j connections, and live PyTorch inference.

3. **`/frontend` (Investigator Dashboard)**
   A React/Vite/TypeScript frontend rendering a high-performance force-directed graph UI (via `react-force-graph-2d`). 
   - Live stream visualization of transactions and embedded fraud typologies.
   - Integrated Case Management queue to review and escalate fraud alerts directly connected to the backend.

---

## 💻 Tech Stack

*   **Backend:** Python, FastAPI, WebSockets, PyTorch Geometric, Neo4j Python Driver.
*   **Frontend:** React, TypeScript, Vite, Force-Graph.
*   **Database Engine:** Neo4j Aura (Cloud managed graph logic & k-hop pathfinding).

---

## ⚙️ Quick Start (Real-Time Demo)

### 1. Backend Setup
```bash
cd backend
# Create and activate environment
uv venv .venv
.venv/Scripts/activate # Windows
# Install dependencies
uv pip install -r requirements.txt 

# Launch the FastAPI orchestrator
python demo_server.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
*Navigate to `http://localhost:5173` to access the Investigator Command Center. Click "START DEMO" to watch the real-time inference loop build the graph and flag fraud.*

---

## 🧪 Model Training

If you wish to retrain or fine-tune the model on cloud hardware (e.g., A100 instances):
```bash
cd model
python training.py --dataset Small_LI --epochs 100 --batch_size 256
```
*Note: The model utilizes a custom Focal Loss implementation to heavily penalize false negatives on the highly imbalanced (1:1000) fraud dataset.*

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
