Master Specification: Intelligent Fund Flow Tracking (IFFT)

1. Hackathon Problem Statement

Objective: Develop an intelligent system that maps and visualizes end-to-end fund movements within a bank across accounts, products, branches, and channels.

Core Requirements:

Graph Intelligence: Move beyond tabular data to relational network analysis.

Typology Detection: Identify: Rapid layering, circular transactions (round-tripping), structuring (smurfing), dormant account activation, and profile mismatches.

Investigative UX: Enable investigators to trace the complete journey of funds.

Regulatory Compliance: Generate Evidence Packages for the Financial Intelligence Unit (FIU).

2. Research Foundations (The Competitive Edge)

Our research identifies that legacy enterprise systems (e.g., Oracle FCCM) are failing because they are Reactive and Siloed. We are pitching a shift toward Contextual Proactive Intelligence.

Key Research Pillars:

Spatial-Temporal Hybrid Modeling: Research (Verlaan et al., 2025) proves that detection requires a dual-input model. We propose a Temporal Graph Neural Network (TGNN) that processes spatial context (network topology) and temporal velocity (transaction frequency) simultaneously.

Line Graph Multi-View Learning: Instead of nodes as accounts, we propose viewing Transactions as Nodes. This allows the system to track the "momentum" and "decay" of a specific fund unit as it traverses the network.

Explainable AI (XAI): Using GNNExplainer to provide "Sub-graph Attribution." This ensures that when an alert is raised, the system can point to the exact 5 transactions out of 5 million that created the risk, satisfying the "Right to Explanation" under emerging AI regulations.

3. Target State Architecture (The Pitch)

This is the enterprise-scale architecture we are pitching as the future of banking compliance.

Tier 1: The Unified Data Fabric (Ingestion)

Real-Time Streaming: Ingestion of SWIFT, POS, Wire, and Internal transfers via a "Data Fabric" that abstracts core banking silos.

Identity Resolution: Real-time matching of disparate records into a single "Golden Entity" using fuzzy matching and graph clustering.

Tier 2: The Graph Intelligence Engine (TGNN Core)

Temporal Graph Neurons: A hybrid LSTM-GNN engine that assigns a "Risk Momentum Score" to every fund movement.

Behavioral Biometrics: An "Invisible Layer" that captures keystroke dynamics and navigation flow during the transaction to detect duress or bot-control (RATs).

Tier 3: The Agentic Investigation Hub (Reporting)

Co-Investigator AI: An LLM-based agentic framework that auto-drafts SAR narratives.

XAI Layer: SHAP/LIME-based explanations for every alert to reduce the "Black Box" risk.

4. Prototype Implementation (The MVB)

The prototype is a functional "Proof of Concept" designed to demonstrate the Target State's effectiveness using available local resources.

Logic Layer (Python/FastAPI): Serves as the "Orchestrator," simulating the high-level TGNN logic using optimized Cypher queries and Z-Score anomaly detection.

Graph Layer (Neo4j): Provides the relational memory for the prototype, enabling sub-second multi-hop pathfinding.

Presentation Layer (React/TypeScript): A high-fidelity "Investigator Command Center." It uses force-graph to visualize the spatial clusters and the "Path Tracer" to isolate evidence.

Real-Time Bridge (Socket.io): Connects the backend simulation to the UI, allowing the graph to "grow" in real-time as transactions are processed.

5. Fraud Detection Logic (Algorithmic Detail)

Pattern

Pitch Logic (Target State)

Prototype Logic (MVB Simulation)

Circular Flow

TGNN Cycle-Detection Motifs

MATCH path = (p:Person)-[:TRANSFER*2..5]->(p)

Rapid Layering

Temporal Momentum Analysis ($\Delta t \to 0$)

duration.between(in.ts, out.ts).minutes < 30

Structuring

Sequential Sequence Clustering

SUM(tx.amount) > $50k where tx.amount < $10k

Dormant Act.

LSTM-based Anomaly "Burst" Detection

TimeGap > 180d AND Z-Score(amount) > 3.0

Mismatch

GraphSAGE Embedding Cosine Similarity

KYC Profile vs. Actual Cluster volume deviation.

6. Demo Video Strategy: "The Story of Discovery"

We demonstrate the "Intelligence" of the system through a 3-minute scripted walkthrough:

The Baseline: Show the Dashboard in "Stable State." Monitor background noise.

The Event: Trigger the "Injection" (Python script). The graph begins to populate.

The Detection: An alert card appears (Red Pulse). The "Investigator" (User) clicks it.

The Trace: The graph isolates the "Poisoned Path." The investigator sees the circular flow visualized as a physical loop.

The FIU Package: The investigator clicks "Generate Evidence." An automated, professional report is created, showcasing the system's "Reasoning."

7. Operational Instructions for Development

Neo4j: Must be running on bolt://localhost:7687.

Backend (Python): Use uvicorn main:app --port 8000. This manages the simulation logic.

Data Generation: Run python data_generator.py to create the "poisoned" CSV files.

Frontend (React): Use the professional TypeScript Dashboard to monitor the feed. Ensure Socket.io is connected to receive real-time updates.