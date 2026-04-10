"""
run_demo.py — FCCI Single-Command Demo Bootstrap
==================================================
Runs the complete setup pipeline in one shot:
  1. Generates synthetic transaction data (no fraud labels).
  2. Clears any previous demo data from Neo4j.
  3. Loads all Person nodes and TRANSFER relationships.
  4. Runs the algorithmic detection suite across every entity.
  5. Writes risk_flag / risk_type properties on detected nodes & edges.
  6. Prints a judge-ready guide with the Neo4j Browser URL and query plan.

Usage:
  python run_demo.py
  (Neo4j must be running on bolt://localhost:7687)
"""

import sys
import time
import pandas as pd
from generator import generate_demo_data
from orchestrator import IFFTOrchestrator

# ── Connection settings ────────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "12345678"

BANNER = """
╔══════════════════════════════════════════════════════════════════════════╗
║        IFFT — Intelligent Fund Flow Tracking  │  Demo Bootstrap         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

DEMO_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════╗
║                  ✅  NEO4J GRAPH LOADED SUCCESSFULLY                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  1. Open Neo4j Browser:  http://localhost:7474                           ║
║     Login: neo4j / 12345678                                              ║
║                                                                          ║
║  2. Load the visual style (enables colour-coded risk view):              ║
║     → Gear icon (bottom-left) → Graph Style → Load style                ║
║     → Drag and drop:  backend/neo4j_style.grass                         ║
║                                                                          ║
║  3. Run the demo queries in order from:  backend/demo_queries.cypher     ║
║                                                                          ║
║  DEMO STORY (3-minute walkthrough):                                      ║
║  ─────────────────────────────────                                       ║
║  STEP 0  │ Stable State   │ Full graph — all nodes visible               ║
║  STEP 1  │ Detection      │ Circular loop: Alice→Bob→Charlie→Alice       ║
║  STEP 2  │ Detection      │ Rapid layering: Global Logistics chain       ║
║  STEP 3  │ Detection      │ Structuring: 12 sub-$10k cash deposits       ║
║  STEP 4  │ Detection      │ Dormant activation: Dormant_Dave's spike     ║
║  STEP 5  │ Risk View      │ Isolated sub-graph of ALL flagged entities   ║
║  STEP 6  │ Alert Table    │ Evidence package summary                     ║
║  STEP 7  │ KYC Deviation  │ Declared vs actual volume — profile mismatch ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def run():
    print(BANNER)

    # ── Step 1: Generate CSV data ─────────────────────────────────────────
    print("[1/4] Generating synthetic transaction data...")
    generate_demo_data()

    # ── Step 2: Connect to Neo4j ──────────────────────────────────────────
    print("[2/4] Connecting to Neo4j and loading graph...")
    try:
        orc = IFFTOrchestrator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    except Exception as exc:
        print(f"\n  ❌  Could not connect to Neo4j at {NEO4J_URI}")
        print(f"      Error: {exc}")
        print("      Make sure Neo4j Desktop is running and the DB is started.")
        sys.exit(1)

    # Clear previous data for a clean demo
    orc.clear_graph()
    print("  [neo4j] Previous data cleared.")

    # Load entities (Person nodes with KYC profiles)
    orc.load_entities("entities.csv")

    # Load every transaction as a raw TRANSFER — no fraud labels at this stage
    df_tx = pd.read_csv("transactions.csv")
    print(f"  [neo4j] Loading {len(df_tx)} raw transactions into the graph...")
    for _, row in df_tx.iterrows():
        orc.ingest_transaction(row.to_dict())
        time.sleep(0.05)   # small delay makes the console feel live
    print(f"  [neo4j] Graph loaded: {len(df_tx)} TRANSFER relationships created.")

    # ── Step 3: Run algorithmic detection ─────────────────────────────────
    print("\n[3/4] Running intelligence detection suite...")
    print("      (Detection queries run against the raw graph — no hardcoded labels)\n")
    orc.tag_fraud_patterns()

    # ── Step 4: Print guide ───────────────────────────────────────────────
    print("\n[4/4] Done!")
    orc.close()
    print(DEMO_GUIDE)


if __name__ == "__main__":
    run()
