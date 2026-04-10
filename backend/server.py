"""
server.py — IFFT Dashboard Backend
====================================
FastAPI server providing:
  - POST /api/upload    → Upload CSV, import into Neo4j, return graph
  - GET  /api/graph     → Current graph state from Neo4j
  - GET  /api/patterns  → Fraud pattern counts
  - GET  /api/stats     → Summary statistics
  - WS   /ws/inference  → Stream real-time TGNN inference results
"""

import asyncio
import json
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
import uvicorn


# ── Model Architecture (must match inference_bridge.py) ──────────────────

class InductiveTGNN(nn.Module):
    def __init__(self, node_feat_dim=3):
        super().__init__()
        self.encoder = nn.Linear(node_feat_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32 * 2 + 2, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, u_feat, v_feat, amt, vel):
        u_emb = F.leaky_relu(self.bn1(self.encoder(u_feat)))
        v_emb = F.leaky_relu(self.bn1(self.encoder(v_feat)))
        x = torch.cat([u_emb, v_emb, amt, vel], dim=-1)
        x = F.leaky_relu(self.bn2(self.fc1(x)))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))


# ── App Setup ────────────────────────────────────────────────────────────

app = FastAPI(title="IFFT Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "12345678"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Track the last uploaded CSV so inference knows what to process
current_csv_path: str | None = None

# Pre-load model at module level
MODEL = None
MODEL_META: dict = {}


def load_model():
    global MODEL, MODEL_META
    try:
        checkpoint = torch.load(
            "inductive_tgnn.pth", map_location="cpu", weights_only=False
        )
        MODEL = InductiveTGNN(node_feat_dim=checkpoint["node_feat_dim"])
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        MODEL.eval()
        MODEL_META = {
            "p_mean": checkpoint["p_mean"],
            "p_std": checkpoint["p_std"],
            "node_feat_dim": checkpoint["node_feat_dim"],
        }
        print("🧠 TGNN Model loaded successfully")
    except Exception as e:
        print(f"⚠️  Model load failed (inference will be unavailable): {e}")


load_model()


@app.on_event("shutdown")
def shutdown():
    driver.close()


# ── Helper: build graph payload from Neo4j ───────────────────────────────

def _build_graph_payload() -> dict:
    with driver.session() as session:
        nodes_res = session.run("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[t:TRANSACT]-()
            WITH p, count(t) AS degree, collect(DISTINCT CASE WHEN t.is_alert = true THEN coalesce(t.type, 'Unknown') ELSE null END) AS alert_types
            RETURN p.id AS id,
                   coalesce(p.status, 'STABLE') AS status,
                   coalesce(p.risk_score, 0) AS risk_score,
                   coalesce(p.is_fraud, false) AS is_fraud,
                   coalesce(p.in_loop, false) AS in_loop,
                   degree,
                   alert_types
        """)
        nodes = []
        for r in nodes_res:
            d = dict(r)
            reasons = set(d.pop("alert_types", []))
            if None in reasons: 
                reasons.remove(None)
            if d.get("in_loop"):
                reasons.add("Circular Loop")
            d["reasons"] = list(reasons)
            nodes.append(d)

        links_res = session.run("""
            MATCH (s:Person)-[t:TRANSACT]->(r:Person)
            RETURN s.id AS source, r.id AS target,
                   t.tx_id AS tx_id, t.amount AS amount,
                   coalesce(t.type, 'Unknown') AS type,
                   coalesce(t.is_alert, false) AS is_alert,
                   coalesce(t.in_loop, false) AS in_loop,
                   coalesce(t.risk_score, 0) AS risk_score
        """)
        links = [dict(r) for r in links_res]

    return {"nodes": nodes, "links": links}


# ── REST Endpoints ───────────────────────────────────────────────────────

@app.get("/api/graph")
def get_graph():
    return _build_graph_payload()


@app.post("/api/upload")
async def upload_and_import(file: UploadFile = File(...)):
    global current_csv_path

    filepath = os.path.join(UPLOAD_DIR, "latest.csv")
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    current_csv_path = filepath

    df = pd.read_csv(filepath)

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        try:
            session.run(
                "CREATE CONSTRAINT person_id IF NOT EXISTS "
                "FOR (p:Person) REQUIRE p.id IS UNIQUE"
            )
        except Exception:
            pass

        query = """
        UNWIND $rows AS row
        MERGE (s:Person {id: row.sender})
          ON CREATE SET s.status = 'STABLE', s.risk_score = 0, s.is_fraud = false
        MERGE (r:Person {id: row.receiver})
          ON CREATE SET r.status = 'STABLE', r.risk_score = 0, r.is_fraud = false
        CREATE (s)-[t:TRANSACT {
            tx_id: row.id,
            amount: toFloat(row.amount),
            timestamp: row.ts,
            type: row.type,
            label: toInteger(row.label)
        }]->(r)
        """
        batch_size = 2500
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size].to_dict("records")
            session.run(query, rows=batch)

    return _build_graph_payload()


@app.get("/api/patterns")
def get_patterns():
    with driver.session() as session:
        res = session.run("""
            MATCH ()-[t:TRANSACT]->()
            WHERE t.is_alert = true
            RETURN coalesce(t.type, 'Unknown') AS type, count(t) AS count
            ORDER BY count DESC
        """)
        patterns = [dict(r) for r in res]

        loop_res = session.run(
            "MATCH (p:Person) WHERE p.in_loop = true RETURN count(p) AS c"
        )
        loop_count = loop_res.single()["c"]

    return {"patterns": patterns, "loop_nodes": loop_count}


@app.get("/api/stats")
def get_stats():
    with driver.session() as session:
        r = session.run("""
            MATCH (p:Person)
            WITH count(p) AS nodes
            OPTIONAL MATCH ()-[t:TRANSACT]->()
            RETURN nodes, count(t) AS edges
        """).single()
        flagged = session.run(
            "MATCH (p:Person) WHERE p.is_fraud = true RETURN count(p) AS c"
        ).single()["c"]
    return {
        "total_nodes": r["nodes"],
        "total_edges": r["edges"],
        "flagged_nodes": flagged,
    }


# ── WebSocket: Real-time Inference ───────────────────────────────────────

@app.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()

    if MODEL is None:
        await websocket.send_json(
            {"type": "error", "data": {"message": "Model not loaded"}}
        )
        await websocket.close()
        return

    try:
        csv_path = current_csv_path or "inference_test_data.csv"
        profiles_path = "user_profiles.csv"

        if not os.path.exists(csv_path):
            await websocket.send_json(
                {"type": "error", "data": {"message": f"CSV not found: {csv_path}"}}
            )
            await websocket.close()
            return

        df = pd.read_csv(csv_path, parse_dates=["ts"]).sort_values("ts")
        profiles = pd.read_csv(profiles_path).set_index("node_id")

        total = len(df)
        await websocket.send_json({"type": "inference_start", "data": {"total": total}})

        # Reset markers
        with driver.session() as session:
            session.run("""
                MATCH (p:Person)
                SET p.status = 'STABLE', p.risk_score = 0, p.is_fraud = false
                WITH p
                MATCH (p)-[t:TRANSACT]-()
                SET t.is_alert = false, t.in_loop = false, t.risk_score = 0
            """)

        last_tx: dict = {}
        alerts_by_type: dict = {}
        processed = 0
        p_mean = MODEL_META["p_mean"]
        p_std = MODEL_META["p_std"]
        nfd = MODEL_META["node_feat_dim"]

        with torch.no_grad():
            for _, row in df.iterrows():
                processed += 1

                if row.sender not in profiles.index:
                    if processed % 200 == 0:
                        await websocket.send_json(
                            {"type": "progress", "data": {"processed": processed, "total": total}}
                        )
                    continue

                sender_profile = profiles.loc[row.sender]
                is_merchant = sender_profile["is_merchant"]
                acc_age = sender_profile["acc_age_days"]
                is_high_risk_zone = sender_profile["risk_region"]

                u_raw = profiles.loc[row.sender].values
                v_raw = (
                    profiles.loc[row.receiver].values
                    if row.receiver in profiles.index
                    else np.zeros(nfd)
                )

                u_f = torch.tensor((u_raw - p_mean) / p_std, dtype=torch.float).unsqueeze(0)
                v_f = torch.tensor((v_raw - p_mean) / p_std, dtype=torch.float).unsqueeze(0)

                t_curr = row.ts
                if row.sender in last_tx:
                    dt_min = (t_curr - last_tx[row.sender]).total_seconds() / 60.0
                    vel_score = 1.0 / (1.0 + np.log1p(dt_min))
                else:
                    vel_score = 0.0
                last_tx[row.sender] = t_curr

                vel_tensor = torch.tensor([[vel_score]], dtype=torch.float)
                amt_log = np.log1p(row.amount) / np.log1p(50000.0)
                amt_tensor = torch.tensor([[amt_log]], dtype=torch.float)

                risk_score = MODEL(u_f, v_f, amt_tensor, vel_tensor).item()

                base_threshold = 0.45
                if row.amount <= 1200:
                    base_threshold = 0.38
                merchant_buffer = 0.05 if is_merchant else 0
                current_threshold = base_threshold + merchant_buffer
                if is_high_risk_zone:
                    current_threshold -= 0.05
                if acc_age < 365:
                    current_threshold -= 0.03

                is_alert = risk_score > current_threshold

                if is_alert:
                    status_label = "CRITICAL" if risk_score > 0.7 else "SUSPICIOUS"

                    with driver.session() as session:
                        session.run("""
                            MATCH (s:Person {id: $s_id})
                            MATCH (r:Person {id: $r_id})
                            SET s.status = $status, s.risk_score = $risk, s.is_fraud = true
                            WITH s, r
                            MATCH (s)-[t:TRANSACT {tx_id: $tx_id}]->(r)
                            SET t.risk_score = $risk, t.is_alert = true
                        """, s_id=row.sender, r_id=row.receiver,
                            status=status_label, risk=round(risk_score * 100, 2),
                            tx_id=row.id)

                    tx_type = str(getattr(row, "type", "Unknown"))
                    alerts_by_type[tx_type] = alerts_by_type.get(tx_type, 0) + 1

                    await websocket.send_json({
                        "type": "alert",
                        "data": {
                            "tx_id": str(row.id),
                            "sender": str(row.sender),
                            "receiver": str(row.receiver),
                            "risk_score": round(risk_score * 100, 2),
                            "status": status_label,
                            "tx_type": tx_type,
                            "amount": round(float(row.amount), 2),
                        },
                    })
                    await asyncio.sleep(0.02)

                if processed % 200 == 0:
                    await websocket.send_json(
                        {"type": "progress", "data": {"processed": processed, "total": total}}
                    )

        # Circular loop tagging
        with driver.session() as session:
            session.run("""
                MATCH path = (p:Person)-[:TRANSACT*2..4]->(p)
                WHERE all(t IN relationships(path) WHERE t.is_alert = true)
                FOREACH (t IN relationships(path) | SET t.in_loop = true)
                FOREACH (n IN nodes(path) | SET n.in_loop = true)
            """)
            loop_count = session.run(
                "MATCH (p:Person) WHERE p.in_loop = true RETURN count(p) AS c"
            ).single()["c"]

        await websocket.send_json({
            "type": "inference_complete",
            "data": {
                "processed": total,
                "patterns": alerts_by_type,
                "loop_nodes": loop_count,
            },
        })

    except WebSocketDisconnect:
        print("Client disconnected during inference")
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "data": {"message": str(e)}})
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
