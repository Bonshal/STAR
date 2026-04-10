import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from neo4j import GraphDatabase

# --- 1. ARCHITECTURE (Must match the trained inductive_tgnn.pth) ---
class InductiveTGNN(nn.Module):
    def __init__(self, node_feat_dim=3):
        super(InductiveTGNN, self).__init__()
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

class Neo4jInferenceBridge:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # 1. Load Brain & Normalization Metadata
        print("🧠 Connecting to Inductive Intelligence...")
        try:
            # Loading with weights_only=False to support numpy scalars in checkpoint
            checkpoint = torch.load("inductive_tgnn.pth", map_location="cpu", weights_only=False)
            self.model = InductiveTGNN(node_feat_dim=checkpoint['node_feat_dim'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.p_mean = checkpoint['p_mean']
            self.p_std = checkpoint['p_std']
            self.node_feat_dim = checkpoint['node_feat_dim']
        except Exception as e:
            print(f"❌ Initialization Error: {e}")
            raise e

    def close(self):
        self.driver.close()

    def execute_bridge(self, csv_input="inference_test_data.csv"):
        print("🛡️  Starting Behavioral Inference Bridge...")
        
        # 2. Load Environment (Profiles & Test Data)
        df = pd.read_csv(csv_input, parse_dates=['ts']).sort_values('ts')
        profiles = pd.read_csv("user_profiles.csv").set_index('node_id')

        print(f"🔍 Processing {len(df)} transactions into Neo4j...")
        results = []
        last_tx = {} # To calculate real-time velocity per sender
        
        with self.driver.session() as session:
            # 3. Clean Slate for Bloom Styling
            print("🧹 Clearing previous risk markers from Graph...")
            session.run("""
                MATCH (p:Person) SET p.status = 'STABLE', p.risk_score = 0, p.is_fraud = false 
                WITH p
                MATCH (p)-[t:TRANSACT]-() SET t.is_alert = false, t.in_loop = false, t.risk_score = 0
            """)

            with torch.no_grad():
                for _, row in df.iterrows():
                    # --- A. INDUCTIVE FEATURE PREP ---
                    sender_profile = profiles.loc[row.sender]
                    is_merchant = sender_profile['is_merchant']
                    acc_age = sender_profile['acc_age_days']
                    is_high_risk_zone = sender_profile['risk_region']

                    u_raw = profiles.loc[row.sender].values
                    v_raw = profiles.loc[row.receiver].values if row.receiver in profiles.index else np.zeros(self.node_feat_dim)
                    
                    # Normalize using training stats
                    u_f = torch.tensor((u_raw - self.p_mean) / self.p_std, dtype=torch.float).unsqueeze(0)
                    v_f = torch.tensor((v_raw - self.p_mean) / self.p_std, dtype=torch.float).unsqueeze(0)
                    
                    # --- B. VELOCITY & LOG-AMOUNT ---
                    # FIX: Correct Velocity Initialization
                    # In training, first transaction = 0 velocity. 
                    # Previous 'last_tx.get(s, t_curr)' caused 1.0 velocity (max risk).
                    t_curr = row.ts
                    if row.sender in last_tx:
                        dt_min = (t_curr - last_tx[row.sender]).total_seconds() / 60.0
                        vel_score = 1.0 / (1.0 + np.log1p(dt_min))
                    else:
                        vel_score = 0.0 # Match training data baseline
                    
                    last_tx[row.sender] = t_curr
                    vel_tensor = torch.tensor([[vel_score]], dtype=torch.float)
                    
                    # Log-Scaling Amount (Must match train_tgnn.py exactly)
                    amt_log = np.log1p(row.amount) / np.log1p(50000.0)
                    amt_tensor = torch.tensor([[amt_log]], dtype=torch.float)

                    # --- C. INFERENCE ---
                    risk_score = self.model(u_f, v_f, amt_tensor, vel_tensor).item()
                    
                    # --- D. THRESHOLD STRATEGY (The "Goldilocks" Demo Logic) ---
                    # Adjusted slightly higher (0.38) to sharpen precision
                    base_threshold = 0.45
                    if row.amount <= 1200:
                        base_threshold = 0.38
                    
                    # Profile Buffers
                    merchant_buffer = 0.05 if is_merchant else 0
                    current_threshold = base_threshold + merchant_buffer
                    
                    # Dynamic risk modifiers
                    if is_high_risk_zone: current_threshold -= 0.05
                    if acc_age < 365: current_threshold -= 0.03
                        
                    is_alert = 1 if risk_score > current_threshold else 0
                    
                    # --- E. NEO4J WRITE-BACK ---
                    if is_alert:
                        status_label = "CRITICAL" if risk_score > 0.7 else "SUSPICIOUS"
                        session.run("""
                            MATCH (s:Person {id: $s_id})
                            MATCH (r:Person {id: $r_id})
                            SET s.status = $status, s.risk_score = $risk, s.is_fraud = true
                            WITH s, r
                            MATCH (s)-[t:TRANSACT {tx_id: $tx_id}]->(r)
                            SET t.risk_score = $risk, t.is_alert = true
                        """, s_id=row.sender, r_id=row.receiver, status=status_label, risk=round(risk_score*100, 2), tx_id=row.id)

                    results.append({'id': row.id, 'type': row.type, 'actual': int(row.label), 'pred': is_alert})

            # --- F. CIRCULAR LOOP TAGGING (For Bloom 'Expand' Feature) ---
            print("🔄 Tagging Circular Fraud Loops for visualization...")
            session.run("""
                MATCH path = (p:Person)-[:TRANSACT*2..4]->(p)
                WHERE all(t IN relationships(path) WHERE t.is_alert = true)
                FOREACH (t IN relationships(path) | SET t.in_loop = true)
                FOREACH (n IN nodes(path) | SET n.in_loop = true)
            """)

        # Performance Report
        res_df = pd.DataFrame(results)
        print("\n" + "="*65)
        print("🏆  INFERENCE BRIDGE COMPLETE")
        print("="*65)
        print(classification_report(res_df['actual'], res_df['pred'], target_names=['Clean', 'Fraud'], zero_division=0))
        print("\n✨ Data Synchronized. Open Bloom and style nodes by 'status' and relationships by 'in_loop'.")

if __name__ == "__main__":
    # Ensure these match your Neo4j Desktop settings
    URI = "neo4j://127.0.0.1:7687"
    USER = "neo4j"
    PWD = "12345678" 

    bridge = Neo4jInferenceBridge(URI, USER, PWD)
    try:
        bridge.execute_bridge("inference_test_data.csv")
    finally:
        bridge.close()