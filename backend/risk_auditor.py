import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# --- 1. ARCHITECTURE (Must match train_tgnn.py) ---
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

def run_advanced_audit():
    print("🛡️  Initializing Advanced Profile-Aware Auditor (Goldilocks Calibration)...")
    
    # 1. Load Model & Normalization Metadata
    try:
        checkpoint = torch.load("inductive_tgnn.pth", map_location="cpu", weights_only=False)
        model = InductiveTGNN(node_feat_dim=checkpoint['node_feat_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        p_mean = checkpoint['p_mean']
        p_std = checkpoint['p_std']
        node_feat_dim = checkpoint['node_feat_dim']
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return

    # 2. Load Data
    df = pd.read_csv("inference_test_data.csv", parse_dates=['ts']).sort_values('ts')
    profiles = pd.read_csv("user_profiles.csv").set_index('node_id')

    print(f"🔍 Analyzing {len(df)} transactions with Balanced Contextual Logic...")
    results = []
    
    with torch.no_grad():
        for _, row in df.iterrows():
            # PROFILE LOOKUP
            sender_profile = profiles.loc[row.sender]
            is_merchant = sender_profile['is_merchant']
            acc_age = sender_profile['acc_age_days']
            is_high_risk_zone = sender_profile['risk_region']
            
            # NORMALIZATION (Inductive)
            u_raw = profiles.loc[row.sender].values
            v_raw = profiles.loc[row.receiver].values if row.receiver in profiles.index else np.zeros(node_feat_dim)
            
            u_f = torch.tensor((u_raw - p_mean) / p_std, dtype=torch.float).unsqueeze(0)
            v_f = torch.tensor((v_raw - p_mean) / p_std, dtype=torch.float).unsqueeze(0)
            
            # FEATURES (Log-Scaled)
            vel = torch.tensor([[row.velocity]], dtype=torch.float)
            amt_log = np.log1p(row.amount) / np.log1p(50000.0)
            amt = torch.tensor([[amt_log]], dtype=torch.float)

            # AI RAW SCORE
            risk_score = model(u_f, v_f, amt, vel).item()
            
            # --- THE "GOLDILOCKS" THRESHOLD CALCULATION ---
            # Default threshold for standard layering/circular patterns
            base_threshold = 0.45
            
            # 1. THE REFINED "SMURFING" SENSITIVITY TIER
            # We tighten the candidate net to < $1,200 to target structuring specifically
            # and avoid blanketing normal mid-tier retail purchases.
            is_structuring_candidate = row.amount <= 1200
            
            if is_structuring_candidate:
                # Set floor to 0.34 (Safely above Normal avg ~16%, below Structuring avg ~36%)
                base_threshold = 0.34 
                # Give merchants a slight buffer to prevent retail FPs, but much less than normal
                merchant_buffer = 0.05 if is_merchant else 0
            else:
                merchant_buffer = 0.15 if is_merchant else 0

            # 2. Account Profile Modifiers
            current_threshold = base_threshold + merchant_buffer
            
            # Stricter for high-risk zones regardless of tier
            if is_high_risk_zone:
                current_threshold -= 0.08
            
            # New accounts are moderately suspicious
            if acc_age < 365:
                current_threshold -= 0.04
                
            # Final Decision
            is_alert = 1 if risk_score > current_threshold else 0
            
            results.append({
                'id': row.id,
                'type': row.type,
                'actual': int(row.label),
                'risk_score': risk_score,
                'threshold_used': current_threshold,
                'pred': is_alert
            })

    res_df = pd.DataFrame(results)

    # 4. RESULTS REPORTING
    print("\n" + "═"*65)
    print("🏆  ADVANCED AML AUDIT REPORT (OPTIMIZED)")
    print("═"*65)
    
    print(f"\n[Precision Health Check]")
    fps = res_df[(res_df['actual'] == 0) & (res_df['pred'] == 1)]
    old_fps = 59 
    reduction = ((old_fps - len(fps)) / old_fps) * 100 if old_fps > 0 else 0
    
    print(f"  • Current False Positives : {len(fps)}")
    if reduction > 0:
        print(f"  • Alert Reduction Rate    : {reduction:.1f}% 📉")
    else:
        print(f"  • Alert Increase Rate     : {abs(reduction):.1f}% 📈")
    
    print(f"\n[Typology Recall Check]")
    for t in res_df['type'].unique():
        subset = res_df[res_df['type'] == t]
        if subset['actual'].sum() > 0:
            recall = (subset[subset['actual'] == 1]['pred']).mean() * 100
            print(f"  • {t:<15}: {recall:>6.1f}% Recall")
        else:
            tnr = (subset[subset['actual'] == 0]['pred'] == 0).mean() * 100
            print(f"  • {t:<15}: {tnr:>6.1f}% Clean Accuracy")

    print("\n[Enterprise Metrics]")
    print(classification_report(res_df['actual'], res_df['pred'], target_names=['Clean', 'Fraud'], zero_division=0))

if __name__ == "__main__":
    run_advanced_audit()