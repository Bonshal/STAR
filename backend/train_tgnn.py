import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import os

# --- 1. ARCHITECTURE ---
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

# --- 2. STABLE FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, weights=None):
        # clamp to prevent log(0) which causes NaN loss
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if weights is not None:
            F_loss = F_loss * weights
            
        return F_loss.mean()

# --- 3. DATASET ---
class FastFinancialDataset(Dataset):
    def __init__(self, df, profiles_norm, node_feat_dim):
        print(f"⚡ Vectorizing {len(df)} rows with Log-Scaling...")
        self.node_feat_dim = node_feat_dim
        profile_tensor = torch.tensor(profiles_norm.values, dtype=torch.float)
        id_to_idx = {id: i for i, id in enumerate(profiles_norm.index)}
        
        self.u_feats = torch.stack([profile_tensor[id_to_idx[s]] if s in id_to_idx else torch.zeros(node_feat_dim) for s in df.sender])
        self.v_feats = torch.stack([profile_tensor[id_to_idx[r]] if r in id_to_idx else torch.zeros(node_feat_dim) for r in df.receiver])
        self.vels = torch.tensor(df.velocity.values, dtype=torch.float).unsqueeze(1)
        
        # LOG-SCALING: This is the source of the "mismatch" if inference isn't updated!
        # Normalizes amount into a range where small differences ($800 vs $2000) are visible to the AI.
        self.amts = torch.tensor(np.log1p(df.amount.values) / np.log1p(50000.0), dtype=torch.float).unsqueeze(1)
        self.labels = torch.tensor(df.label.values, dtype=torch.float).unsqueeze(1)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.u_feats[idx], self.v_feats[idx], self.amts[idx], self.vels[idx], self.labels[idx]

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for u_f, v_f, amt, vel, labels in loader:
            u_f, v_f, amt, vel = u_f.to(device), v_f.to(device), amt.to(device), vel.to(device)
            preds = model(u_f, v_f, amt, vel)
            all_preds.extend((preds > 0.50).cpu().numpy())
            all_labels.extend(labels.numpy())
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    return p, r, f1

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("synthetic_bank_data.csv", parse_dates=['ts']).sort_values('ts')
    profiles = pd.read_csv("user_profiles.csv").set_index('node_id')
    
    train_df, test_df = train_test_split(df, test_size=0.15, shuffle=False)
    p_mean, p_std = profiles.mean(), profiles.std()
    profiles_norm = (profiles - p_mean) / p_std
    
    train_ds = FastFinancialDataset(train_df, profiles_norm, profiles.shape[1])
    test_ds = FastFinancialDataset(test_df, profiles_norm, profiles.shape[1])
    
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4096)
    
    model = InductiveTGNN(node_feat_dim=profiles.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    pos_weight = (len(df[df.label==0]) / len(df[df.label==1]))

    print(f"🚀 Training 100k TGNN...")
    for epoch in range(20):
        model.train()
        weighted_loss_sum = 0
        raw_loss_sum = 0 
        
        for u_f, v_f, amt, vel, labels in train_loader:
            u_f, v_f, amt, vel, labels = u_f.to(device), v_f.to(device), amt.to(device), vel.to(device), labels.to(device)
            
            optimizer.zero_grad()
            preds = model(u_f, v_f, amt, vel)
            
            # --- WEIGHTING ---
            weights = torch.ones_like(labels)
            weights[labels == 1] = pos_weight
            
            # Structuring Mask (using Log-Scaled boundaries)
            # 0.7 on the log scale is ~ $1,900
            struct_mask = (labels == 1) & (amt < 0.7) & (vel > 0.4)
            weights[struct_mask] *= 300.0
            
            loss = criterion(preds, labels, weights=weights)
            loss.backward()
            optimizer.step()
            
            weighted_loss_sum += loss.item()
            with torch.no_grad():
                raw_loss_sum += F.binary_cross_entropy(preds, labels).item()
            
        avg_weighted = weighted_loss_sum / len(train_loader)
        avg_raw = raw_loss_sum / len(train_loader)
        
        scheduler.step(avg_weighted)
        prec, rec, f1 = evaluate(model, test_loader, device)
        
        print(f"📈 Ep {epoch:02d} | W-Loss: {avg_weighted:.4f} | R-Loss: {avg_raw:.4f} | F1: {f1:.2f} (P:{prec:.2f} R:{rec:.2f})")

    torch.save({
        'model_state_dict': model.to('cpu').state_dict(), 
        'p_mean': p_mean.values, 
        'p_std': p_std.values, 
        'node_feat_dim': profiles.shape[1]
    }, "inductive_tgnn.pth")
    print("✅ Training Complete.")

if __name__ == "__main__": train()