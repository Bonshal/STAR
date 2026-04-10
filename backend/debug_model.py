import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class TemporalGraphModel(nn.Module):
    def __init__(self, num_nodes, node_dim=16):
        super(TemporalGraphModel, self).__init__()
        self.memory = nn.Parameter(torch.randn(num_nodes, node_dim) * 0.1)
        self.fc1 = nn.Linear(node_dim * 2 + 2, 64)
        self.fc2 = nn.Linear(64, 1)
    def forward(self, u, v, amt, vel):
        x = torch.cat([self.memory[u], self.memory[v], amt, vel], dim=-1)
        return torch.sigmoid(self.fc2(F.leaky_relu(self.fc1(x))))

def debug_check():
    print("🔍 Testing Model against Known Fraud Motifs...")
    checkpoint = torch.load("trained_tgnn.pth", map_location="cpu")
    user_map = checkpoint['user_map']
    model = TemporalGraphModel(checkpoint['num_nodes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    df = pd.read_csv("synthetic_bank_data.csv", parse_dates=['ts']).sort_values('ts')
    
    # Let's find 5 known Fraud rows and 5 known Normal rows
    fraud_samples = df[df['label'] == 1].head(10)
    normal_samples = df[df['label'] == 0].head(10)
    samples = pd.concat([fraud_samples, normal_samples])

    print(f"\n{'TYPE':<15} | {'ACTUAL':<8} | {'AI RISK %':<10} | {'MATCH'}")
    print("-" * 50)

    for i in range(1, len(samples)):
        row = samples.iloc[i]
        prev = df.iloc[df.index[df['id'] == row['id']][0] - 1]
        
        u, v = user_map[row.sender], user_map[row.receiver]
        dt = (row.ts - prev.ts).total_seconds() / 60.0
        vel = torch.tensor([1.0 / (1.0 + np.log1p(dt))], dtype=torch.float)
        amt = torch.tensor([row.amount / 50000.0], dtype=torch.float)
        
        with torch.no_grad():
            risk = model(u, v, amt, vel).item()
        
        match = "✅" if (risk > 0.5) == bool(row['label']) else "❌"
        actual_str = "FRAUD" if row['label'] == 1 else "CLEAN"
        print(f"{row['type']:<15} | {actual_str:<8} | {risk*100:>8.2f}% | {match}")

if __name__ == "__main__":
    debug_check()