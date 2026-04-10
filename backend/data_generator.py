import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_enterprise_data(output_file="synthetic_bank_data.csv"):
    # Target: 100,000 transactions
    num_users = 2500 
    users = [f"User_{i}" for i in range(num_users)]
    
    print(f"🏗️ Phase 1: Generating Profiles for {num_users} Entities...")
    user_profiles = []
    for u in users:
        user_profiles.append({
            'node_id': u,
            'acc_age_days': random.randint(1, 5000),
            'is_merchant': 1 if random.random() > 0.94 else 0,
            'risk_region': 1 if random.random() > 0.97 else 0
        })
    pd.DataFrame(user_profiles).to_csv("user_profiles.csv", index=False)

    tx_data = []
    start_time = datetime(2025, 1, 1)

    # 1. HIGH-QUALITY NORMAL NOISE (90,000 rows)
    # We use a Pareto distribution for amounts to mimic real economic behavior
    print("🏗️ Phase 2: Generating 90,000 high-quality normal transactions...")
    for i in range(90000):
        s, r = random.sample(users, 2)
        # Log-normal distribution for realistic 'Retail' vs 'Wholesale' mix
        amt = np.random.lognormal(mean=5.5, sigma=1.5)
        tx_data.append([f"N_{i}", s, r, amt, start_time + timedelta(seconds=i*30), 0, "Normal"])

    # 2. STRATEGIC FRAUD MOTIFS (10,000 rows)
    print("🕵️ Phase 3: Injecting 10,000 adversarial sequences...")
    
    # Complex Circular Flows (A -> B -> C -> D -> A)
    for j in range(500):
        group = random.sample(users, 4)
        t = start_time + timedelta(minutes=random.randint(0, 100000))
        amt = random.uniform(8000, 12000)
        for m in range(4):
            tx_data.append([f"CF_{j}_{m}", group[m], group[(m+1)%4], amt * (1 - m*0.001), 
                            t + timedelta(seconds=m*40), 1, "Circular"])

    # High-Velocity Layering (Deep Chains)
    for k in range(600):
        chain = random.sample(users, 6)
        t = start_time + timedelta(minutes=random.randint(0, 100000))
        amt = random.uniform(20000, 45000)
        for m in range(5):
            tx_data.append([f"LY_{k}_{m}", chain[m], chain[m+1], amt, 
                            t + timedelta(seconds=m*15), 1, "Layering"])

    # Smurfing/Structuring (Many-to-One aggregation)
    for l in range(250):
        target = random.choice(users)
        t = start_time + timedelta(minutes=random.randint(0, 100000))
        for m in range(15):
            sender = random.choice(users)
            tx_data.append([f"ST_{l}_{m}", sender, target, random.uniform(850, 990), 
                            t + timedelta(minutes=m*2), 1, "Structuring"])

    df = pd.DataFrame(tx_data, columns=['id', 'sender', 'receiver', 'amount', 'ts', 'label', 'type'])
    df = df.sort_values('ts')
    
    # High-Quality Feature Engineering
    print("🧠 Phase 4: Pre-calculating Temporal Velocity Vectors...")
    df['velocity'] = 0.0
    last_tx = {}
    
    # Extract values as numpy arrays for high-performance iteration
    senders = df['sender'].values
    tss = df['ts'].values # This is numpy datetime64
    vels = np.zeros(len(df))
    
    for idx, (s, t) in enumerate(zip(senders, tss)):
        if s in last_tx:
            # Use numpy timedelta division to avoid AttributeError and maintain speed
            # (t - last_tx[s]) yields a numpy.timedelta64 object
            dt_minutes = (t - last_tx[s]) / np.timedelta64(1, 'm')
            vels[idx] = 1.0 / (1.0 + np.log1p(dt_minutes))
        else:
            vels[idx] = 0.0 # First time user for this session
        last_tx[s] = t
    
    df['velocity'] = vels
    df.to_csv(output_file, index=False)
    print(f"✨ 100,000 Transactions Validated & Saved to {output_file}.")

if __name__ == "__main__":
    generate_enterprise_data()