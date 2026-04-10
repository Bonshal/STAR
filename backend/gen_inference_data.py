import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_challenge_set(output_file="inference_test_data.csv"):
    """
    Generates a high-quality "Gold Standard" test set for model validation.
    Includes exactly 500 transactions across all discussed fraud motifs.
    """
    print("🏗️ Generating Inference Challenge Set (500 Transactions)...")
    
    # Load existing profiles to ensure consistency with the 100k training set
    try:
        profiles = pd.read_csv("user_profiles.csv")
        users = profiles['node_id'].tolist()
    except FileNotFoundError:
        print("❌ Error: user_profiles.csv not found. Please run the training data generator first.")
        return

    tx_data = []
    start_time = datetime.now()

    # 1. NORMAL BACKGROUND (400 transactions)
    # Varied, low-risk behavior
    for i in range(400):
        s, r = random.sample(users, 2)
        amt = np.random.lognormal(mean=5.0, sigma=1.0)
        tx_data.append([f"T_NORM_{i}", s, r, amt, start_time + timedelta(minutes=i*10), 0, "Normal"])

    # 2. CIRCULAR FLOW MOTIF (30 transactions)
    # Pattern: A -> B -> C -> A (Rapid movement)
    for j in range(10):
        loop = random.sample(users, 3)
        t = start_time + timedelta(hours=j*2)
        amt = random.uniform(8000, 11000)
        for m in range(3):
            tx_data.append([f"T_CIRC_{j}_{m}", loop[m], loop[(m+1)%3], amt - (m*10), 
                            t + timedelta(seconds=m*45), 1, "Circular"])

    # 3. RAPID LAYERING MOTIF (40 transactions)
    # Pattern: A -> B -> C -> D -> E (Instant hops)
    for k in range(10):
        chain = random.sample(users, 5)
        t = start_time + timedelta(hours=k*3 + 1)
        amt = random.uniform(15000, 25000)
        for m in range(4):
            tx_data.append([f"T_LAYR_{k}_{m}", chain[m], chain[m+1], amt, 
                            t + timedelta(seconds=m*20), 1, "Layering"])

    # 4. STRUCTURING MOTIF (30 transactions)
    # Pattern: Many small deposits to one target
    for l in range(3):
        target = random.choice(users)
        t = start_time + timedelta(hours=l*5)
        for m in range(10):
            sender = random.choice(users)
            tx_data.append([f"T_STRC_{l}_{m}", sender, target, random.uniform(800, 950), 
                            t + timedelta(minutes=m*2), 1, "Structuring"])

    df = pd.DataFrame(tx_data, columns=['id', 'sender', 'receiver', 'amount', 'ts', 'label', 'type'])
    df = df.sort_values('ts')
    
    # Calculate Velocity for the test set
    print("🧠 Feature Engineering: Calculating Test Set Velocity...")
    df['velocity'] = 0.0
    last_tx = {}
    senders = df['sender'].values
    tss = df['ts'].values
    vels = np.zeros(len(df))
    
    for idx, (s, t) in enumerate(zip(senders, tss)):
        if s in last_tx:
            dt_minutes = (t - last_tx[s]) / np.timedelta64(1, 'm')
            vels[idx] = 1.0 / (1.0 + np.log1p(dt_minutes))
        else:
            vels[idx] = 0.0
        last_tx[s] = t
    
    df['velocity'] = vels
    df.to_csv(output_file, index=False)
    print(f"✅ Success! Inference Challenge Set saved to {output_file}.")

if __name__ == "__main__":
    generate_challenge_set()