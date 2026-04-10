import pandas as pd
import torch
import random
from datetime import datetime

def generate_challenge_csv(checkpoint_path="trained_tgnn.pth", output_file="test_challenges.csv"):
    print(f"🎲 Loading user map from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        user_map = checkpoint['user_map']
        users = list(user_map.keys())
    except FileNotFoundError:
        print("❌ Error: trained_tgnn.pth not found. Train the model first!")
        return

    print("🏗️ Creating unseen transaction scenarios...")
    test_cases = []
    
    # 1. Normal Peer Transfer (Clean)
    test_cases.append({
        'scenario': "Normal Peer Transfer",
        'sender': random.choice(users), 
        'receiver': random.choice(users),
        'amount': 150.0, 
        'dt_min': 1440.0, # 1 day later
        'label': 0
    })

    # 2. Rapid Circular Hop (Fraud - High Velocity Loop)
    # Picking three random users to form a loop
    u1, u2, u3 = random.sample(users, 3)
    test_cases.append({
        'scenario': "Rapid Circular Hop",
        'sender': u1, 'receiver': u2,
        'amount': 5000.0, 'dt_min': 1.2, # Extremely fast
        'label': 1
    })

    # 3. Smurfing Deposit (Fraud - Structuring)
    test_cases.append({
        'scenario': "Smurfing Deposit",
        'sender': random.choice(users), 'receiver': random.choice(users),
        'amount': 9200.0, 'dt_min': 4.5, # Fast and near threshold
        'label': 1
    })

    # 4. Large Normal Purchase (Clean - High Amount but Slow)
    test_cases.append({
        'scenario': "Large Normal Purchase",
        'sender': random.choice(users), 'receiver': random.choice(users),
        'amount': 19500.0, 'dt_min': 6000.0, # Very slow
        'label': 0
    })

    # 5. Rapid Layering Step (Fraud - Chain behavior)
    test_cases.append({
        'scenario': "Rapid Layering Hop",
        'sender': random.choice(users), 'receiver': random.choice(users),
        'amount': 15000.0, 'dt_min': 0.5, # 30 seconds gap
        'label': 1
    })

    df = pd.DataFrame(test_cases)
    df.to_csv(output_file, index=False)
    print(f"✅ Success! Generated {len(df)} unseen challenges in {output_file}")

if __name__ == "__main__":
    generate_challenge_csv()