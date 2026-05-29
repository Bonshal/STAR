"""
demo_scenario.py — Curated Demo Graph for TGNN POC Video
==========================================================
Generates a small, visually clean graph (~40 nodes, ~120 edges) with:
  - Named entities (banks, companies, individuals)
  - 3 embedded fraud patterns the TGNN should catch
  - Proper edge features matching training format:
    [Timestamp, Amount, Currency, Format, InPort, OutPort, InTimeDelta, OutTimeDelta]
"""

import numpy as np
import torch
from dataclasses import dataclass, field

# ── Entity Names ─────────────────────────────────────────────────────────

ENTITY_NAMES = [
    # Legitimate businesses
    "GlobalTrade Corp", "Metro Bank", "Apex Financial", "Summit Holdings",
    "Pacific Ventures", "Sterling Capital", "Nordic Trust",
    # Individuals (normal)
    "Alice Chen", "Bob Martinez", "Carol Wright", "David Kim",
    "Emma Johnson", "Frank Liu", "Grace Park", "Henry Adams",
    "Isabel Torres", "James Wilson", "Karen Lee", "Liam Brown",
    "Maya Singh", "Noah Garcia", "Olivia Reed", "Patrick Murphy",
    # Shell companies (used in fraud)
    "Oceanic Consulting", "Vortex Ltd", "Phantom Intl", "Eclipse Ventures",
    "Shadow Partners", "Nexus Trading", "Cipher Holdings", "Drift Logistics",
    # Mules/suspicious individuals
    "Viktor Petrov", "Yuki Tanaka", "Reza Mohammadi", "Chen Wei",
    "Lucia Fernandez", "Omar Hassan", "Svetlana Kozlov",
    # More legitimate padding
    "Coastal Shipping", "River Finance", "Delta Insurance", "Pinnacle Wealth",
]


@dataclass
class Transaction:
    """A single transaction in the demo scenario."""
    sender_idx: int
    receiver_idx: int
    amount: float
    timestamp: float       # seconds from epoch
    currency: int          # 0=USD, 1=EUR, 2=GBP, 3=CNY
    payment_format: int    # 0=Wire, 1=ACH, 2=Cheque, 3=Crypto, 4=Cash
    is_fraud: int          # 0 or 1
    tx_type: str           # "Normal", "Circular", "Layering", "Structuring"
    label: str = ""        # Human-readable description


def build_demo_scenario():
    """
    Build the full demo scenario with transactions.

    Returns a dict with:
      - entity_names: list of str
      - transactions: list of Transaction
      - node_id_map: dict mapping name -> index
    """
    names = ENTITY_NAMES[:42]
    name_to_idx = {n: i for i, n in enumerate(names)}
    transactions = []

    base_time = 0.0  # We'll use relative timestamps

    # ────────────────────────────────────────────────────────────
    # PHASE 1: Normal Background Traffic (~80 transactions)
    # ────────────────────────────────────────────────────────────
    rng = np.random.RandomState(42)

    normal_pairs = [
        ("Alice Chen", "Metro Bank"), ("Bob Martinez", "Apex Financial"),
        ("Carol Wright", "Summit Holdings"), ("David Kim", "Pacific Ventures"),
        ("Emma Johnson", "Sterling Capital"), ("Frank Liu", "Nordic Trust"),
        ("Grace Park", "Metro Bank"), ("Henry Adams", "Apex Financial"),
        ("Isabel Torres", "GlobalTrade Corp"), ("James Wilson", "Summit Holdings"),
        ("Karen Lee", "Pacific Ventures"), ("Liam Brown", "Sterling Capital"),
        ("Maya Singh", "Nordic Trust"), ("Noah Garcia", "Metro Bank"),
        ("Olivia Reed", "Apex Financial"), ("Patrick Murphy", "GlobalTrade Corp"),
        ("Metro Bank", "Apex Financial"), ("Summit Holdings", "Pacific Ventures"),
        ("Sterling Capital", "Nordic Trust"), ("GlobalTrade Corp", "Metro Bank"),
        ("Coastal Shipping", "River Finance"), ("Delta Insurance", "Pinnacle Wealth"),
        ("River Finance", "Metro Bank"), ("Pinnacle Wealth", "Apex Financial"),
        ("Alice Chen", "Bob Martinez"), ("Carol Wright", "David Kim"),
        ("Emma Johnson", "Frank Liu"), ("Grace Park", "Henry Adams"),
        ("Metro Bank", "GlobalTrade Corp"), ("Apex Financial", "Sterling Capital"),
    ]

    t = base_time
    for i, (sender, receiver) in enumerate(normal_pairs):
        # Normal amounts: $50 - $5,000
        amt = rng.lognormal(6.5, 0.8)
        amt = min(max(amt, 50), 8000)
        currency = rng.choice([0, 0, 0, 1, 2])  # Mostly USD
        fmt = rng.choice([0, 0, 1, 1, 2])        # Mostly Wire/ACH
        t += rng.uniform(1800, 7200)              # 30min to 2hr gaps

        transactions.append(Transaction(
            sender_idx=name_to_idx[sender],
            receiver_idx=name_to_idx[receiver],
            amount=round(amt, 2),
            timestamp=t,
            currency=currency,
            payment_format=fmt,
            is_fraud=0,
            tx_type="Normal",
            label=f"{sender} → {receiver}"
        ))

    # More normal background (batch 2) — fills gaps
    for i in range(50):
        s_name = names[rng.randint(0, 20)]
        r_name = names[rng.randint(0, 20)]
        while r_name == s_name:
            r_name = names[rng.randint(0, 20)]

        amt = rng.lognormal(6.0, 1.0)
        amt = min(max(amt, 30), 6000)
        t += rng.uniform(600, 3600)

        transactions.append(Transaction(
            sender_idx=name_to_idx[s_name],
            receiver_idx=name_to_idx[r_name],
            amount=round(amt, 2),
            timestamp=t,
            currency=rng.choice([0, 0, 1]),
            payment_format=rng.choice([0, 1, 1]),
            is_fraud=0,
            tx_type="Normal",
            label=f"{s_name} → {r_name}"
        ))

    # ────────────────────────────────────────────────────────────
    # PHASE 2: FRAUD PATTERN 1 — Circular Laundering Loop
    # Viktor → Oceanic → Vortex → Phantom → Viktor
    # High amounts, rapid succession, crypto payments
    # ────────────────────────────────────────────────────────────
    t += 1200  # 20 minutes after last normal
    loop_nodes = ["Viktor Petrov", "Oceanic Consulting", "Vortex Ltd", "Phantom Intl"]
    loop_amount = 47500.0

    for step in range(4):
        sender = loop_nodes[step]
        receiver = loop_nodes[(step + 1) % 4]
        # Slight amount decay to simulate fees
        amt = loop_amount * (1 - step * 0.002)

        transactions.append(Transaction(
            sender_idx=name_to_idx[sender],
            receiver_idx=name_to_idx[receiver],
            amount=round(amt, 2),
            timestamp=t + step * 45,  # 45 seconds apart!
            currency=0,               # USD
            payment_format=3,          # Crypto
            is_fraud=1,
            tx_type="Circular",
            label=f"🔴 LOOP: {sender} → {receiver}"
        ))

    # ────────────────────────────────────────────────────────────
    # PHASE 2B: More normal traffic between fraud patterns
    # ────────────────────────────────────────────────────────────
    t += 3600
    for i in range(15):
        s_name = names[rng.randint(0, 16)]
        r_name = names[rng.randint(0, 16)]
        while r_name == s_name:
            r_name = names[rng.randint(0, 16)]
        amt = rng.lognormal(6.2, 0.7)
        amt = min(max(amt, 100), 5000)
        t += rng.uniform(300, 1800)

        transactions.append(Transaction(
            sender_idx=name_to_idx[s_name],
            receiver_idx=name_to_idx[r_name],
            amount=round(amt, 2),
            timestamp=t,
            currency=0,
            payment_format=rng.choice([0, 1]),
            is_fraud=0,
            tx_type="Normal",
            label=f"{s_name} → {r_name}"
        ))

    # ────────────────────────────────────────────────────────────
    # PHASE 3: FRAUD PATTERN 2 — Rapid Layering Chain
    # Reza → Eclipse → Shadow → Nexus → Cipher → Drift
    # Large amount moving through 5 hops in under 2 minutes
    # ────────────────────────────────────────────────────────────
    t += 900
    chain = ["Reza Mohammadi", "Eclipse Ventures", "Shadow Partners",
             "Nexus Trading", "Cipher Holdings", "Drift Logistics"]
    chain_amount = 92000.0

    for step in range(5):
        sender = chain[step]
        receiver = chain[step + 1]

        transactions.append(Transaction(
            sender_idx=name_to_idx[sender],
            receiver_idx=name_to_idx[receiver],
            amount=round(chain_amount, 2),
            timestamp=t + step * 18,  # 18 seconds apart!
            currency=1,               # EUR
            payment_format=0,          # Wire
            is_fraud=1,
            tx_type="Layering",
            label=f"🟠 LAYER: {sender} → {receiver}"
        ))

    # ────────────────────────────────────────────────────────────
    # PHASE 3B: Interleave normal
    # ────────────────────────────────────────────────────────────
    t += 5400
    for i in range(10):
        s_name = names[rng.randint(0, 18)]
        r_name = names[rng.randint(0, 18)]
        while r_name == s_name:
            r_name = names[rng.randint(0, 18)]
        amt = rng.lognormal(5.8, 0.9)
        amt = min(max(amt, 40), 4000)
        t += rng.uniform(600, 2400)

        transactions.append(Transaction(
            sender_idx=name_to_idx[s_name],
            receiver_idx=name_to_idx[r_name],
            amount=round(amt, 2),
            timestamp=t,
            currency=0,
            payment_format=rng.choice([0, 1, 2]),
            is_fraud=0,
            tx_type="Normal",
            label=f"{s_name} → {r_name}"
        ))

    # ────────────────────────────────────────────────────────────
    # PHASE 4: FRAUD PATTERN 3 — Structuring (Smurfing)
    # Multiple accounts send sub-$9,500 to Chen Wei & Yuki Tanaka
    # ────────────────────────────────────────────────────────────
    t += 600
    smurfers = ["Lucia Fernandez", "Omar Hassan", "Svetlana Kozlov",
                "Viktor Petrov", "Reza Mohammadi"]
    targets = ["Chen Wei", "Yuki Tanaka"]

    for target in targets:
        for j, smurf in enumerate(smurfers):
            amt = rng.uniform(4800, 9400)
            transactions.append(Transaction(
                sender_idx=name_to_idx[smurf],
                receiver_idx=name_to_idx[target],
                amount=round(amt, 2),
                timestamp=t + j * 120,  # 2 minutes apart
                currency=0,
                payment_format=4,        # Cash
                is_fraud=1,
                tx_type="Structuring",
                label=f"🔴 SMURF: {smurf} → {target}"
            ))
        t += 900

    # ────────────────────────────────────────────────────────────
    # PHASE 5: Final normal trailing traffic
    # ────────────────────────────────────────────────────────────
    t += 1800
    for i in range(10):
        s_name = names[rng.randint(0, 20)]
        r_name = names[rng.randint(0, 20)]
        while r_name == s_name:
            r_name = names[rng.randint(0, 20)]
        amt = rng.lognormal(6.0, 0.8)
        amt = min(max(amt, 60), 5000)
        t += rng.uniform(1200, 3600)

        transactions.append(Transaction(
            sender_idx=name_to_idx[s_name],
            receiver_idx=name_to_idx[r_name],
            amount=round(amt, 2),
            timestamp=t,
            currency=0,
            payment_format=rng.choice([0, 1]),
            is_fraud=0,
            tx_type="Normal",
            label=f"{s_name} → {r_name}"
        ))

    # Sort all by timestamp
    transactions.sort(key=lambda tx: tx.timestamp)

    return {
        "entity_names": names,
        "transactions": transactions,
        "node_id_map": name_to_idx,
    }


def scenario_to_pyg_tensors(scenario: dict):
    """
    Convert the scenario into PyG-compatible tensors.

    Edge features (8 dims, matching GATe training format):
      [Timestamp, Amount, Currency, PaymentFormat, InPort, OutPort, InTimeDelta, OutTimeDelta]

    Returns dict with:
      - x: node features [num_nodes, 1]  (placeholder 1s)
      - edge_index: [2, num_edges]
      - edge_attr: [num_edges, 8]  (raw, un-normalized)
      - y: [num_edges] ground truth labels
      - timestamps: [num_edges] raw timestamps
    """
    txs = scenario["transactions"]
    num_nodes = len(scenario["entity_names"])
    num_edges = len(txs)

    # Node features: all ones (matching training)
    x = torch.ones((num_nodes, 1), dtype=torch.float32)

    # Build edge_index and base edge_attr (4 features)
    src = []
    dst = []
    base_attrs = []  # [Timestamp, Amount, Currency, Format]
    labels = []
    timestamps = []

    for tx in txs:
        src.append(tx.sender_idx)
        dst.append(tx.receiver_idx)
        base_attrs.append([tx.timestamp, tx.amount, tx.currency, tx.payment_format])
        labels.append(tx.is_fraud)
        timestamps.append(tx.timestamp)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr_base = torch.tensor(base_attrs, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    timestamps_t = torch.tensor(timestamps, dtype=torch.float32)

    # Normalize timestamps to start from 0
    edge_attr_base[:, 0] = edge_attr_base[:, 0] - edge_attr_base[:, 0].min()

    # ── Compute ports (matching data_util.py logic) ──
    # InPort: unique neighbor ordering for incoming edges
    # OutPort: unique neighbor ordering for outgoing edges
    in_ports = _compute_ports(edge_index, timestamps_t, direction="in")
    out_ports = _compute_ports(edge_index, timestamps_t, direction="out")

    # ── Compute time deltas (matching data_util.py logic) ──
    in_tds = _compute_time_deltas(edge_index, timestamps_t, direction="in")
    out_tds = _compute_time_deltas(edge_index, timestamps_t, direction="out")

    # Concatenate all 8 features
    edge_attr = torch.cat([
        edge_attr_base,           # [Timestamp, Amount, Currency, Format]
        in_ports.unsqueeze(1),    # InPort
        out_ports.unsqueeze(1),   # OutPort
        in_tds.unsqueeze(1),      # InTimeDelta
        out_tds.unsqueeze(1),     # OutTimeDelta
    ], dim=1)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": y,
        "timestamps": timestamps_t,
    }


def _compute_ports(edge_index, timestamps, direction="in"):
    """Compute port numberings matching data_util.py's ports() function."""
    num_edges = edge_index.shape[1]
    port_values = torch.zeros(num_edges)

    if direction == "in":
        # For each destination node, assign port numbers based on unique sources
        target_nodes = edge_index[1]
        source_nodes = edge_index[0]
    else:
        # For outgoing: flip perspective
        target_nodes = edge_index[0]
        source_nodes = edge_index[1]

    # Group by target node
    node_ids = target_nodes.unique()
    for node_id in node_ids:
        mask = target_nodes == node_id
        edge_indices = torch.where(mask)[0]
        sources = source_nodes[edge_indices]
        times = timestamps[edge_indices]

        # Sort by time
        sort_idx = times.argsort()
        sorted_sources = sources[sort_idx]
        sorted_edge_indices = edge_indices[sort_idx]

        # Assign unique port numbers
        seen = {}
        for i, (src, eidx) in enumerate(zip(sorted_sources.tolist(), sorted_edge_indices.tolist())):
            if src not in seen:
                seen[src] = len(seen)
            port_values[eidx] = seen[src]

    return port_values


def _compute_time_deltas(edge_index, timestamps, direction="in"):
    """Compute time deltas matching data_util.py's time_deltas() function."""
    num_edges = edge_index.shape[1]
    td_values = torch.zeros(num_edges)

    if direction == "in":
        target_nodes = edge_index[1]
    else:
        target_nodes = edge_index[0]

    node_ids = target_nodes.unique()
    for node_id in node_ids:
        mask = target_nodes == node_id
        edge_indices = torch.where(mask)[0]
        times = timestamps[edge_indices]

        # Sort by time
        sort_idx = times.argsort()
        sorted_times = times[sort_idx]
        sorted_edge_indices = edge_indices[sort_idx]

        # Compute deltas
        for i in range(len(sorted_times)):
            if i == 0:
                td_values[sorted_edge_indices[i]] = 0.0
            else:
                td_values[sorted_edge_indices[i]] = sorted_times[i] - sorted_times[i - 1]

    return td_values


def build_graph_json(scenario: dict):
    """
    Build the JSON payload for the frontend graph visualization.
    Returns dict with 'nodes' and 'links' arrays.
    """
    names = scenario["entity_names"]
    txs = scenario["transactions"]

    # Calculate node degrees and fraud involvement
    node_info = {}
    for name in names:
        node_info[name] = {
            "id": name,
            "status": "STABLE",
            "risk_score": 0,
            "is_fraud": False,
            "in_loop": False,
            "degree": 0,
            "reasons": [],
        }

    for tx in txs:
        s_name = names[tx.sender_idx]
        r_name = names[tx.receiver_idx]
        node_info[s_name]["degree"] += 1
        node_info[r_name]["degree"] += 1

    nodes = list(node_info.values())

    links = []
    for i, tx in enumerate(txs):
        links.append({
            "source": names[tx.sender_idx],
            "target": names[tx.receiver_idx],
            "tx_id": f"TX_{i:04d}",
            "amount": tx.amount,
            "type": tx.tx_type,
            "is_alert": False,
            "in_loop": False,
            "risk_score": 0,
            "is_fraud_gt": tx.is_fraud,  # Ground truth (not shown to user initially)
        })

    return {"nodes": nodes, "links": links}


if __name__ == "__main__":
    scenario = build_demo_scenario()
    print(f"Entities: {len(scenario['entity_names'])}")
    print(f"Transactions: {len(scenario['transactions'])}")

    fraud_count = sum(1 for tx in scenario["transactions"] if tx.is_fraud)
    normal_count = len(scenario["transactions"]) - fraud_count
    print(f"Normal: {normal_count}, Fraud: {fraud_count}")
    print(f"Fraud ratio: {fraud_count / len(scenario['transactions']) * 100:.1f}%")

    # Test tensor conversion
    tensors = scenario_to_pyg_tensors(scenario)
    print(f"\nTensor shapes:")
    print(f"  x: {tensors['x'].shape}")
    print(f"  edge_index: {tensors['edge_index'].shape}")
    print(f"  edge_attr: {tensors['edge_attr'].shape}")
    print(f"  y: {tensors['y'].shape}")

    # Show fraud transactions
    print(f"\nFraud transactions:")
    for tx in scenario["transactions"]:
        if tx.is_fraud:
            print(f"  {tx.label} | ${tx.amount:,.2f} | {tx.tx_type}")
