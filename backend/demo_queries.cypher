// =========================================================================
// FCCI DEMO QUERY LIBRARY — Neo4j Browser
// =========================================================================
// Paste each block into Neo4j Browser one at a time during the demo.
// Load neo4j_style.grass first (gear icon → "Load style") so risk nodes
// appear in red and edges are colour-coded by channel.
// =========================================================================


// ─────────────────────────────────────────────────────────────────────────
// STEP 0 — STABLE STATE: Full graph overview
// Shows judges the complete transaction network before any analysis.
// ─────────────────────────────────────────────────────────────────────────
MATCH (n)-[r:TRANSFER]->(m)
RETURN n, r, m;


// ─────────────────────────────────────────────────────────────────────────
// STEP 1 — PATTERN DETECTION: Circular Round-Tripping
//
// Algorithm: graph cycle traversal (2–4 hops).
// Finds any Person node where money leaves and returns — no labels needed.
// The returned PATH can be visualised as a physical loop in the browser.
// ─────────────────────────────────────────────────────────────────────────
MATCH path = (p:Person)-[:TRANSFER*2..4]->(p)
RETURN path;


// ─────────────────────────────────────────────────────────────────────────
// STEP 2 — PATTERN DETECTION: Rapid Layering (Short Dwell Time)
//
// Algorithm: temporal gap between inbound and outbound transfer < 30 min,
// with near-full pass-through (≥95% of inbound amount exits).
// ─────────────────────────────────────────────────────────────────────────
MATCH (sender)-[inbound:TRANSFER]->(hub:Person)-[outbound:TRANSFER]->(destination)
WHERE duration.between(inbound.timestamp, outbound.timestamp).minutes < 30
  AND inbound.amount >= (outbound.amount * 0.95)
RETURN sender, inbound, hub, outbound, destination;


// ─────────────────────────────────────────────────────────────────────────
// STEP 3 — PATTERN DETECTION: Structuring / Smurfing
//
// Algorithm: aggregate all sub-$10k transfers to the same recipient.
// Triggers when cumulative total > $50k — bypasses the reporting threshold.
// ─────────────────────────────────────────────────────────────────────────
MATCH (depositor)-[t:TRANSFER]->(recipient:Person)
WHERE t.amount < 10000
WITH recipient, sum(t.amount) AS total_deposited, count(t) AS deposit_count
WHERE total_deposited > 50000
RETURN recipient.name              AS Recipient,
       round(total_deposited)      AS `Total Deposited ($)`,
       deposit_count               AS `Number of Deposits`,
       round(total_deposited / deposit_count) AS `Average per Deposit ($)`;


// ─────────────────────────────────────────────────────────────────────────
// STEP 3b — Visualise the structuring subgraph (all deposit edges)
// ─────────────────────────────────────────────────────────────────────────
MATCH (depositor)-[t:TRANSFER]->(recipient:Person {name: "Charlie Day"})
WHERE t.amount < 10000
RETURN depositor, t, recipient;


// ─────────────────────────────────────────────────────────────────────────
// STEP 4 — PATTERN DETECTION: Dormant Account Activation
//
// Algorithm: time gap between first and last transaction > 365 days,
// combined with a peak transfer > 10× the declared expected_volume.
// ─────────────────────────────────────────────────────────────────────────
MATCH (p:Person)-[t:TRANSFER]->()
WITH p,
     p.expected_volume           AS declared_monthly,
     min(t.timestamp)            AS first_seen,
     max(t.timestamp)            AS last_seen,
     max(t.amount)               AS peak_transfer,
     count(t)                    AS tx_count
WHERE duration.between(first_seen, last_seen).days > 365
  AND peak_transfer > declared_monthly * 10
RETURN p.name               AS Entity,
       tx_count              AS `Total Transactions`,
       round(declared_monthly) AS `Declared Monthly Vol ($)`,
       round(peak_transfer)  AS `Peak Transfer ($)`,
       toString(first_seen)  AS `First Active`,
       toString(last_seen)   AS `Last Active`;


// ─────────────────────────────────────────────────────────────────────────
// STEP 4b — Dormant account full journey
// ─────────────────────────────────────────────────────────────────────────
MATCH (p:Person {name: "Dormant_Dave"})-[t:TRANSFER]->(r)
RETURN p, t, r;


// ─────────────────────────────────────────────────────────────────────────
// STEP 5 — RISK INTELLIGENCE VIEW: All flagged nodes and edges
//
// run_demo.py runs the detectors and writes risk_flag=true back onto
// nodes and relationships — this view isolates the "poisoned" sub-graph.
// ─────────────────────────────────────────────────────────────────────────
MATCH (n:Person)-[t:TRANSFER]->(m:Person)
WHERE n.risk_flag = true OR m.risk_flag = true OR t.risk_flag = true
RETURN n, t, m;


// ─────────────────────────────────────────────────────────────────────────
// STEP 6 — ALERT SUMMARY TABLE (for Evidence Package narrative)
// ─────────────────────────────────────────────────────────────────────────
MATCH (p:Person)
WHERE p.risk_flag = true
RETURN p.name       AS `Flagged Entity`,
       p.occupation AS `Occupation`,
       p.risk_type  AS `Detected Pattern`,
       p.risk_count AS `Alert Count`
ORDER BY p.risk_count DESC;


// ─────────────────────────────────────────────────────────────────────────
// STEP 7 — EVIDENCE PACKAGE: Profile mismatch (KYC vs actual volume)
//
// Shows total volume transacted vs declared expected_volume.
// Deviation ratio surfaces entities whose behaviour contradicts their KYC.
// ─────────────────────────────────────────────────────────────────────────
MATCH (p:Person)-[t:TRANSFER]->()
WITH p,
     p.expected_volume          AS declared,
     sum(t.amount)              AS actual_total,
     count(t)                   AS tx_count
WHERE actual_total > 0
RETURN p.name                            AS Entity,
       round(declared)                   AS `Declared Vol ($)`,
       round(actual_total)               AS `Actual Total ($)`,
       round(actual_total / declared, 1) AS `Deviation Ratio`,
       tx_count                          AS `Transactions`
ORDER BY `Deviation Ratio` DESC
LIMIT 10;
