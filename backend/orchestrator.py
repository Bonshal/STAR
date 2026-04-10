import pandas as pd
from neo4j import GraphDatabase


class IFFTOrchestrator:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def clear_graph(self):
        """Wipe the database for a clean demo run."""
        self.run_query("MATCH (n) DETACH DELETE n")

    def load_entities(self, csv_path):
        df = pd.read_csv(csv_path)
        query = """
        UNWIND $rows AS row
        MERGE (p:Person {name: row.name})
        SET p.occupation = row.occupation,
            p.expected_volume = toFloat(row.expected_volume)
        """
        self.run_query(query, {"rows": df.to_dict("records")})

    def ingest_transaction(self, tx):
        """
        Write a single transaction into the graph as a :TRANSFER relationship.
        No fraud labels here — this is raw observed data.
        """
        query = """
        MERGE (s:Person {name: $sender})
        MERGE (r:Person {name: $receiver})
        CREATE (s)-[t:TRANSFER {
            id: $tx_id,
            amount: toFloat($amount),
            timestamp: datetime($ts),
            channel: $channel
        }]->(r)
        """
        self.run_query(query, tx)

    # ------------------------------------------------------------------ #
    #  DETECTION SUITE — each method runs a graph query and returns alerts #
    # ------------------------------------------------------------------ #

    def _check_circular(self, name):
        """
        TGNN proxy: look for any cycle 2–4 hops long that returns to the
        same Person. No hardcoded labels — Cypher traverses the live graph.
        """
        query = """
        MATCH path = (p:Person {name: $name})-[:TRANSFER*2..4]->(p)
        RETURN p.name AS entity, length(path) AS hops
        """
        results = self.run_query(query, {"name": name})
        return [
            {
                "type": "Circular Round-Tripping",
                "severity": "Critical",
                "description": f"Funds returned to source via {r['hops']}-hop loop.",
            }
            for r in results
        ]

    def _check_layering(self, name):
        """
        Temporal-momentum proxy: money passes through a node in <30 minutes
        and exits at ≥95% of the inbound amount (minimal processing time).
        """
        query = """
        MATCH (s)-[in:TRANSFER]->(m:Person {name: $name})-[out:TRANSFER]->(r)
        WHERE duration.between(in.timestamp, out.timestamp).minutes < 30
          AND in.amount >= (out.amount * 0.95)
        RETURN m.name AS entity, out.amount AS amt
        """
        results = self.run_query(query, {"name": name})
        return [
            {
                "type": "Rapid Layering",
                "severity": "High",
                "description": f"Immediate pass-through of ${r['amt']:,.0f} detected.",
            }
            for r in results
        ]

    def _check_structuring(self, name):
        """
        Sequential-cluster proxy: many sub-£10k transfers that cumulatively
        exceed the reporting threshold — classic 'smurfing'.
        """
        query = """
        MATCH (s)-[t:TRANSFER]->(r:Person {name: $name})
        WHERE t.amount < 10000
        WITH r, sum(t.amount) AS total, count(t) AS cnt
        WHERE total > 50000
        RETURN r.name AS entity, total, cnt
        """
        results = self.run_query(query, {"name": name})
        return [
            {
                "type": "Structuring (Smurfing)",
                "severity": "Medium",
                "description": (
                    f"Received ${r['total']:,.0f} via {r['cnt']} "
                    "sub-threshold deposits."
                ),
            }
            for r in results
        ]

    def _check_dormant(self, name):
        """
        Isolation-Forest proxy: account had no activity for >365 days then
        received/sent a transaction with a Z-Score amount anomaly.
        We approximate this by checking a large time gap between first and
        last transaction combined with a final transaction >> expected_volume.
        """
        query = """
        MATCH (p:Person {name: $name})
        WITH p,
             p.expected_volume AS declared
        MATCH (p)-[t:TRANSFER]->()
        WITH p, declared,
             min(t.timestamp) AS first_tx,
             max(t.timestamp) AS last_tx,
             max(t.amount)    AS peak_amount,
             count(t)         AS tx_count
        WHERE duration.between(first_tx, last_tx).days > 365
          AND peak_amount > declared * 10
        RETURN p.name AS entity, peak_amount, declared
        """
        results = self.run_query(query, {"name": name})
        return [
            {
                "type": "Dormant Account Activation",
                "severity": "High",
                "description": (
                    f"Account dormant for >1 year; peak transfer "
                    f"${r['peak_amount']:,.0f} vs declared ${r['declared']:,.0f}/mo."
                ),
            }
            for r in results
        ]

    def run_intelligence_suite(self, target_entity):
        """Run all detectors against a single entity; return combined alerts."""
        alerts = []
        alerts.extend(self._check_circular(target_entity))
        alerts.extend(self._check_layering(target_entity))
        alerts.extend(self._check_structuring(target_entity))
        alerts.extend(self._check_dormant(target_entity))
        return alerts

    # ------------------------------------------------------------------ #
    #  TAGGING — writes detection RESULTS back onto the graph so Neo4j    #
    #  Browser can colour flagged nodes/edges without any hardcoding.     #
    # ------------------------------------------------------------------ #

    def tag_fraud_patterns(self):
        """
        Run every detector across all Person nodes.  When a pattern is found,
        write risk metadata onto the relevant nodes and relationships so
        Neo4j Browser can render them in a distinct visual style.

        This is ONLY called AFTER the detection queries have run — the tags
        are the OUTPUT of the algorithm, not pre-seeded labels.
        """
        with self.driver.session() as session:
            all_entities = session.run(
                "MATCH (p:Person) RETURN p.name AS name"
            )
            names = [r["name"] for r in all_entities]

        flagged = {}
        for name in names:
            alerts = self.run_intelligence_suite(name)
            if alerts:
                flagged[name] = [a["type"] for a in alerts]

        if not flagged:
            print("  [detector] No fraud patterns found — check your data.")
            return

        for entity_name, pattern_types in flagged.items():
            primary_type = pattern_types[0]

            # Tag the Person node
            self.run_query(
                """
                MATCH (p:Person {name: $name})
                SET p.risk_flag  = true,
                    p.risk_type  = $rtype,
                    p.risk_count = $count
                """,
                {
                    "name": entity_name,
                    "rtype": primary_type,
                    "count": len(pattern_types),
                },
            )

            # Tag the relationships that flow THROUGH this flagged node
            self.run_query(
                """
                MATCH ()-[t:TRANSFER]->(p:Person {name: $name})
                SET t.risk_flag = true, t.risk_type = $rtype
                """,
                {"name": entity_name, "rtype": primary_type},
            )
            self.run_query(
                """
                MATCH (p:Person {name: $name})-[t:TRANSFER]->()
                SET t.risk_flag = true, t.risk_type = $rtype
                """,
                {"name": entity_name, "rtype": primary_type},
            )

            print(f"  [detector] ⚠  {entity_name:30s} → {', '.join(pattern_types)}")

    # ------------------------------------------------------------------ #
    #  PATH TRACING — used by the live demo to isolate the evidence path  #
    # ------------------------------------------------------------------ #

    def trace_alert_path(self, entity_name, pattern_type):
        """Return the specific node names and transfer IDs in the pattern."""
        if "Circular" in pattern_type:
            query = """
            MATCH path = (p:Person {name: $name})-[:TRANSFER*2..4]->(p)
            RETURN [n IN nodes(path) | n.name] AS nodes,
                   [r IN relationships(path) | r.id] AS links
            LIMIT 1
            """
        elif "Layering" in pattern_type:
            query = """
            MATCH (s)-[in:TRANSFER]->(m:Person {name: $name})-[out:TRANSFER]->(r)
            RETURN [s.name, m.name, r.name] AS nodes,
                   [in.id, out.id] AS links
            LIMIT 1
            """
        else:  # Structuring / Dormant
            query = """
            MATCH (s)-[t:TRANSFER]->(r:Person {name: $name})
            WHERE t.amount < 10000
            RETURN collect(s.name) + r.name AS nodes,
                   collect(t.id) AS links
            """

        results = self.run_query(query, {"name": entity_name})
        if results:
            return {"nodes": results[0]["nodes"], "links": results[0]["links"]}
        return {"nodes": [], "links": []}