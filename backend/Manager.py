import pandas as pd
from neo4j import GraphDatabase
import datetime

class IFFTManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri,auth=(user, password))


    def close(self):
        self.driver.close()

    def run_query(self, query, parameters = None):
        with self.driver.sesion() as session:
            result = session.run(query, parameters)
            return [record for record in result]
        
        #datra ingestion
    def ingest_transaction(self, tx_data): 
        """
        tx_data: Dictionary containing sender, receiver, amount, timestamp, channel
        Reasoning: MERGE creates nodes only if they don't exist. 
        Using 'ON CREATE SET' ensures profile info is only set once.
        """

        query = """
        MERGE (s:Person {name: $sender_name})
        ON CREATE SET s.occupation = $sender_job, s.expected_volume = $expected_vol
        
        MERGE (r:Person {name: $receiver_name})
        
        MERGE (s)-[t:TRANSFER {
            id: $tx_id, 
            amount: toFloat($amount), 
            timestamp: datetime($ts),
            channel: $channel
        }]->(r)
        """
        self.run_query(query, tx_data)

    def detect_circular_flow(self, max_hops = 4):
        """
        Finds money starting and ending at the same Person.
        Returns: List of 'circles' for the Evidence Package.
        """

        query = f"""
        MATCH path = (p:Person)-[:TRANSFER*2..{max_hops}]->(p)
        WHERE all(t IN relationships(path) WHERE t.amount > 0)
        RETURN p.name as entity, 
            [t in relationships(path) | t.amount] as amounts,
            length(path) as hops
        """

        return self.run_query(query)
    
    def detect_rapid_layering(self, time_window_minutes=30):
        """
        Logic: Find nodes that act as 'hubs' where IN-flow and OUT-flow 
        happen almost simultaneously.
        """
        query = """
        MATCH (s)-[in:TRANSFER]->(m:Person)-[out:TRANSFER]->(r)
        WHERE duration.between(in.timestamp, out.timestamp).minutes < $window
        AND in.amount >= (out.amount * 0.95) // Catching nearly full pass-throughs
        RETURN m.name as layering_node, in.amount as amount_in, out.amount as amount_out
        """
        return self.run_query(query, {"window": time_window_minutes})
    

    def check_profile_mismatch(self, entity_name):
        #1. Get transaction from history from neo4j
        history = self.run_query("""
        MATCH (p:Person {name: $name})-[t:TRANSFER]->()
        RETURN p.expected_volume as declared, t.amount as actual
    """, {"name": entity_name})
        
        df = df.DataFrame(history)
        if df.empty:
            return 0
        #2. statistical Anomaly : is actual volume > 5x declared?
        actual_total = df['actual'].sum()
        declared_limit = df['declared'].iloc[0]


    def generate_fiu_package(self, suspicious_entity):
        report = {
            "entity": suspicious_entity,
            "timestamp": datetime.datetime.now().isoformat(),
            "alerts": [],
            "risk_summary": 0
        }
        
        # Run all checks
        layering = self.detect_rapid_layering()
        circles = self.detect_circular_flows()
        mismatch_score = self.check_profile_mismatch(suspicious_entity)
        
        # Build narrative
        if mismatch_score > 70:
            report["alerts"].append(f"CRITICAL: Profile Mismatch detected ({mismatch_score}% deviation).")
        
        # Logic to filter specific entity from layering/circles results
        # ...
        
        return report

    