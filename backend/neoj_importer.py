import pandas as pd
from neo4j import GraphDatabase
import time

class Neo4jImporter:
    def __init__(self, uri, user, password):
        # Using the URI from your screenshot
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_data(self, csv_path):
        if not pd.io.common.file_exists(csv_path):
            print(f"❌ Error: {csv_path} not found!")
            return

        df = pd.read_csv(csv_path)
        print(f"📦 Starting import of {len(df)} transactions...")

        with self.driver.session() as session:
            # 1. CLEAN SLATE: Wipe existing data for a fresh demo run
            print("🧹 Wiping existing graph data...")
            session.run("MATCH (n) DETACH DELETE n")

            # 2. PERFORMANCE: Create uniqueness constraints
            print("⚙️ Setting up graph constraints...")
            session.run("CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE")

            # 3. THE HEAVY LIFTING: Batched UNWIND
            # This is the industry-standard way to move large CSVs into Neo4j
            query = """
            UNWIND $rows AS row
            MERGE (s:Person {id: row.sender})
            ON CREATE SET s.status = 'STABLE', s.risk_score = 0, s.is_fraud = false, s.in_loop = false
            
            MERGE (r:Person {id: row.receiver})
            ON CREATE SET r.status = 'STABLE', r.risk_score = 0, r.is_fraud = false, r.in_loop = false
            
            CREATE (s)-[t:TRANSACT {
                tx_id: row.id,
                amount: toFloat(row.amount),
                timestamp: row.ts,
                type: row.type,
                label: toInteger(row.label),
                is_alert: false,
                in_loop: false,
                risk_score: 0
            }]->(r)
            """

            start_time = time.time()
            batch_size = 2500
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i : i + batch_size].to_dict('records')
                session.run(query, rows=batch)
                print(f"   ✅ Processed {i + len(batch)} / {len(df)} rows...")

            end_time = time.time()
            print(f"\n✨ Graph construction complete in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Credentials matching your Neo4j Desktop screenshot
    # Check if your password is 'password' or something else you set!
    URI = "neo4j://127.0.0.1:7687" 
    USER = "neo4j"
    PWD = "12345678" 

    importer = Neo4jImporter(URI, USER, PWD)
    try:
        importer.import_data("inference_test_data.csv")
    finally:
        importer.close()