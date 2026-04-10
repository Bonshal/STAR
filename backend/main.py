import asyncio
import pandas as pd
import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from orchestrator import IFFTOrchestrator 

api_app = FastAPI()
api_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = socketio.ASGIApp(sio, other_asgi_app=api_app)

ORC = IFFTOrchestrator("bolt://localhost:7687", "neo4j", "12345678")

async def _run_sim():
    df_tx = await asyncio.to_thread(pd.read_csv, "transactions.csv")
    await asyncio.to_thread(ORC.load_entities, "entities.csv")
    for _, row in df_tx.iterrows():
        tx_dict = row.to_dict()
        await asyncio.to_thread(ORC.ingest_transaction, tx_dict)
        await sio.emit("new_transaction", tx_dict)
        findings = await asyncio.to_thread(ORC.run_intelligence_suite, tx_dict['receiver'])
        for alert in findings:
            await sio.emit("fraud_alert", {
                "id": f"AL-{tx_dict['tx_id']}",
                "entity": tx_dict['receiver'],
                "type": alert['type'],
                "severity": alert['severity'],
                "timestamp": tx_dict['ts'],
                "description": alert['description']
            })
        await asyncio.sleep(1.5)
    await sio.emit("simulation_complete")

@sio.event
async def start_simulation(sid, data):
    sio.start_background_task(_run_sim)

@sio.event
async def request_path_trace(sid, data):
    """
    Client sends { entity: string, type: string }
    """
    path_data = await asyncio.to_thread(ORC.trace_alert_path, data['entity'], data['type'])
    await sio.emit("highlight_path", path_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
