import asyncio
import websockets
import json

async def listen_to_telemetry():
    url = "ws://127.0.0.1:8000/v1/session/project_x/telemetry"
    print(f"Connecting to stream: {url}")
    
    async with websockets.connect(url) as websocket:
        print("Connected. Awaiting events...\n")
        try:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get("event") == "scratchpad_retry":
                    print(f"⚠️  [ALERT] Agent: {data['telemetry']['active_agent']} failed citation check!")
                    print(f"   └── {data['error_details']['rejected_facts_count']} facts rejected.\n")
                else:
                    print(f"✅ [METRIC] Chunk cleared by {data['telemetry']['active_agent']}")
                    eff = data['token_metrics']['compression_efficiency_delta'] * 100
                    print(f"   └── Compression Efficiency: {eff:.2f}%\n")
        except websockets.exceptions.ConnectionClosed:
            print("Connection severed.")

if __name__ == "__main__":
    asyncio.run(listen_to_telemetry())