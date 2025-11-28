import time
import pandas as pd
from src.models.prediction import PredictionEngine
from data_acquisition import MockHistoricalDataClient, HistoricalDataClient

def test_pipeline_latency():
    print("Initializing Prediction Engine...")
    engine = PredictionEngine()
    
    print("Fetching Real Data (via HistoricalDataClient)...")
    client = HistoricalDataClient()
    # Use the game ID we know works or fetch one
    games = client.get_season_games()
    game_id = games[0] if games else '0022300001'
    pbp_df = client.get_game_pbp(game_id)
    
    print("Starting Latency Test (processing 50 events)...")
    latencies = []
    
    for _, row in pbp_df.iterrows():
        # Simulate live feed processing
        result = engine.process_event(row)
        latencies.append(result['latency_ms'])
        
        # Print first few results
        if len(latencies) <= 5:
            print(f"Prob: {result['probability']:.4f} (CI: {result['ci_lower']:.4f}-{result['ci_upper']:.4f}) | Latency: {result['latency_ms']:.2f}ms")
            
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    
    print(f"\nAverage Latency: {avg_latency:.2f} ms")
    print(f"Max Latency: {max_latency:.2f} ms")
    
    if avg_latency < 100:
        print("SUCCESS: Latency is under 100ms.")
    else:
        print("WARNING: Latency is over 100ms.")

if __name__ == "__main__":
    test_pipeline_latency()
