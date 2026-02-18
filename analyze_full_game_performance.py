
import pandas as pd
import sqlite3
import numpy as np

def analyze_full_game():
    print("Loading data...")
    # 1. VOLATILITY DATA (For Mean & Std Analysis)
    vol_df = pd.read_csv('/Users/jonathanbierly/Desktop/Classes/Projects/Kalshi/reports/volatility/vol_comparison_data.csv')
    vol_df['game_id'] = vol_df['game_id'].astype(str).str.zfill(10)

    # Load final scores
    conn = sqlite3.connect('/Users/jonathanbierly/Desktop/Classes/Projects/Kalshi/data/nba_data.db')
    games_df = pd.read_sql_query("SELECT game_id, home_score, away_score FROM games", conn)
    conn.close()
    
    games_df['final_diff'] = games_df['home_score'] - games_df['away_score']
    
    # Merge Vol Data
    df_vol = pd.merge(vol_df, games_df[['game_id', 'final_diff']], on='game_id', how='inner')
    
    # Define Quarters
    bins = [0, 720, 1440, 2160, 2881]
    labels = ['Q4 (Late)', 'Q3', 'Q2', 'Q1 (Early)']
    df_vol['quarter'] = pd.cut(df_vol['seconds_remaining'], bins=bins, labels=labels)
    
    # 2. MARKETS DATA (For Brier Score Analysis)
    df_mk = pd.read_csv('/Users/jonathanbierly/Desktop/Classes/Projects/Kalshi/data/model_vs_market_comparison.csv')
    # Filter valid rows
    df_mk = df_mk.dropna(subset=['model_brier', 'market_brier'])
    df_mk['quarter'] = pd.cut(df_mk['seconds_remaining'], bins=bins, labels=labels)
    
    print("\n" + "="*95)
    print(f"{'Quarter':<12} | {'Metric':<15} | {'Model':<10} | {'Market':<10} | {'Verdict':<25}")
    print("="*95)
    
    for q in labels[::-1]: # Q1 -> Q4
        # Vol Stats
        q_vol = df_vol[df_vol['quarter'] == q].copy()
        
        # Market Stats
        q_mk = df_mk[df_mk['quarter'] == q].copy()
        
        if len(q_vol) == 0: continue
        
        print(f"\n--- {q} ---")
        
        # 1. MEAN ACCURACY (MAE)
        mae_model = abs(q_vol['final_diff'] - q_vol['model_mean']).mean()
        mae_market = abs(q_vol['final_diff'] - q_vol['market_loc']).mean()
        diff_mae = (mae_market - mae_model) / mae_market
        winner_mae = "✅ Model" if mae_model < mae_market else "❌ Market"
        print(f"{'':<12} | {'MAE (Error)':<15} | {mae_model:<10.2f} | {mae_market:<10.2f} | {winner_mae} ({diff_mae:+.1%})")
        
        # 2. VOLATILITY CALIBRATION
        # Realized Vol = RMS of residuals
        q_vol['resid'] = q_vol['final_diff'] - q_vol['model_mean']
        realized = np.sqrt((q_vol['resid']**2).mean())
        
        vol_model = q_vol['model_std'].mean()
        vol_market = q_vol['market_std'].mean()
        
        # Error from realized
        err_model_vol = abs(vol_model - realized)
        err_market_vol = abs(vol_market - realized)
        winner_vol = "✅ Model" if err_model_vol < err_market_vol else "❌ Market"
        
        print(f"{'':<12} | {'Realized Vol':<15} | {realized:<10.2f} | {'-':<10} | (Truth)")
        print(f"{'':<12} | {'Pred Vol':<15} | {vol_model:<10.2f} | {vol_market:<10.2f} | {winner_vol} (Err: {err_model_vol:.2f} vs {err_market_vol:.2f})")
        
        # 3. BRIER SCORE
        if len(q_mk) > 0:
            brier_model = q_mk['model_brier'].mean()
            brier_market = q_mk['market_brier'].mean()
            diff_brier = (brier_market - brier_model) / brier_market
            winner_brier = "✅ Model" if brier_model < brier_market else "❌ Market"
            print(f"{'':<12} | {'Brier Score':<15} | {brier_model:<10.4f} | {brier_market:<10.4f} | {winner_brier} ({diff_brier:+.1%})")
        else:
            print(f"{'':<12} | {'Brier Score':<15} | N/A        | N/A        | No Data")

if __name__ == "__main__":
    analyze_full_game()
