import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.training import prepare_training_data
from data.database import DatabaseManager

def get_corrs():
    print("Loading data...")
    X, _ = prepare_training_data()
    db = DatabaseManager()
    
    # Get final scores
    query = """
        SELECT game_id, home_score - away_score as final_diff 
        FROM (
            SELECT *, ROW_NUMBER() OVER(PARTITION BY game_id ORDER BY period DESC, remaining_time ASC) as rn 
            FROM pbp_events
        ) 
        WHERE rn = 1
    """
    final_diffs = pd.read_sql(query, db.engine)
    X = X.merge(final_diffs, on='game_id')
    X['score_remainder'] = X['final_diff'] - X['score_diff']
    
    # Filter to numeric only for corr
    numeric_X = X.select_dtypes(include=[np.number])
    corrs = numeric_X.corr()['score_remainder'].sort_values(ascending=False)
    
    print("\nTop Positive Correlations with score_remainder:")
    print(corrs.head(20))
    
    print("\nTop Negative Correlations with score_remainder:")
    print(corrs.tail(20))

if __name__ == "__main__":
    get_corrs()
