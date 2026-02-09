import pandas as pd
print("DEBUG: Using spread_src/features/engineering.py")
import numpy as np
from src.data.database import DatabaseManager, PlayerAdvancedStats, TeamBasicStats, Game
from datetime import timedelta

class TeamStatsEngine:
    """
    Calculates and provides team-level features (Season & Recent)
    based strictly on PRIOR games to prevent leakage.
    """
    def __init__(self):
        self.db = DatabaseManager()
        self.raw_df = self._load_data()
        self.features_df = self._compute_rolling_features()
        self.latest_team_stats = self._compute_latest_stats()
        # PRE-CACHE for fast lookup: (game_id, team_id) -> row
        self._cache = {}
        for _, row in self.features_df.iterrows():
            self._cache[(row['game_id'], row['team_id'])] = row.to_dict()
        
    def _load_data(self):
        """Loads all team basic stats sorted by date."""
        # TeamBasicStats now has game_date, so we don't strictly need to join Game for date
        # But let's join just to be safe and use Game.date as the source of truth for sorting
        query = """
        SELECT t.*, g.season
        FROM team_basic_stats t
        JOIN games g ON t.game_id = g.game_id
        ORDER BY g.date ASC
        """
        return pd.read_sql(query, self.db.engine)
        
    def _compute_rolling_features(self):
        """Computes season-to-date and last-10 stats for every game-team combo."""
        df = self.raw_df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Calculate derived metrics
        df['possessions'] = 0.96 * (df['fga'] + 0.44 * df['fta'] - df['oreb'] + df['tov'])
        df['possessions'] = df['possessions'].replace(0, 1)
        df['off_rtg'] = 100 * df['pts'] / df['possessions']
        df['pace'] = 48 * df['possessions'] / df['min'].replace(0, 48)
        
        # Self-join to get opponent stats for defensive rating
        opp_df = df[['game_id', 'team_id', 'pts', 'possessions']].rename(
            columns={'team_id': 'opp_id', 'pts': 'opp_pts', 'possessions': 'opp_poss'}
        )
        
        df_merged = pd.merge(df, opp_df, on='game_id')
        df_merged = df_merged[df_merged['team_id'] != df_merged['opp_id']]
        
        df_merged['def_rtg'] = 100 * df_merged['opp_pts'] / df_merged['opp_poss']
        df_merged['win'] = df_merged['wl'].apply(lambda x: 1 if x == 'W' else 0)
        df_merged['win_margin'] = df_merged['pts'] - df_merged['opp_pts']
        
        # Quarter scoring consistency proxy
        df_merged['pts_var'] = df_merged['pts']
        
        # New defensive stats
        df_merged['stl_rate'] = df_merged['stl'] / df_merged['possessions']
        df_merged['blk_rate'] = df_merged['blk'] / df_merged['possessions']
        
        # Calculate SOS (Strength of Schedule)
        # We need the opponent's win percentage entering the game.
        # This is a bit of a circular dependency in rolling, so we'll do an approximating merge.
        # First, we need a lookup for (team, game_id) -> win_pct
        df_merged['game_date'] = pd.to_datetime(df_merged['game_date'])
        
        # We'll compute the win_pct separately and then merge back for SOS
        win_pct_lookup = []
        for (team_id, season), team_df in df_merged.groupby(['team_id', 'season']):
            team_df = team_df.sort_values('game_date')
            wp = team_df['win'].expanding().mean().shift(1).fillna(0.5)
            lookup = pd.DataFrame({
                'game_id': team_df['game_id'],
                'team_id': team_id,
                'prior_win_pct': wp
            })
            win_pct_lookup.append(lookup)
        
        wp_df = pd.concat(win_pct_lookup)
        
        # Merge opponent's prior win pct back into df_merged
        df_merged = pd.merge(
            df_merged, 
            wp_df.rename(columns={'team_id': 'opp_id', 'prior_win_pct': 'opp_prior_win_pct'}),
            on=['game_id', 'opp_id'],
            how='left'
        )
        
        features = []
        
        # Group by Team AND Season to prevent stats leaking across seasons
        for (team_id, season), team_df in df_merged.groupby(['team_id', 'season']):
            team_df = team_df.sort_values('game_date')
            
            # Season-to-Date (Expanding, shifted to exclude current game)
            season_stats = team_df[['off_rtg', 'def_rtg', 'win', 'pace', 'stl_rate', 'blk_rate', 'win_margin']].expanding().mean().shift(1)
            season_stats.columns = [f'team_season_{c}' for c in season_stats.columns]
            season_stats = season_stats.rename(columns={'team_season_win': 'team_season_win_pct'})
            
            # Recent (Last 10, shifted)
            # Add SOS and consistency (pts variance) to recent stats
            recent_stats = team_df[['off_rtg', 'def_rtg', 'win', 'pace', 'stl_rate', 'blk_rate', 'win_margin', 'opp_prior_win_pct', 'pts_var']].rolling(window=10, min_periods=1).mean().shift(1)
            # For consistency, we want the rolling STDEV of points
            pts_consistency = team_df['pts'].rolling(window=10, min_periods=1).std().shift(1)
            
            recent_stats.columns = [f'team_recent_{c}' for c in recent_stats.columns]
            recent_stats = recent_stats.rename(columns={'team_recent_win': 'team_recent_win_pct', 'team_recent_opp_prior_win_pct': 'team_recent_SOS'})
            recent_stats['team_recent_scoring_consistency'] = pts_consistency
            
            # Combine
            team_feats = pd.concat([team_df[['game_id', 'team_id', 'game_date', 'side']], season_stats, recent_stats], axis=1)
            
            # Add Context
            team_feats['rest_days'] = team_df['game_date'].diff().dt.days.fillna(3)
            team_feats['is_home'] = team_df['side'].apply(lambda x: 1 if x == 'Home' else 0)
            
            features.append(team_feats)
            
        return pd.concat(features)

    def _compute_latest_stats(self):
        """
        Computes the LATEST available stats for each team (unshifted).
        This represents the stats entering the NEXT game (i.e., a live game).
        """
        df = self.raw_df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Calculate derived metrics
        df['possessions'] = 0.96 * (df['fga'] + 0.44 * df['fta'] - df['oreb'] + df['tov'])
        df['possessions'] = df['possessions'].replace(0, 1)
        df['off_rtg'] = 100 * df['pts'] / df['possessions']
        df['pace'] = 48 * df['possessions'] / df['min'].replace(0, 48)
        
        # Self-join to get opponent stats
        opp_df = df[['game_id', 'team_id', 'pts', 'possessions']].rename(
            columns={'team_id': 'opp_id', 'pts': 'opp_pts', 'possessions': 'opp_poss'}
        )
        
        df_merged = pd.merge(df, opp_df, on='game_id')
        df_merged = df_merged[df_merged['team_id'] != df_merged['opp_id']]
        
        df_merged['def_rtg'] = 100 * df_merged['opp_pts'] / df_merged['opp_poss']
        df_merged['win'] = df_merged['wl'].apply(lambda x: 1 if x == 'W' else 0)
        df_merged['win_margin'] = df_merged['pts'] - df_merged['opp_pts']
        
        # New defensive stats
        df_merged['stl_rate'] = df_merged['stl'] / df_merged['possessions']
        df_merged['blk_rate'] = df_merged['blk'] / df_merged['possessions']
        
        # Calculate SOS for latest stats
        df_merged['game_date'] = pd.to_datetime(df_merged['game_date'])
        win_pct_lookup = []
        for (team_id, season), team_df in df_merged.groupby(['team_id', 'season']):
            team_df = team_df.sort_values('game_date')
            wp = team_df['win'].expanding().mean().fillna(0.5)
            lookup = pd.DataFrame({
                'game_id': team_df['game_id'],
                'team_id': team_id,
                'prior_win_pct': wp
            })
            win_pct_lookup.append(lookup)
        wp_df = pd.concat(win_pct_lookup)
        df_merged = pd.merge(
            df_merged, 
            wp_df.rename(columns={'team_id': 'opp_id', 'prior_win_pct': 'opp_prior_win_pct'}),
            on=['game_id', 'opp_id'],
            how='left'
        )
        
        latest_stats = {}
        
        # Group by Team AND Season
        for (team_id, season), team_df in df_merged.groupby(['team_id', 'season']):
            team_df = team_df.sort_values('game_date')
            
            if team_df.empty: continue
                
            # Season-to-Date (Expanding, NO SHIFT)
            season_stats = team_df[['off_rtg', 'def_rtg', 'win', 'pace', 'stl_rate', 'blk_rate', 'win_margin']].expanding().mean().iloc[-1]
            
            # Recent (Last 10, NO SHIFT)
            recent_stats = team_df[['off_rtg', 'def_rtg', 'win', 'pace', 'stl_rate', 'blk_rate', 'win_margin', 'opp_prior_win_pct']].rolling(window=10, min_periods=1).mean().iloc[-1]
            
            # Rest Days relative to now logic stays in get_latest_features
            last_game_date = team_df['game_date'].iloc[-1]
            
            stats = {}
            # Season
            stats['team_season_off_rtg'] = season_stats['off_rtg']
            stats['team_season_def_rtg'] = season_stats['def_rtg']
            stats['team_season_win_pct'] = season_stats['win']
            stats['team_season_pace'] = season_stats['pace']
            stats['team_season_stl_rate'] = season_stats['stl_rate']
            stats['team_season_blk_rate'] = season_stats['blk_rate']
            stats['team_season_win_margin'] = season_stats['win_margin']
            
            # Recent
            stats['team_recent_off_rtg'] = recent_stats['off_rtg']
            stats['team_recent_def_rtg'] = recent_stats['def_rtg']
            stats['team_recent_win_pct'] = recent_stats['win']
            stats['team_recent_pace'] = recent_stats['pace']
            stats['team_recent_stl_rate'] = recent_stats['stl_rate']
            stats['team_recent_blk_rate'] = recent_stats['blk_rate']
            stats['team_recent_win_margin'] = recent_stats['win_margin']
            stats['team_recent_SOS'] = recent_stats['opp_prior_win_pct']
            stats['team_recent_scoring_consistency'] = team_df['pts'].rolling(window=10, min_periods=1).std().iloc[-1]
            
            stats['last_game_date'] = last_game_date
            
            latest_stats[team_id] = stats
            
        return latest_stats

    def get_latest_features(self, team_id):
        """Returns the latest available features for a team (for live inference)."""
        if team_id not in self.latest_team_stats:
            return {}
            
        stats = self.latest_team_stats[team_id].copy()
        last_date = stats.pop('last_game_date')
        
        # Calculate rest days relative to NOW
        rest_days = (pd.Timestamp.now() - last_date).days
        stats['rest_days'] = rest_days
        
        # 'is_home' and 'side' are context dependent, caller must handle or we pass it in?
        # The caller (Orchestrator) knows if this team is home or away in the LIVE game.
        # But the model expects 'is_home' feature.
        # We return the raw stats, Orchestrator adds 'is_home'.
        
        # Wait, get_features returns 'home_...' or 'away_...' keys.
        # Let's return generic keys and let Orchestrator rename.
        return stats

    def get_features(self, game_id, home_team_id, away_team_id):
        """Returns a dict of features for the game (OPTIMIZED)."""
        home_row = self._cache.get((game_id, home_team_id))
        away_row = self._cache.get((game_id, away_team_id))
        
        feat_dict = {}
        if home_row:
            for col, val in home_row.items():
                if col not in ['game_id', 'team_id', 'game_date']:
                    feat_dict[f'home_{col}'] = val
        if away_row:
            for col, val in away_row.items():
                if col not in ['game_id', 'team_id', 'game_date']:
                    feat_dict[f'away_{col}'] = val
        return feat_dict

        return feat_dict

class RosterEngine:
    """
    Calculates roster-aggregated features based on the players who actually played.
    """
    def __init__(self):
        self.db = DatabaseManager()
        self.raw_df = self._load_data()
        self.player_features = self._compute_player_rolling_features()
        self.latest_player_stats = self._compute_latest_player_stats()
        # PRE-CACHE
        print("Caching roster features...")
        self._cache = {}
        # Group by game and team once
        for (gid, tid), group in self.player_features.groupby(['game_id', 'team_id']):
            # We still need the weighting logic, but we can pre-calculate it for each game-team once
            self._cache[(gid, tid)] = self._precalc_roster_stats(group)
    
    def _precalc_roster_stats(self, team_p):
        """Internal helper to calculate weighted stats for a group of players."""
        item = {}
        # Season Stats
        season_cols = [c for c in team_p.columns if 'player_season_' in c and 'min_float' not in c]
        season_weights = team_p['player_season_min_float'].fillna(0)
        if season_weights.sum() > 0:
            for col in season_cols:
                metric_name = col.replace('player_', 'roster_')
                valid = team_p[col].notna()
                if valid.sum() > 0:
                    w = season_weights[valid]
                    if w.sum() > 0:
                        item[metric_name] = np.average(team_p.loc[valid, col], weights=w)
        
        # Recent Stats
        recent_cols = [c for c in team_p.columns if 'player_recent_' in c and 'min_float' not in c]
        recent_weights = team_p['player_recent_min_float'].fillna(0)
        
        # Add unweighted averages for PIE to catch rotation shifts
        if 'player_recent_pie' in team_p.columns:
            item['roster_recent_pie_unweighted'] = team_p['player_recent_pie'].mean()
        if 'player_season_pie' in team_p.columns:
            item['roster_season_pie_unweighted'] = team_p['player_season_pie'].mean()

        if recent_weights.sum() > 0:
            for col in recent_cols:
                metric_name = col.replace('player_', 'roster_')
                valid = team_p[col].notna()
                if valid.sum() > 0:
                    w = recent_weights[valid]
                    if w.sum() > 0:
                        item[metric_name] = np.average(team_p.loc[valid, col], weights=w)
        
        # Star Impact: Top 3 and Top 5 PIE (Weighted average of the heavy hitters)
        if 'player_recent_pie' in team_p.columns:
            # Sort by PERFORMANCE (PIE) to find the stars
            top_players = team_p.sort_values('player_recent_pie', ascending=False).head(5)
            # Weighted PIE of Top 3
            t3 = top_players.head(3)
            # Use minutes as weights for the average of these stars
            w3 = t3['player_recent_min_float'].fillna(10).clip(lower=1)
            item['roster_recent_top3_pie'] = np.average(t3['player_recent_pie'].fillna(0), weights=w3)
                
            # Missing Starter Count: How many of the Top 5 starters are effectively "gone"?
            # Since this is a per-game feature, and we are in _precalc_roster_stats,
            # this logic is slightly different during live inference.
            # In training data, we just use the players who actually played.
            # So "Missing_Starter_Count" doesn't quite apply here in historical view.
        
        return item
        
    def _load_data(self):
        """Loads all player advanced stats sorted by date."""
        query = """
        SELECT p.*, g.date as game_date, g.season
        FROM player_advanced_stats p
        JOIN games g ON p.game_id = g.game_id
        ORDER BY g.date ASC
        """
        return pd.read_sql(query, self.db.engine)
        
    def _parse_minutes(self, min_str):
        """Converts 'MM:SS' or 'MM' to float minutes."""
        try:
            if ':' in str(min_str):
                parts = str(min_str).split(':')
                return float(parts[0]) + float(parts[1])/60
            return float(min_str)
        except:
            return 0.0

    def _compute_player_rolling_features(self):
        """
        Computes season and recent stats for every player-game.
        """
        df = self.raw_df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['min_float'] = df['minutes'].apply(self._parse_minutes)
        
        # Metrics to track
        # Switched to estimated ratings and removed net_rating
        metrics = ['est_off_rating', 'est_def_rating', 'pie', 'est_usg_pct', 'ts_pct', 'ast_pct']
        
        features = []
        
        # This might be slow if we iterate 500 players. 
        # But pandas groupby apply is decent.
        
        # Group by Player AND Season
        for (player_id, season), p_df in df.groupby(['player_id', 'season']):
            p_df = p_df.sort_values('game_date')
            
            # Metrics to roll (include minutes now)
            cols_to_roll = metrics + ['min_float']
            
            # Shift(1) to use only prior games
            season_stats = p_df[cols_to_roll].expanding().mean().shift(1)
            season_stats.columns = [f'player_season_{c}' for c in season_stats.columns]
            
            recent_stats = p_df[cols_to_roll].rolling(window=10, min_periods=1).mean().shift(1)
            recent_stats.columns = [f'player_recent_{c}' for c in recent_stats.columns]
            
            # Combine
            p_feats = pd.concat([p_df[['game_id', 'team_id', 'player_id', 'game_date']], season_stats, recent_stats], axis=1)
            features.append(p_feats)
            
        return pd.concat(features)

    def _compute_latest_player_stats(self):
        """
        Computes the LATEST unshifted stats for every player.
        """
        df = self.raw_df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['min_float'] = df['minutes'].apply(self._parse_minutes)
        
        metrics = ['est_off_rating', 'est_def_rating', 'pie', 'est_usg_pct', 'ts_pct', 'ast_pct']
        cols_to_roll = metrics + ['min_float']
        
        latest_stats_dict = {}
        
        # Group by Player AND Season
        for (player_id, season), p_df in df.groupby(['player_id', 'season']):
            p_df = p_df.sort_values('game_date')
            
            if p_df.empty: continue
            
            # Season (Expanding, NO SHIFT)
            season_vals = p_df[cols_to_roll].expanding().mean().iloc[-1]
            season_vals.index = [f'player_season_{c}' for c in season_vals.index]
            
            # Recent (Rolling, NO SHIFT)
            recent_vals = p_df[cols_to_roll].rolling(window=10, min_periods=1).mean().iloc[-1]
            recent_vals.index = [f'player_recent_{c}' for c in recent_vals.index]
            
            # Combine
            combined = pd.concat([season_vals, recent_vals])
            combined['player_id'] = player_id
            combined['team_id'] = p_df['team_id'].iloc[-1] # Most recent team
            
            # Overwrite to keep only the latest season's stats
            latest_stats_dict[player_id] = combined
            
        if not latest_stats_dict:
            return pd.DataFrame()
            
        return pd.DataFrame(list(latest_stats_dict.values()))
        
    def get_features(self, game_id, home_team_id, away_team_id):
        """Returns aggregated player stats for the game (OPTIMIZED)."""
        home_feats = self._cache.get((game_id, home_team_id), {})
        away_feats = self._cache.get((game_id, away_team_id), {})
        
        feat_dict = {}
        for k, v in home_feats.items():
            feat_dict[f'home_{k}'] = v
        for k, v in away_feats.items():
            feat_dict[f'away_{k}'] = v
        return feat_dict

    def get_projected_roster_features(self, team_id, player_ids=None):
        """
        Calculates roster features for a specific list of players (e.g. active roster).
        If player_ids is None, falls back to top 10 players by recent minutes.
        """
        # We need the latest 'player_season_...' and 'player_recent_...' values for each player
        # We use self.latest_player_stats which is UNSHIFTED (correct for live inference)
        
        if self.latest_player_stats.empty:
            return {}
            
        team_players = self.latest_player_stats[self.latest_player_stats['team_id'] == team_id]
        
        if team_players.empty:
            return {}
            
        if player_ids:
            # Filter for provided players
            active_players = team_players[team_players['player_id'].isin(player_ids)]
            print(f"  [RosterEngine] Using {len(active_players)} live player IDs for team {team_id}")
        else:
            # Fallback: Top 10 by recent minutes
            print(f"  [RosterEngine] WARNING: No player IDs provided for team {team_id}. Falling back to top 10 players.")
            active_players = team_players.sort_values('player_recent_min_float', ascending=False).head(10)
            
        if active_players.empty:
            return {}
            
        feat_dict = {}
        prefix = 'home' # This method returns generic keys, caller handles prefixing? 
        # Actually, caller usually expects home_... or away_...
        # Let's return generic keys and let caller rename, OR ask for prefix.
        # Let's return generic keys (roster_...) and let caller handle.
        
        # Calculate Weighted Stats
        # Season
        season_cols = [c for c in active_players.columns if 'player_season_' in c and 'min_float' not in c]
        season_weights = active_players['player_season_min_float'].fillna(0)
        
        if season_weights.sum() > 0:
            for col in season_cols:
                metric_name = col.replace('player_', 'roster_')
                valid = active_players[col].notna()
                if valid.sum() > 0:
                    w = season_weights[valid]
                    if w.sum() > 0:
                        val = np.average(active_players.loc[valid, col], weights=w)
                        feat_dict[metric_name] = val
                    else:
                        feat_dict[metric_name] = 0.0
                else:
                    feat_dict[metric_name] = 0.0
        else:
             for col in season_cols:
                metric_name = col.replace('player_', 'roster_')
                feat_dict[metric_name] = 0.0
                
        # Recent
        recent_cols = [c for c in active_players.columns if 'player_recent_' in c and 'min_float' not in c]
        recent_weights = active_players['player_recent_min_float'].fillna(0)
        
        # Add unweighted averages for PIE
        if 'player_recent_pie' in active_players.columns:
            feat_dict['roster_recent_pie_unweighted'] = active_players['player_recent_pie'].mean()
        if 'player_season_pie' in active_players.columns:
            feat_dict['roster_season_pie_unweighted'] = active_players['player_season_pie'].mean()

        if recent_weights.sum() > 0:
            for col in recent_cols:
                metric_name = col.replace('player_', 'roster_')
                valid = active_players[col].notna()
                if valid.sum() > 0:
                    w = recent_weights[valid]
                    if w.sum() > 0:
                        val = np.average(active_players.loc[valid, col], weights=w)
                        feat_dict[metric_name] = val
                    else:
                        feat_dict[metric_name] = 0.0
                else:
                    feat_dict[metric_name] = 0.0
        else:
            for col in recent_cols:
                metric_name = col.replace('player_', 'roster_')
                feat_dict[metric_name] = 0.0
        
        # Star Impact (Live Inference version)
        if not active_players.empty and 'player_recent_pie' in active_players.columns:
            # Top 3 PIE (Performance-based)
            top3 = active_players.sort_values('player_recent_pie', ascending=False).head(3)
            w3 = top3['player_recent_min_float'].fillna(1).clip(lower=1)
            feat_dict['roster_recent_top3_pie'] = np.average(top3['player_recent_pie'].fillna(0), weights=w3)
            
            # Missing Starter Count
            # 1. Identify "Expected Starters" (Top 5 players by recent mins in the ENTIRE team pool)
            entire_team = self.latest_player_stats[self.latest_player_stats['team_id'] == team_id]
            expected_starters = entire_team.sort_values('player_recent_min_float', ascending=False).head(5)['player_id'].tolist()
            # 2. See how many are in our provided active_players list
            active_ids = active_players['player_id'].tolist()
            missing = [pid for pid in expected_starters if pid not in active_ids]
            feat_dict['roster_missing_starter_count'] = len(missing)
            
        return feat_dict

class FeatureEngine:
    """
    Maintains the state of a game and calculates real-time features for inference.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.home_stats = {'fgm': 0, 'fga': 0, 'fg3m': 0, 'to': 0, 'reb': 0, 'pts': 0, 'fta': 0, 'oreb': 0, 'stl': 0, 'blk': 0}
        self.away_stats = {'fgm': 0, 'fga': 0, 'fg3m': 0, 'to': 0, 'reb': 0, 'pts': 0, 'fta': 0, 'oreb': 0, 'stl': 0, 'blk': 0}
        self.history = [] # List of (seconds_remaining, score_diff)
        self.points_per_q = {1: [], 2: [], 3: [], 4: []} # Historical pts for consistency
        self.current_features = {}
        self.lead_changes = 0
        self.last_leader = 0 # 1 for home, -1 for away, 0 for tie

    def update(self, event_row: pd.Series):
        """Updates game state based on a single PBP event row and returns the new feature vector."""
        event_type = event_row.get('event_type', 0)
        description = str(event_row.get('description', '')).lower()
        
        # Determine which team performed the action
        is_home = (event_row.get('player_team_id') == event_row.get('home_team_id'))
        stats = self.home_stats if is_home else self.away_stats
        
        # Update Stats based on Event Type
        event_type_str = str(event_type).lower()
        
        if event_type == 1 or 'made shot' in event_type_str: # Make
            stats['fgm'] += 1
            stats['fga'] += 1
            pts = 3 if '3pt' in description else 2
            stats['pts'] += pts
            if pts == 3: stats['fg3m'] += 1
        elif event_type == 2 or 'missed shot' in event_type_str: # Miss
            stats['fga'] += 1
        elif event_type == 4 or 'rebound' in event_type_str: # Rebound
            stats['reb'] += 1
            if 'offensive' in description: stats['oreb'] += 1
        elif event_type == 5 or 'turnover' in event_type_str: # Turnover
            stats['to'] += 1
            if 'steal' in description:
                # The OTHER team got the steal
                other_stats = self.away_stats if is_home else self.home_stats
                other_stats['stl'] += 1
        elif 'free throw' in event_type_str:
            stats['fta'] += 1
            if 'miss' not in description:
                stats['pts'] += 1
        elif 'block' in description:
            # The OTHER team got the block
            other_stats = self.away_stats if is_home else self.home_stats
            other_stats['blk'] += 1
            
        # Time remaining logic
        seconds_remaining = event_row['remaining_time']
        if event_row['period'] <= 4:
            seconds_remaining += (4 - event_row['period']) * 720
            
        # Update history for momentum (keep last 10 mins)
        new_diff = event_row['score_diff']
        self.history.append((seconds_remaining, new_diff))
        
        # Track Lead Changes
        current_leader = np.sign(new_diff) if new_diff != 0 else 0
        if current_leader != 0 and self.last_leader != 0 and current_leader != self.last_leader:
            self.lead_changes += 1
        if current_leader != 0:
            self.last_leader = current_leader

        if len(self.history) > 1000: # Safety cap
             self.history = [h for h in self.history if h[0] < seconds_remaining + 600]

        return self.calculate_current_features(event_row['score_diff'], seconds_remaining, event_row['period'], event_row['game_id'], event_row['home_team_id'], event_row['away_team_id'])

    def calculate_current_features(self, score_diff, seconds_remaining, period, game_id, home_id, away_id):
        """Calculates features from current internal state."""
        # Multi-window Momentum: Change in score diff over various windows
        windows = {
            'momentum_2min': 120,
            'momentum_4min': 240,
            'momentum_6min': 360,
            'momentum_8min': 480,
            'momentum_10min': 600
        }
        
        momentum_features = {}
        if self.history:
            for feat_name, window_secs in windows.items():
                start_score = self.history[0][1]
                for h_time, h_score in reversed(self.history):
                    if h_time > seconds_remaining + window_secs:
                        start_score = h_score
                        break
                momentum_features[feat_name] = score_diff - start_score
        else:
            momentum_features = {feat_name: 0 for feat_name in windows}
        
        # Original 5-min momentum for backward compatibility (if needed)
        # But we'll mostly use the new ones
        momentum_5min = momentum_features.get('momentum_6min', 0) # approximation or keep as is
        start_score_5min = self.history[0][1] if self.history else 0
        if self.history:
            for h_time, h_score in reversed(self.history):
                if h_time > seconds_remaining + 300:
                    start_score_5min = h_score
                    break
        momentum_5min = score_diff - start_score_5min
            
        # Lead Changes: Count lead swaps since game start
        current_leader = 1 if score_diff > 0 else (-1 if score_diff < 0 else 0)
        if current_leader != 0 and self.last_leader != 0 and current_leader != self.last_leader:
            self.lead_changes += 1
        if current_leader != 0:
            self.last_leader = current_leader
            
        # Score Volatility: Std dev of score diff over last 5 minutes
        score_volatility = 0
        if self.history:
            window_scores = [h[1] for h in self.history if h[0] > seconds_remaining and h[0] <= seconds_remaining + 300]
            if len(window_scores) > 1:
                score_volatility = np.std(window_scores)

        # Live Pace: possessions per 48 mins
        home_poss = 0.96 * (self.home_stats['fga'] + 0.44 * self.home_stats['fta'] - self.home_stats['oreb'] + self.home_stats['to'])
        away_poss = 0.96 * (self.away_stats['fga'] + 0.44 * self.away_stats['fta'] - self.away_stats['oreb'] + self.away_stats['to'])
        total_poss = home_poss + away_poss
        
        elapsed_mins = (2880 - seconds_remaining) / 60
        live_pace = (total_poss / max(elapsed_mins, 1)) * 48

        # 3P Reliance
        home_3p_reliance = (self.home_stats['fg3m'] * 3) / max(self.home_stats['pts'], 1)
        away_3p_reliance = (self.away_stats['fg3m'] * 3) / max(self.away_stats['pts'], 1)

        # Live Defensive Rates (per minute elapsed)
        elapsed_mins = max((2880 - seconds_remaining) / 60, 1)
        home_steal_rate = self.home_stats['stl'] / elapsed_mins
        away_steal_rate = self.away_stats['stl'] / elapsed_mins
        home_block_rate = self.home_stats['blk'] / elapsed_mins
        away_block_rate = self.away_stats['blk'] / elapsed_mins

        self.current_features = {
            'score_diff': score_diff,
            'seconds_remaining': seconds_remaining,
            'period': period,
            'lead_changes': self.lead_changes,
            'score_volatility': score_volatility,
            'home_efg': self._calc_efg(self.home_stats),
            'away_efg': self._calc_efg(self.away_stats),
            'turnover_diff': self.home_stats['to'] - self.away_stats['to'],
            'home_rebound_rate': self._calc_reb_rate(self.home_stats['reb'], self.away_stats['reb']),
            'required_catchup_rate': abs(score_diff) / (seconds_remaining + 1),
            'live_pace': live_pace,
            'score_momentum': momentum_5min,
            **momentum_features,
            'home_3p_reliance': home_3p_reliance,
            'away_3p_reliance': away_3p_reliance,
            'home_steal_rate': home_steal_rate,
            'away_steal_rate': away_steal_rate,
            'home_block_rate': home_block_rate,
            'away_block_rate': away_block_rate,
            'game_id': game_id,
            'home_team_id': home_id,
            'away_team_id': away_id
        }
        return self.current_features

    def _calc_efg(self, stats):
        if stats['fga'] == 0: return 0.0
        return (stats['fgm'] + 0.5 * stats['fg3m']) / stats['fga']

    def _calc_reb_rate(self, home_reb, away_reb):
        total = home_reb + away_reb
        if total == 0: return 0.5
        return home_reb / total

def create_live_features(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Batch processes a PBP DataFrame to generate features for every row."""
    engine = FeatureEngine()
    features = []
    
    for _, row in pbp_df.iterrows():
        # Ensure player_team_id exists for stat attribution
        if 'player_team_id' not in row:
            row['player_team_id'] = row['home_team_id'] # Fallback
            
        feat = engine.update(row)
        features.append(feat)
        
    return pd.DataFrame(features)

def add_advanced_features(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the PBP DataFrame with pre-game advanced features (Team Strength, Momentum, Roster).
    """
    print("Initializing Advanced Feature Engines...")
    team_engine = TeamStatsEngine()
    roster_engine = RosterEngine()
    
    # Get unique games
    games = pbp_df[['game_id', 'home_team_id', 'away_team_id']].drop_duplicates()
    
    advanced_features = []
    
    print(f"Generating advanced features for {len(games)} games...")
    for _, row in games.iterrows():
        game_id = row['game_id']
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
        # Get Team Stats
        team_feats = team_engine.get_features(game_id, home_id, away_id)
        
        # Get Roster Stats
        roster_feats = roster_engine.get_features(game_id, home_id, away_id)
        
        # Combine
        combined = {**team_feats, **roster_feats}
        combined['game_id'] = game_id
        
        advanced_features.append(combined)
        
    adv_df = pd.DataFrame(advanced_features)
    
    if adv_df.empty:
        print("Warning: No advanced features generated.")
        return pbp_df
        
    # Merge back to PBP
    # We merge on game_id. These features are constant for the whole game.
    merged_df = pd.merge(pbp_df, adv_df, on='game_id', how='left')
    
    
    return merged_df

# Advanced features - using only recent stats (removed season stats due to high correlation)
ADVANCED_FEATURES_LIST = [
    # Team performance
    'home_team_season_win_pct', 'home_team_season_off_rtg', 'home_team_season_def_rtg',
    'home_team_recent_off_rtg', 'home_team_recent_def_rtg', 'home_team_recent_win_pct', 'home_team_recent_pace',
    'home_team_recent_SOS', 'home_team_recent_scoring_consistency',
    'away_team_season_win_pct', 'away_team_season_off_rtg', 'away_team_season_def_rtg',
    'away_team_recent_off_rtg', 'away_team_recent_def_rtg', 'away_team_recent_win_pct', 'away_team_recent_pace',
    'away_team_recent_SOS', 'away_team_recent_scoring_consistency',
    
    # Rest 
    'home_rest_days', 'away_rest_days',
    
    # Roster recent performance (last 10 games)
    'home_roster_recent_est_off_rating', 'home_roster_recent_est_def_rating', 
    'home_roster_recent_est_usg_pct',
    'home_roster_recent_top3_pie', 'home_roster_missing_starter_count',
    'away_roster_recent_est_off_rating', 'away_roster_recent_est_def_rating', 
    'away_roster_recent_est_usg_pct',
    'away_roster_recent_top3_pie', 'away_roster_missing_starter_count'
]

VOLATILITY_FEATURES_LIST = [
    'lead_changes', 'score_volatility'
]

# Interaction features for variance prediction
# These help NGBoost learn that variance depends on BOTH time and margin
INTERACTION_FEATURES_LIST = [
    'time_x_margin', 'time_x_abs_margin', 'margin_squared',
    'log_time', 'time_proportion',
    'volatility_x_time',
    'home_season_margin_x_time', 'away_season_margin_x_time',
    'home_recent_margin_x_time', 'away_recent_margin_x_time',
    'home_pie_x_time', 'away_pie_x_time',
    'net_star_pie_diff'
]

BASE_FEATURES_LIST = [
    'score_diff', 'seconds_remaining', 'period', 'home_efg', 'away_efg', 
    'turnover_diff', 'home_rebound_rate', 'required_catchup_rate',
    'live_pace', 'score_momentum',
    'momentum_2min', 'momentum_4min', 'momentum_6min', 'momentum_8min', 'momentum_10min',
    'home_3p_reliance', 'away_3p_reliance',
    'home_steal_rate', 'away_steal_rate', 'home_block_rate', 'away_block_rate'
] + VOLATILITY_FEATURES_LIST


def add_interaction_features(df_or_dict):
    """
    Add interaction features for variance prediction.
    
    Works with both DataFrames (for training) and dicts (for inference).
    
    Args:
        df_or_dict: Either a pandas DataFrame or a dict of features
        
    Returns:
        Same type as input with interaction features added
    """
    if isinstance(df_or_dict, dict):
        # Dict case (inference)
        result = dict(df_or_dict)
        score_diff = result.get('score_diff', 0.0)
        seconds = result.get('seconds_remaining', 0.0)
        
        result['time_x_margin'] = seconds * score_diff
        result['time_x_abs_margin'] = seconds * abs(score_diff)
        result['margin_squared'] = score_diff ** 2
        result['log_time'] = np.log1p(seconds)
        result['time_proportion'] = seconds / 2880.0
        result['volatility_x_time'] = result.get('score_volatility', 0.0) * seconds
        
        # PRIOR INTERACTIONS: Allow pre-game stats to decay as time runs out
        result['home_season_margin_x_time'] = result.get('home_team_season_win_margin', 0.0) * result['time_proportion']
        result['away_season_margin_x_time'] = result.get('away_team_season_win_margin', 0.0) * result['time_proportion']
        result['home_recent_margin_x_time'] = result.get('home_team_recent_win_margin', 0.0) * result['time_proportion']
        result['away_recent_margin_x_time'] = result.get('away_team_recent_win_margin', 0.0) * result['time_proportion']
        result['home_pie_x_time'] = result.get('home_roster_recent_pie', 0.1) * result['time_proportion']
        result['away_pie_x_time'] = result.get('away_roster_recent_pie', 0.1) * result['time_proportion']
        
        # NEW: Star Difference
        result['net_star_pie_diff'] = result.get('home_roster_recent_top3_pie', 0) - result.get('away_roster_recent_top3_pie', 0)
        
        return result
    else:
        # DataFrame case (training)
        df = df_or_dict.copy()
        
        # Ensure base columns exist for math (using .get() or fillna)
        score_diff = df['score_diff']
        seconds = df['seconds_remaining']
        
        df['time_x_margin'] = seconds * score_diff
        df['time_x_abs_margin'] = seconds * np.abs(score_diff)
        df['margin_squared'] = score_diff ** 2
        df['log_time'] = np.log1p(seconds)
        df['time_proportion'] = seconds / 2880.0
        
        volatility = df['score_volatility'] if 'score_volatility' in df.columns else 0.0
        df['volatility_x_time'] = volatility * seconds
        
        # PRIOR INTERACTIONS: Allow pre-game stats to decay as time runs out
        df['home_season_margin_x_time'] = df.get('home_team_season_win_margin', 0.0) * df['time_proportion']
        df['away_season_margin_x_time'] = df.get('away_team_season_win_margin', 0.0) * df['time_proportion']
        df['home_recent_margin_x_time'] = df.get('home_team_recent_win_margin', 0.0) * df['time_proportion']
        df['away_recent_margin_x_time'] = df.get('away_team_recent_win_margin', 0.0) * df['time_proportion']
        df['home_pie_x_time'] = df.get('home_roster_recent_pie', 0.1) * df['time_proportion']
        df['away_pie_x_time'] = df.get('away_roster_recent_pie', 0.1) * df['time_proportion']
        
        # NEW: Star Difference
        df['net_star_pie_diff'] = df.get('home_roster_recent_top3_pie', 0) - df.get('away_roster_recent_top3_pie', 0)
        
        return df
