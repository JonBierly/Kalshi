"""
S3-based NBA data client.

Fetches game data from the unblocked NBA S3 bucket instead of stats.nba.com.
This bypasses the Akamai WAF that blocks datacenter IPs from accessing
stats.nba.com endpoints.

S3 Base URL:
    https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/NBA/liveData/

Available endpoints:
    /scoreboard/todaysScoreboard_00.json
    /boxscore/boxscore_{gameId}.json
    /playbyplay/playbyplay_{gameId}.json
"""

import time
import requests
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime


S3_BASE = "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/NBA/liveData"

# NBA team tricode -> team ID mapping (same as nba_api.stats.static.teams)
_TEAM_IDS = {
    'ATL': 1610612737, 'BOS': 1610612738, 'BKN': 1610612751, 'CHA': 1610612766,
    'CHI': 1610612741, 'CLE': 1610612739, 'DAL': 1610612742, 'DEN': 1610612743,
    'DET': 1610612765, 'GSW': 1610612744, 'HOU': 1610612745, 'IND': 1610612754,
    'LAC': 1610612746, 'LAL': 1610612747, 'MEM': 1610612763, 'MIA': 1610612748,
    'MIL': 1610612749, 'MIN': 1610612750, 'NOP': 1610612740, 'NYK': 1610612752,
    'OKC': 1610612760, 'ORL': 1610612753, 'PHI': 1610612755, 'PHX': 1610612756,
    'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'TOR': 1610612761,
    'UTA': 1610612762, 'WAS': 1610612764,
}


class S3DataClient:
    """
    NBA data client that fetches from the unblocked S3 bucket.
    Drop-in replacement for HistoricalDataClient on VPS.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})

    # ── Game Discovery ──────────────────────────────────────────────

    def get_season_games(self, season: str = '2025-26') -> List[str]:
        """
        Discover game IDs by scanning sequential IDs via S3 box score endpoint.

        NBA game IDs:
            Regular season: 002250XXXX (for 2025-26)
            Playoffs:       004250XXXX

        We scan forward from the last known game until we hit consecutive 403s.
        """
        # Determine prefix from season
        # NBA game IDs are 10 digits: 002 + 2-digit year + 0 + 4-digit game number
        # Example: 0022500928 = 002 + 25 + 0 + 0928 (2025-26 regular season game 928)
        start_year = season.split('-')[0][-2:]  # '25'
        prefix = f"002{start_year}0"  # Regular season prefix (6 digits)

        print(f"Discovering {season} games from S3 (prefix {prefix})...")

        game_ids = []
        consecutive_misses = 0
        max_misses = 20  # Stop after 20 consecutive misses

        for game_num in range(1, 1400):  # Max ~1230 regular season games
            game_id = f"{prefix}{game_num:04d}"
            try:
                url = f"{S3_BASE}/boxscore/boxscore_{game_id}.json"
                r = self.session.get(url, timeout=10)
                if r.status_code == 200:
                    # Verify it's a real finished or live game
                    data = r.json()
                    status = data.get('game', {}).get('gameStatus', 0)
                    if status >= 2:  # Live (2) or Final (3)
                        game_ids.append(game_id)
                        consecutive_misses = 0
                    else:
                        # Scheduled but not started — stop scanning
                        consecutive_misses += 1
                else:
                    consecutive_misses += 1
            except Exception:
                consecutive_misses += 1

            if consecutive_misses >= max_misses:
                break

            # Light rate limiting
            if game_num % 50 == 0:
                print(f"  Scanned {game_num} IDs, found {len(game_ids)} games so far...")
                time.sleep(0.5)

        print(f"  Found {len(game_ids)} games for {season}")
        return game_ids

    def get_todays_game_ids(self, date_str: Optional[str] = None) -> List[str]:
        """
        Robust discovery of today's Game IDs by scanning forward from the latest 
        known game in the database. 
        
        This bypassing the need for a 'current' scoreboard file which 
        often remains stale for 24+ hours on S3.
        """
        if date_str is None:
            # Use Eastern time for NBA date
            from zoneinfo import ZoneInfo
            date_str = datetime.now(ZoneInfo("America/New_York")).strftime('%Y-%m-%d')
        
        print(f"Scanning S3 for games on {date_str}...")
        
        # 1. Determine starting point
        # Since we don't have easy DB access here, we'll probe backwards from a high 
        # number or use a reasonable guess for the current season (2025-26).
        # Yesterday we saw 0022500961. Today we saw 0022500975.
        # We'll probe around the last known active range.
        prefix = "002250"
        
        # Let's hit the scoreboard first just to get a baseline ID if available
        start_id = 900 # Safe baseline for mid-March
        try:
            r = self.session.get(f"{S3_BASE}/scoreboard/todaysScoreboard_00.json", timeout=5)
            if r.status_code == 200:
                data = r.json()
                games = data.get('scoreboard', {}).get('games', [])
                if games:
                    # Extract numeric part of gameId
                    gid = games[0]['gameId']
                    start_id = int(gid[-4:]) - 30 # Backtrack 30 games to be safe
        except:
            pass

        game_ids = []
        consecutive_misses = 0
        max_misses = 30
        
        for i in range(start_id, start_id + 100):
            game_id = f"{prefix}{i:04d}"
            try:
                url = f"{S3_BASE}/boxscore/boxscore_{game_id}.json"
                r = self.session.get(url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    game_time_utc = data.get('game', {}).get('gameTimeUTC', '')
                    # Convert UTC (e.g., 2026-03-15T02:30:00Z) to ET YYYY-MM-DD
                    if game_time_utc:
                        from zoneinfo import ZoneInfo
                        dt_utc = datetime.fromisoformat(game_time_utc.replace('Z', '+00:00'))
                        dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
                        game_date = dt_et.strftime('%Y-%m-%d')
                        
                        if game_date == date_str:
                            game_ids.append(game_id)
                    consecutive_misses = 0
                elif r.status_code == 403:
                    consecutive_misses += 1
                else:
                    consecutive_misses += 1
            except:
                consecutive_misses += 1
            
            if consecutive_misses >= max_misses:
                break
                
        print(f"  Smart Scan found {len(game_ids)} games for {date_str}")
        return game_ids

    # ── Play-by-Play ────────────────────────────────────────────────

    def get_game_pbp(self, game_id: str) -> pd.DataFrame:
        """
        Fetch play-by-play from S3 and convert to project schema.
        Returns DataFrame matching DataSchema.COLUMNS.
        """
        from data.acquisition import DataSchema

        url = f"{S3_BASE}/playbyplay/playbyplay_{game_id}.json"
        try:
            r = self.session.get(url, timeout=15)
            if r.status_code != 200:
                print(f"  PBP not available for {game_id} (HTTP {r.status_code})")
                return pd.DataFrame(columns=DataSchema.COLUMNS)

            data = r.json()
            actions = data.get('game', {}).get('actions', [])
            if not actions:
                return pd.DataFrame(columns=DataSchema.COLUMNS)

            df = pd.DataFrame(actions)

            # Parse clock: "PT05M30.00S" -> seconds
            def parse_clock(clock_str):
                if not isinstance(clock_str, str):
                    return 0
                clock_str = clock_str.replace('PT', '').replace('M', ':').replace('S', '')
                try:
                    if ':' in clock_str:
                        m, s = clock_str.split(':')
                        return int(m) * 60 + float(s)
                    return float(clock_str)
                except (ValueError, TypeError):
                    return 0

            # Build processed DataFrame
            processed = pd.DataFrame()
            processed['game_id'] = game_id
            processed['timestamp'] = pd.to_datetime('now')
            processed['period'] = df['period']
            processed['remaining_time'] = df['clock'].apply(parse_clock)

            # Forward-fill scores (non-scoring events have empty strings)
            s_home = df['scoreHome'].replace('', np.nan).ffill().fillna(0)
            s_away = df['scoreAway'].replace('', np.nan).ffill().fillna(0)
            processed['home_score'] = pd.to_numeric(s_home, errors='coerce').fillna(0).astype(int)
            processed['away_score'] = pd.to_numeric(s_away, errors='coerce').fillna(0).astype(int)
            processed['score_diff'] = processed['home_score'] - processed['away_score']

            processed['event_type'] = df.get('actionType', '')
            processed['player_id'] = df.get('personId', 0)
            processed['description'] = df.get('description', '')
            processed['home_team_id'] = 0
            processed['away_team_id'] = 0
            processed['player_team_id'] = df.get('teamId', 0)
            processed['event_num'] = df.get('actionNumber', range(len(df)))

            # Set home/away team IDs from the game data (PBP usually lacks this, so fallback to boxscore)
            game_info = data.get('game', {})
            home_id = game_info.get('homeTeam', {}).get('teamId', 0)
            away_id = game_info.get('awayTeam', {}).get('teamId', 0)
            
            if home_id == 0 or away_id == 0:
                info = self.get_game_info(game_id)
                if info:
                    home_id = info.get('home_team_id', 0)
                    away_id = info.get('away_team_id', 0)
                    
            processed['home_team_id'] = home_id
            processed['away_team_id'] = away_id

            cols = DataSchema.COLUMNS + ['event_num']
            if 'player_team_id' in processed.columns:
                cols.append('player_team_id')

            return processed[cols]

        except Exception as e:
            print(f"  PBP error for {game_id}: {e}")
            return pd.DataFrame(columns=DataSchema.COLUMNS)

    # ── Box Score (Advanced Player Stats) ───────────────────────────

    def get_advanced_boxscore(self, game_id: str) -> pd.DataFrame:
        """
        Fetch box score from S3 and map player stats to the format
        expected by DatabaseManager.save_advanced_stats().

        Note: S3 box score has basic stats but NOT the advanced metrics
        (offensiveRating, etc.) that come from stats.nba.com's
        BoxScoreAdvancedV3. We compute what we can and leave the rest as 0.
        """
        url = f"{S3_BASE}/boxscore/boxscore_{game_id}.json"
        try:
            r = self.session.get(url, timeout=15)
            if r.status_code != 200:
                print(f"  Box score not available for {game_id}")
                return pd.DataFrame()

            data = r.json()
            game = data.get('game', {})
            rows = []

            for side in ('homeTeam', 'awayTeam'):
                team = game.get(side, {})
                team_id = team.get('teamId', 0)
                team_tri = team.get('teamTricode', '')
                team_city = team.get('teamCity', '')
                team_name = team.get('teamName', '')
                team_slug = team.get('teamSlug', '')
                team_stats = team.get('statistics', {})

                # Compute team-level possessions for rating calculations
                fga = team_stats.get('fieldGoalsAttempted', 0)
                fta = team_stats.get('freeThrowsAttempted', 0)
                oreb = team_stats.get('reboundsOffensive', 0)
                tov = team_stats.get('turnoversTotal', 0)
                team_poss = max(1, 0.96 * (fga + 0.44 * fta - oreb + tov))

                for player in team.get('players', []):
                    stats = player.get('statistics', {})
                    pid = player.get('personId', 0)
                    if pid == 0:
                        continue

                    # Parse minutes: "PT29M50.40S" -> "29:50"
                    minutes_raw = stats.get('minutes', 'PT00M00.00S')
                    minutes_str = self._parse_minutes_to_str(minutes_raw)
                    minutes_float = self._parse_minutes_to_float(minutes_raw)

                    # Compute basic advanced stats where possible
                    p_fga = stats.get('fieldGoalsAttempted', 0)
                    p_fta = stats.get('freeThrowsAttempted', 0)
                    p_fgm = stats.get('fieldGoalsMade', 0)
                    p_fg3m = stats.get('threePointersMade', 0)
                    p_pts = stats.get('points', 0)
                    p_ast = stats.get('assists', 0)
                    p_tov = stats.get('turnovers', 0)
                    p_oreb = stats.get('reboundsOffensive', 0)
                    p_dreb = stats.get('reboundsDefensive', 0)
                    p_reb = stats.get('reboundsTotal', 0)

                    # True shooting
                    tsa = p_fga + 0.44 * p_fta
                    ts_pct = (p_pts / (2 * tsa)) if tsa > 0 else 0.0

                    # Effective FG%
                    efg_pct = ((p_fgm + 0.5 * p_fg3m) / p_fga) if p_fga > 0 else 0.0

                    # Assist-to-turnover
                    ast_tov = (p_ast / p_tov) if p_tov > 0 else 0.0

                    # Usage rate (simplified)
                    usg = ((p_fga + 0.44 * p_fta + p_tov) / team_poss) if team_poss > 0 else 0.0

                    row = {
                        'personId': pid,
                        'teamId': team_id,
                        'teamCity': team_city,
                        'teamName': team_name,
                        'teamSlug': team_slug,
                        'teamTricode': team_tri,
                        'firstName': player.get('firstName', ''),
                        'familyName': player.get('familyName', ''),
                        'nameI': player.get('nameI', ''),
                        'playerSlug': '',
                        'position': player.get('position', ''),
                        'jerseyNum': player.get('jerseyNum', ''),
                        'comment': player.get('comment', ''),
                        'minutes': minutes_str,

                        # Computed advanced stats
                        'offensiveRating': 0.0,       # Can't compute without lineup data
                        'estimatedOffensiveRating': 0.0,
                        'defensiveRating': 0.0,
                        'estimatedDefensiveRating': 0.0,
                        'netRating': 0.0,
                        'estimatedNetRating': 0.0,

                        'assistPercentage': 0.0,
                        'assistToTurnover': ast_tov,
                        'assistRatio': 0.0,

                        'offensiveReboundPercentage': 0.0,
                        'defensiveReboundPercentage': 0.0,
                        'reboundPercentage': 0.0,

                        'turnoverRatio': (p_tov / team_poss * 100) if team_poss > 0 else 0.0,
                        'effectiveFieldGoalPercentage': efg_pct,
                        'trueShootingPercentage': ts_pct,
                        'usagePercentage': usg * 100,
                        'estimatedUsagePercentage': 0.0,

                        'pace': 0.0,
                        'estimatedPace': 0.0,
                        'pacePer40': 0.0,
                        'possessions': 0,
                        'PIE': 0.0,
                    }
                    rows.append(row)

            return pd.DataFrame(rows) if rows else pd.DataFrame()

        except Exception as e:
            print(f"  Box score error for {game_id}: {e}")
            return pd.DataFrame()

    # ── Team Game Log ───────────────────────────────────────────────

    def get_team_game_log(self, team_id: int, season: str = '2025-26',
                          season_type: str = 'Regular Season') -> pd.DataFrame:
        """
        Build a team game log from S3 box scores.

        Since S3 doesn't have a team game log endpoint, we scan all games
        and filter for this team. Returns DataFrame with columns matching
        what DatabaseManager.save_team_basic_stats() expects.
        """
        # This is expensive (scans all games). Callers should use
        # get_all_team_logs() instead which does a single scan.
        print(f"  Building game log for team {team_id} from S3...")
        return self._build_team_log_for_team(team_id, season)

    def get_all_team_logs(self, season: str = '2025-26',
                          existing_game_ids: set = None) -> pd.DataFrame:
        """
        Efficiently build team game logs for ALL teams in a single scan.
        
        Args:
            season: Season string (e.g., '2025-26')
            existing_game_ids: Set of game IDs already in DB (skip these)
        
        Returns DataFrame with columns matching save_team_basic_stats() format.
        """
        start_year = season.split('-')[0][-2:]
        prefix = f"002{start_year}0"  # Regular season prefix (6 digits)

        if existing_game_ids is None:
            existing_game_ids = set()

        print(f"Scanning S3 for all {season} team stats (prefix {prefix})...")

        all_rows = []
        consecutive_misses = 0
        games_found = 0

        for game_num in range(1, 1400):
            game_id = f"{prefix}{game_num:04d}"

            # Skip games already in DB
            if game_id in existing_game_ids:
                consecutive_misses = 0
                continue

            try:
                url = f"{S3_BASE}/boxscore/boxscore_{game_id}.json"
                r = self.session.get(url, timeout=10)

                if r.status_code != 200:
                    consecutive_misses += 1
                    if consecutive_misses >= 20:
                        break
                    continue

                data = r.json()
                game = data.get('game', {})
                status = game.get('gameStatus', 0)

                if status < 3:  # Not yet final
                    consecutive_misses += 1
                    if consecutive_misses >= 20:
                        break
                    continue

                consecutive_misses = 0
                games_found += 1

                # Extract game date
                game_date_utc = game.get('gameTimeUTC', '')
                try:
                    dt = datetime.fromisoformat(game_date_utc.replace('Z', '+00:00'))
                    game_date_str = dt.strftime('%b %d, %Y')
                except (ValueError, AttributeError):
                    game_date_str = ''

                home = game.get('homeTeam', {})
                away = game.get('awayTeam', {})
                home_tri = home.get('teamTricode', '')
                away_tri = away.get('teamTricode', '')
                home_id = home.get('teamId', 0)
                away_id = away.get('teamId', 0)

                for team_data, is_home in [(home, True), (away, False)]:
                    t_id = team_data.get('teamId', 0)
                    t_tri = team_data.get('teamTricode', '')
                    opp_tri = away_tri if is_home else home_tri
                    stats = team_data.get('statistics', {})
                    t_score = team_data.get('score', 0)
                    opp_score = away.get('score', 0) if is_home else home.get('score', 0)

                    matchup = f"{t_tri} vs. {opp_tri}" if is_home else f"{t_tri} @ {opp_tri}"
                    wl = 'W' if t_score > opp_score else 'L'

                    # Parse team minutes
                    minutes_raw = stats.get('minutes', 'PT240M00.00S')
                    team_min = self._parse_minutes_to_float(minutes_raw)

                    row = {
                        'Game_ID': game_id,
                        'Team_ID': t_id,
                        'MATCHUP': matchup,
                        'GAME_DATE': game_date_str,
                        'WL': wl,
                        'MIN': team_min,
                        'FGM': stats.get('fieldGoalsMade', 0),
                        'FGA': stats.get('fieldGoalsAttempted', 0),
                        'FG_PCT': stats.get('fieldGoalsPercentage', 0.0),
                        'FG3M': stats.get('threePointersMade', 0),
                        'FG3A': stats.get('threePointersAttempted', 0),
                        'FG3_PCT': stats.get('threePointersPercentage', 0.0),
                        'FTM': stats.get('freeThrowsMade', 0),
                        'FTA': stats.get('freeThrowsAttempted', 0),
                        'FT_PCT': stats.get('freeThrowsPercentage', 0.0),
                        'OREB': stats.get('reboundsOffensive', 0),
                        'DREB': stats.get('reboundsDefensive', 0),
                        'REB': stats.get('reboundsTotal', 0),
                        'AST': stats.get('assists', 0),
                        'STL': stats.get('steals', 0),
                        'BLK': stats.get('blocks', 0),
                        'TOV': stats.get('turnoversTotal', 0),
                        'PF': stats.get('foulsPersonal', 0),
                        'PTS': stats.get('points', 0),
                    }
                    all_rows.append(row)

            except Exception as e:
                print(f"  Error on {game_id}: {e}")
                consecutive_misses += 1
                if consecutive_misses >= 20:
                    break
                continue

            if game_num % 100 == 0:
                print(f"  Scanned {game_num} IDs, found {games_found} new games...")
                time.sleep(0.5)

        print(f"  Found {games_found} games, {len(all_rows)} team-game rows")
        return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    # ── Helpers ─────────────────────────────────────────────────────

    def _build_team_log_for_team(self, team_id: int, season: str) -> pd.DataFrame:
        """Build game log for a single team from S3 (wrapper for get_all_team_logs)."""
        all_logs = self.get_all_team_logs(season)
        if all_logs.empty:
            return pd.DataFrame()
        return all_logs[all_logs['Team_ID'] == team_id]

    @staticmethod
    def _parse_minutes_to_str(minutes_raw: str) -> str:
        """Parse 'PT29M50.40S' -> '29:50'."""
        if not isinstance(minutes_raw, str) or not minutes_raw.startswith('PT'):
            return '0:00'
        try:
            cleaned = minutes_raw.replace('PT', '').replace('S', '')
            if 'M' in cleaned:
                m, s = cleaned.split('M')
                return f"{int(m)}:{int(float(s)):02d}"
            else:
                return f"0:{int(float(cleaned)):02d}"
        except (ValueError, TypeError):
            return '0:00'

    @staticmethod
    def _parse_minutes_to_float(minutes_raw: str) -> float:
        """Parse 'PT29M50.40S' -> 29.84 (minutes as float)."""
        if not isinstance(minutes_raw, str) or not minutes_raw.startswith('PT'):
            return 0.0
        try:
            cleaned = minutes_raw.replace('PT', '').replace('S', '')
            if 'M' in cleaned:
                m, s = cleaned.split('M')
                return int(m) + float(s) / 60
            else:
                return float(cleaned) / 60
        except (ValueError, TypeError):
            return 0.0

    def get_game_info(self, game_id: str) -> Optional[dict]:
        """Fetch basic game info (teams, score, status) from S3 box score."""
        try:
            url = f"{S3_BASE}/boxscore/boxscore_{game_id}.json"
            r = self.session.get(url, timeout=10)
            if r.status_code != 200:
                return None
            data = r.json()
            game = data.get('game', {})
            return {
                'game_id': game_id,
                'status': game.get('gameStatus', 0),
                'home_team_id': game.get('homeTeam', {}).get('teamId', 0),
                'away_team_id': game.get('awayTeam', {}).get('teamId', 0),
                'home_score': game.get('homeTeam', {}).get('score', 0),
                'away_score': game.get('awayTeam', {}).get('score', 0),
                'home_tri': game.get('homeTeam', {}).get('teamTricode', ''),
                'away_tri': game.get('awayTeam', {}).get('teamTricode', ''),
                'game_date_utc': game.get('gameTimeUTC', ''),
            }
        except Exception:
            return None
