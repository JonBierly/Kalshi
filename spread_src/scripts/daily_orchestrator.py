#!/usr/bin/env python
"""
Daily Orchestrator for NBA Kalshi Trading System.

Lightweight trader launcher — meant to be started by cron after backfills.
Fetches today's game schedule, waits for game time, runs the trader,
stops after games end, then runs post-game settlements.

Cron setup (on the server):
     0  9 * * *   cd /opt/kalshi && bash run_backfills.sh >> /var/log/kalshi-backfills.log 2>&1
    30  9 * * *   cd /opt/kalshi && docker compose run --rm orchestrator python -m spread_src.scripts.daily_orchestrator --live >> /var/log/kalshi-trader.log 2>&1

Usage:
    python -m spread_src.scripts.daily_orchestrator              # dry-run (no real trades)
    python -m spread_src.scripts.daily_orchestrator --live        # real trades
    python -m spread_src.scripts.daily_orchestrator --skip-wait   # skip sleep, start trader immediately
"""

import os
import sys
import time
import subprocess
import signal
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

ET = ZoneInfo("America/New_York")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Apply CDN→S3 redirect EARLY so any nba_api live calls in this process work on VPS
try:
    from spread_src.patches import apply_patch
    apply_patch()
except Exception:
    pass  # Non-fatal; orchestrator has Kalshi fallback for schedule


def log(msg: str):
    """Print with timestamp."""
    now = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    print(f"[{now}] {msg}", flush=True)


def _patch_nba_api_headers():
    """
    stats.nba.com blocks bare-Python requests from cloud/VPS IPs.
    Patching nba_api's session to look like a real browser fixes the timeout.
    """
    try:
        from nba_api.stats.library.http import NBAStatsHTTP
        NBAStatsHTTP.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Referer': 'https://www.nba.com/',
            'Origin':  'https://www.nba.com',
            'Accept-Language': 'en-US,en;q=0.9',
        })
    except Exception as e:
        log(f"⚠️  Could not patch NBA API headers: {e}")


def get_todays_games() -> list[dict]:
    """
    Fetch today's NBA game schedule using ScoreboardV3 (stats endpoint).

    NOTE: We deliberately use nba_api.stats.endpoints.scoreboardv3 here,
    NOT nba_api.live.nba.endpoints.scoreboard. The live endpoint is known
    to return stale data (yesterday's schedule), making it unusable for
    daily scheduling.

    Returns list of dicts: game_id, home_team, away_team, start_time (ET), status
        status: 1=Scheduled, 2=Live, 3=Final
    """
    from nba_api.stats.endpoints import scoreboardv3

    now_et = datetime.now(ET)
    date_str = now_et.strftime('%Y-%m-%d')
    _patch_nba_api_headers()
    log(f"  Fetching schedule for {date_str} via ScoreboardV3...")

    try:
        board = scoreboardv3.ScoreboardV3(game_date=date_str, timeout=60)
        game_header = board.game_header.get_dict()
        line_score  = board.line_score.get_dict()
        gh_headers = game_header['headers']
        gh_data    = game_header['data']
        ls_headers = line_score['headers']
        ls_data    = line_score['data']
    except Exception as e:
        log(f"⚠️  Error fetching scoreboard: {e}")
        log("  Falling back to Kalshi events API for schedule...")
        return _get_games_from_kalshi()

    # Build header -> column index maps
    gh_idx = {h: i for i, h in enumerate(gh_headers)}
    ls_idx = {h: i for i, h in enumerate(ls_headers)}

    # Build game_id -> (home_tri, away_tri) from LineScore
    # LineScore has 2 rows per game: first = away, second = home (by convention)
    game_teams = {}
    for row in ls_data:
        gid = row[ls_idx['gameId']]
        tri = row[ls_idx['teamTricode']]
        if gid not in game_teams:
            game_teams[gid] = {'away': tri}  # first team row = away
        else:
            game_teams[gid]['home'] = tri     # second team row = home

    result = []
    for row in gh_data:
        game_id  = row[gh_idx['gameId']]
        status   = row[gh_idx['gameStatus']]       # 1=Scheduled, 2=Live, 3=Final
        status_t = row[gh_idx['gameStatusText']]    # e.g. "7:00 pm ET" or "Final"

        start_et = None
        if status == 1:
            # Parse "7:00 pm ET" → datetime
            try:
                t = datetime.strptime(status_t.replace(" ET", "").strip(), "%I:%M %p").time()
                start_et = now_et.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            except Exception:
                log(f"⚠️  Couldn't parse game time: '{status_t}'")
        elif status == 2:
            # Already live — use now so the window extends 4 hours from this moment
            start_et = now_et
        # status == 3 (Final): leave start_et as None, will be excluded from window

        teams_info = game_teams.get(game_id, {})
        home_tri = teams_info.get('home', '???')
        away_tri = teams_info.get('away', '???')

        result.append({
            "game_id":    game_id,
            "home_team":  home_tri,
            "away_team":  away_tri,
            "start_time": start_et,
            "status":     status,
        })

    return result


def _get_games_from_kalshi() -> list[dict]:
    """
    Fallback schedule discovery when stats.nba.com is unreachable (e.g. VPS IP block).
    Queries Kalshi directly for open KXNBASPREAD events today.
    Returns games with start_time=now so the orchestrator proceeds immediately.
    """
    import os
    from data.kalshi import KalshiClient
    key_id   = os.environ.get('KALSHI_KEY_ID', '')
    key_path = os.environ.get('KALSHI_KEY_PATH', 'key.key')
    if not key_id:
        log("  ⚠️ KALSHI_KEY_ID not set, cannot use Kalshi fallback")
        return []

    try:
        kalshi = KalshiClient(key_id, key_path)
        events = kalshi.get_todays_spread_events()
        if not events:
            return []

        now_et = datetime.now(ET)
        # Set stop to 2 AM: use "latest game at 10 PM" so stop = 10 PM + 4 hr = 2 AM
        latest_start = now_et.replace(hour=22, minute=0, second=0, microsecond=0)
        result = []
        for i, ev in enumerate(events):
            # All games use now as start (no sleep), except last game uses 10 PM for window calc
            start_time = latest_start if i == len(events) - 1 else now_et
            result.append({
                "game_id":    ev['event_ticker'],
                "home_team":  ev['home_tri'],
                "away_team":  ev['away_tri'],
                "start_time": start_time,
                "status":     2,
            })
            log(f"  ✓ (via Kalshi) {ev['away_tri']} @ {ev['home_tri']}")
        return result
    except Exception as e:
        log(f"  ⚠️ Kalshi fallback failed: {e}")
        return []




def run_script(module_path: str, args: list[str] = None, label: str = None) -> bool:
    """Run a Python module as a subprocess. Returns True if successful."""
    cmd = [sys.executable, "-m", module_path] + (args or [])
    display = label or module_path
    log(f"  ▶ Running {display}...")
    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1800)
        if r.returncode == 0:
            log(f"  ✅ {display} completed")
            return True
        else:
            log(f"  ❌ {display} failed (exit {r.returncode})")
            for line in (r.stderr or "").strip().split('\n')[-5:]:
                log(f"     {line}")
            return False
    except subprocess.TimeoutExpired:
        log(f"  ⏰ {display} timed out")
        return False
    except Exception as e:
        log(f"  ❌ {display} error: {e}")
        return False


def run_trader(live: bool = False) -> subprocess.Popen:
    """Start the live trader as a background subprocess."""
    cmd = [
        sys.executable, "-m", "spread_src.scripts.simple_live_trader",
        "--interval", "15", "--min-edge", "0.08", "--min-spread", "4",
    ]
    if live:
        cmd.append("--live")
    mode = "LIVE" if live else "DRY-RUN"
    log(f"🚀 Starting trader ({mode} mode)")
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=sys.stdout, stderr=sys.stderr)
    log(f"  Trader PID: {proc.pid}")
    return proc


def stop_trader(proc: subprocess.Popen):
    """Gracefully stop the trader via SIGINT (same as Ctrl+C)."""
    if proc is None or proc.poll() is not None:
        log("  Trader already stopped")
        return
    log("🛑 Stopping trader...")
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
        log("  ✅ Trader stopped gracefully")
    except subprocess.TimeoutExpired:
        log("  ⚠️  Force killing trader...")
        proc.kill()
        proc.wait()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NBA Kalshi Daily Orchestrator")
    parser.add_argument("--live",              action="store_true", help="Real money (default: dry-run)")
    parser.add_argument("--skip-wait",         action="store_true", help="Start trader immediately, skip sleep")
    parser.add_argument("--game-buffer-hours", type=float, default=4.0,  help="Hours past last game start to keep trading (default: 4)")
    parser.add_argument("--pre-game-minutes",  type=float, default=10.0, help="Minutes before first game to start trader (default: 10)")
    args = parser.parse_args()

    log("=" * 60)
    log("🏀 NBA KALSHI DAILY ORCHESTRATOR")
    log("=" * 60)

    # ── 1. Fetch today's schedule ─────────────────────────────────
    log("📅 Fetching today's NBA schedule...")
    games = get_todays_games()

    if not games:
        log("  No games found for today. Exiting.")
        return

    log(f"  {len(games)} game(s):")
    status_label = {1: "Scheduled", 2: "🔴 LIVE", 3: "✅ Final"}
    for g in games:
        t = g['start_time'].strftime('%I:%M %p ET') if g['start_time'] else '???'
        log(f"    {g['away_team']} @ {g['home_team']} — {t} ({status_label.get(g['status'], '?')})")

    # ── 2. Calculate trader window ────────────────────────────────
    active = [g for g in games if g['status'] != 3 and g['start_time']]

    if not active:
        log("  All games already Final. Running post-game settlements only.")
        run_script("spread_src.scripts.settle_historical_trades", [], "Trade Settlements")
        run_script("spread_src.scripts.backfill_outcomes",        [], "Prediction Outcomes")
        return

    earliest = min(g['start_time'] for g in active)
    latest   = max(g['start_time'] for g in active)

    trader_start = earliest - timedelta(minutes=args.pre_game_minutes)
    trader_stop  = latest   + timedelta(hours=args.game_buffer_hours)
    now = datetime.now(ET)

    log(f"\n  ⏰ Trader window:")
    log(f"     Start: {trader_start.strftime('%I:%M %p ET')}  (first tip-off minus {int(args.pre_game_minutes)} min)")
    log(f"     Stop:  {trader_stop.strftime('%I:%M %p ET')}  (last tip-off plus {args.game_buffer_hours:.0f} hr)")

    if now > trader_stop:
        log("  Window already passed. Running post-game settlements only.")
        run_script("spread_src.scripts.settle_historical_trades", [], "Trade Settlements")
        run_script("spread_src.scripts.backfill_outcomes",        [], "Prediction Outcomes")
        return

    # ── 3. Sleep until game time ──────────────────────────────────
    if not args.skip_wait and now < trader_start:
        wait_s = (trader_start - now).total_seconds()
        log(f"\n  💤 Sleeping {wait_s/3600:.1f} hr until {trader_start.strftime('%I:%M %p ET')}...")
        time.sleep(wait_s)

    # ── 4. Run the trader ─────────────────────────────────────────
    log("\n🏀 TRADING WINDOW OPEN")
    proc = run_trader(live=args.live)

    now = datetime.now(ET)
    remaining = (trader_stop - now).total_seconds()
    if remaining > 0:
        log(f"  Trading for {remaining/3600:.1f} hr (until {trader_stop.strftime('%I:%M %p ET')})")
        try:
            proc.wait(timeout=remaining)
            log("  ⚠️  Trader exited before window closed")
        except subprocess.TimeoutExpired:
            log("  ⏰ Trading window closed")
            stop_trader(proc)
    else:
        stop_trader(proc)

    # ── 5. Post-game settlements ──────────────────────────────────
    log("\n📊 Post-game settlements...")
    run_script("spread_src.scripts.settle_historical_trades", [], "Trade Settlements")
    run_script("spread_src.scripts.backfill_outcomes",        [], "Prediction Outcomes")

    log("\n✅ Daily cycle complete!")


if __name__ == "__main__":
    main()
