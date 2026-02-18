#!/usr/bin/env python
"""
Volatility Analysis: Model Std vs Market-Implied Std

Reconstructs model predictions from PBP data and compares the model's
predicted volatility to the market's implied volatility (fitted from
bid/ask strip across Kalshi spread strike levels).

Usage:
    python -m spread_src.scripts.analyze_volatility
    python -m spread_src.scripts.analyze_volatility --date-pattern 26FEB11
    python -m spread_src.scripts.analyze_volatility --sample 20
"""

import argparse
import os
import sys
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spread_src.models.spread_model import SpreadDistributionModel
from spread_src.features.engineering import (
    FeatureEngine, TeamStatsEngine, RosterEngine,
    add_interaction_features
)
from spread_src.data.spread_markets import parse_spread_ticker

DB_PATH = 'data/nba_data.db'
MODEL_PATH = 'models/nba_spread_ngboost_v3_final.pkl'
REPORT_DIR = 'reports/volatility'


def parse_args():
    p = argparse.ArgumentParser(description='Volatility: model vs market implied std')
    p.add_argument('--date-pattern', default=None, help='Filter tickers, e.g. 26FEB11')
    p.add_argument('--sample', type=int, default=None, help='Random sample N games')
    p.add_argument('--snapshot-interval', type=int, default=1, help='Process every Nth snapshot (default 1)')
    p.add_argument('--game-ids', nargs='+', help='Specific game IDs')
    return p.parse_args()


# ── Market Distribution Fitting ──────────────────────────────────────────────

def _normal_survival(x, loc, scale):
    """P(X > x) for X ~ Normal(loc, scale)."""
    return 1 - norm.cdf(x, loc=loc, scale=scale)


def fit_market_distribution(thresholds, survival_probs):
    """Fit normal CDF to market-implied survival function.
    
    Returns (market_loc, market_std) or (None, None).
    """
    if len(thresholds) < 3:
        return None, None
    try:
        weights = np.clip(survival_probs, 0.05, 0.95)
        loc0 = float(np.average(thresholds, weights=weights))
        scale0 = max(float(np.ptp(thresholds)) / 3, 3.0)
        popt, _ = curve_fit(
            _normal_survival, thresholds, survival_probs,
            p0=[loc0, scale0],
            bounds=([-60, 0.5], [60, 60]),
            maxfev=3000
        )
        market_loc, market_std = float(popt[0]), float(popt[1])
        if 0.5 <= market_std <= 50:
            return market_loc, market_std
    except Exception:
        pass
    return None, None


def extract_market_vol(snapshot_tickers, home_tri, away_tri):
    """From tickers with bid/ask at one timestamp, fit market-implied distribution."""
    thresholds, surv_probs = [], []
    for t in snapshot_tickers:
        try:
            parsed = parse_spread_ticker(t['ticker'])
        except Exception:
            continue
        spread_team = parsed['spread_team']
        spread_val = parsed['spread_value']
        mid = np.clip((t['bid'] + t['ask']) / 200.0, 0.01, 0.99)

        if spread_team == home_tri:
            # P(home - away > spread_val) = mid
            thresholds.append(spread_val)
            surv_probs.append(mid)
        elif spread_team == away_tri:
            # P(away - home > spread_val) = mid  →  P(home - away > -spread_val) = 1 - mid
            thresholds.append(-spread_val)
            surv_probs.append(1 - mid)

    if len(thresholds) < 3:
        return None, None

    order = np.argsort(thresholds)
    return fit_market_distribution(np.array(thresholds)[order], np.array(surv_probs)[order])


# ── Data Loading ─────────────────────────────────────────────────────────────

def get_game_ids_with_data(date_pattern=None):
    """Find game IDs that have predictions + PBP + games table entries."""
    conn = sqlite3.connect(DB_PATH)
    if date_pattern:
        rows = conn.execute("""
            SELECT DISTINCT mp.game_id FROM model_predictions mp
            JOIN games g ON mp.game_id = g.game_id
            WHERE EXISTS (SELECT 1 FROM pbp_events pe WHERE pe.game_id = mp.game_id)
              AND mp.ticker LIKE ?
        """, (f'%{date_pattern}%',)).fetchall()
    else:
        rows = conn.execute("""
            SELECT DISTINCT mp.game_id FROM model_predictions mp
            JOIN games g ON mp.game_id = g.game_id
            WHERE EXISTS (SELECT 1 FROM pbp_events pe WHERE pe.game_id = mp.game_id)
        """).fetchall()
    conn.close()
    return [r[0] for r in rows]


def load_market_snapshots(game_id, interval=1):
    """Load predictions grouped by seconds_remaining."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT seconds_remaining, ticker, bid_price, ask_price, actual_outcome
        FROM model_predictions
        WHERE game_id = ? AND bid_price IS NOT NULL AND ask_price IS NOT NULL
          AND bid_price > 0 AND ask_price > 0
        ORDER BY seconds_remaining DESC
    """, (game_id,)).fetchall()
    conn.close()
    
    # Filter snapshots by distinct seconds_remaining
    distinct_secs = sorted(list(set(r[0] for r in rows)), reverse=True)
    target_secs = distinct_secs[::interval]
    target_set = set(target_secs)
    
    snapshots = defaultdict(list)
    for sec, ticker, bid, ask, outcome in rows:
        if sec in target_set:
            snapshots[sec].append({'ticker': ticker, 'bid': bid, 'ask': ask, 'outcome': outcome})
    return snapshots


def load_pbp(game_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM pbp_events WHERE game_id = ? ORDER BY period ASC, remaining_time DESC",
        conn, params=(game_id,))
    conn.close()
    return df


def get_game_teams(game_ids):
    conn = sqlite3.connect(DB_PATH)
    ph = ','.join('?' * len(game_ids))
    rows = conn.execute(
        f"SELECT game_id, home_team_id, away_team_id FROM games WHERE game_id IN ({ph})",
        game_ids).fetchall()
    conn.close()
    return {r[0]: (r[1], r[2]) for r in rows}


# ── Feature Reconstruction ───────────────────────────────────────────────────

def _total_seconds(period, remaining_time):
    if period <= 4:
        return remaining_time + (4 - period) * 720
    return remaining_time


def reconstruct_features_for_game(game_id, home_id, away_id, pbp_df,
                                   team_engine, roster_engine, target_timestamps):
    """Replay PBP events and sample features at target timestamps.
    
    Returns dict: seconds_remaining -> feature_dict
    """
    if pbp_df.empty:
        return {}

    engine = FeatureEngine()
    pbp = pbp_df.copy()
    pbp['home_team_id'] = home_id
    pbp['away_team_id'] = away_id
    pbp['total_seconds'] = pbp.apply(lambda r: _total_seconds(r['period'], r['remaining_time']), axis=1)
    pbp = pbp.sort_values('total_seconds', ascending=False).reset_index(drop=True)

    # Pre-game features (team + roster)
    pregame = {**team_engine.get_features(game_id, home_id, away_id),
               **roster_engine.get_features(game_id, home_id, away_id)}

    targets = sorted(target_timestamps, reverse=True)  # high to low (chronological)
    target_idx = 0
    results = {}

    for _, event in pbp.iterrows():
        engine.lightweight_update(event)
        event_secs = event['total_seconds']

        while target_idx < len(targets) and event_secs <= targets[target_idx]:
            sec = targets[target_idx]
            live = engine.calculate_current_features(
                event['score_diff'], sec, event['period'], game_id, home_id, away_id)
            full = {**live, **pregame}
            full = add_interaction_features(full)
            results[sec] = full
            target_idx += 1

    return results


# ── Main Pipeline ────────────────────────────────────────────────────────────

def build_comparison_df(game_ids, model, team_engine, roster_engine, interval=1):
    """Build DataFrame comparing model std to market-implied std."""
    game_teams = get_game_teams(game_ids)
    valid = [g for g in game_ids if g in game_teams]
    print(f"Processing {len(valid)} games (interval={interval})...")

    rows = []
    for i, gid in enumerate(game_ids):
        if gid not in game_teams: continue
        home_id, away_id = game_teams[gid]

        # Get tricodes from first ticker
        snapshots = load_market_snapshots(gid, interval=interval)
        if not snapshots:
            continue
        first_ticker = next(iter(snapshots.values()))[0]['ticker']
        try:
            parsed = parse_spread_ticker(first_ticker)
            home_tri, away_tri = parsed['home_team'], parsed['away_team']
        except Exception:
            continue

        pbp = load_pbp(gid)
        if pbp.empty:
            continue

        print(f"  [{i+1}/{len(valid)}] {gid} ({away_tri} @ {home_tri})...", end=' ', flush=True)

        target_times = list(snapshots.keys())
        feats_by_time = reconstruct_features_for_game(
            gid, home_id, away_id, pbp, team_engine, roster_engine, target_times)

        count = 0
        for sec in target_times:
            if sec not in feats_by_time:
                continue
            feats = feats_by_time[sec]

            # Model prediction
            try:
                params = model.predict_distribution_params(feats)
                model_mean = float(np.mean(params['mean']))
                model_std = float(np.mean(params['std']))
            except Exception:
                continue
            if model_std <= 0:
                continue

            # Market-implied distribution
            market_loc, market_std = extract_market_vol(snapshots[sec], home_tri, away_tri)
            if market_std is None:
                continue

            vol_ratio = model_std / market_std

            # Game phase
            if sec < 360:
                phase = 'Crunch (<6m)'
            elif sec < 720:
                phase = 'Late (6-12m)'
            elif sec < 1440:
                phase = 'Mid (12-24m)'
            elif sec < 2160:
                phase = 'Early-Mid (24-36m)'
            else:
                phase = 'Early (36-48m)'

            rows.append({
                'game_id': gid,
                'seconds_remaining': sec,
                'score_diff': feats.get('score_diff', 0),
                'model_mean': model_mean,
                'model_std': model_std,
                'market_loc': market_loc,
                'market_std': market_std,
                'vol_ratio': vol_ratio,
                'vol_diff': model_std - market_std,
                'momentum_2min': feats.get('momentum_2min', 0),
                'momentum_4min': feats.get('momentum_4min', 0),
                'score_volatility': feats.get('score_volatility', 0),
                'game_phase': phase,
                'n_tickers': len(snapshots[sec]),
            })
            count += 1

        print(f"{count} snapshots")

    return pd.DataFrame(rows)


# ── Analysis & Charts ────────────────────────────────────────────────────────

def run_analysis(df):
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  VOLATILITY ANALYSIS: Model σ vs Market-Implied σ")
    print("=" * 70)

    # ── 1. Overall Stats ──
    print(f"\n  Total snapshots: {len(df):,}")
    print(f"  Model σ:   mean={df['model_std'].mean():.2f}  median={df['model_std'].median():.2f}")
    print(f"  Market σ:  mean={df['market_std'].mean():.2f}  median={df['market_std'].median():.2f}")
    ratio_mean = df['vol_ratio'].mean()
    print(f"  Vol ratio: mean={ratio_mean:.3f}  median={df['vol_ratio'].median():.3f}")
    direction = "HIGHER" if ratio_mean > 1 else "LOWER"
    pct = abs(ratio_mean - 1) * 100
    print(f"  → Model predicts {direction} volatility than market (by {pct:.1f}%)")

    # ── 2. By Game Phase ──
    phase_order = ['Early (36-48m)', 'Early-Mid (24-36m)', 'Mid (12-24m)',
                   'Late (6-12m)', 'Crunch (<6m)']
    print(f"\n{'─' * 70}")
    print(f"  VOL RATIO BY GAME PHASE")
    print(f"{'─' * 70}")
    hdr = f"  {'Phase':<20} {'N':>8} {'Model σ':>10} {'Market σ':>10} {'Ratio':>8} {'>1 %':>8}"
    print(hdr)
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
    for phase in phase_order:
        s = df[df['game_phase'] == phase]
        if s.empty:
            continue
        gt1 = (s['vol_ratio'] > 1).mean() * 100
        print(f"  {phase:<20} {len(s):>8} {s['model_std'].mean():>10.2f} "
              f"{s['market_std'].mean():>10.2f} {s['vol_ratio'].mean():>8.3f} {gt1:>7.1f}%")

    # ── 3. By Momentum ──
    print(f"\n{'─' * 70}")
    print(f"  VOL RATIO BY 2-MINUTE MOMENTUM")
    print(f"{'─' * 70}")
    df['abs_mom'] = df['momentum_2min'].abs()
    labels = ['Calm (0-2)', 'Active (3-5)', 'Run (6-10)', 'Big Run (>10)']
    bins = pd.cut(df['abs_mom'], bins=[-1, 2, 5, 10, 100], labels=labels)
    print(f"  {'Category':<20} {'N':>8} {'Model σ':>10} {'Market σ':>10} {'Ratio':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")
    for label in labels:
        s = df[bins == label]
        if s.empty:
            continue
        print(f"  {label:<20} {len(s):>8} {s['model_std'].mean():>10.2f} "
              f"{s['market_std'].mean():>10.2f} {s['vol_ratio'].mean():>8.3f}")

    # ── 4. By Score Margin ──
    print(f"\n{'─' * 70}")
    print(f"  VOL RATIO BY SCORE MARGIN")
    print(f"{'─' * 70}")
    margin_labels = ['Tight (0-5)', 'Close (6-10)', 'Comfortable (11-20)', 'Blowout (>20)']
    margin_bins = pd.cut(df['score_diff'].abs(), bins=[-1, 5, 10, 20, 100], labels=margin_labels)
    print(f"  {'Margin':<20} {'N':>8} {'Model σ':>10} {'Market σ':>10} {'Ratio':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")
    for label in margin_labels:
        s = df[margin_bins == label]
        if s.empty:
            continue
        print(f"  {label:<20} {len(s):>8} {s['model_std'].mean():>10.2f} "
              f"{s['market_std'].mean():>10.2f} {s['vol_ratio'].mean():>8.3f}")

    # ── 5. Charts ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model σ vs Market-Implied σ', fontsize=14, fontweight='bold')

    # 5a. Histogram
    axes[0, 0].hist(df['vol_ratio'].clip(0, 3), bins=60, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(1.0, color='red', ls='--', lw=2, label='Ratio = 1.0')
    axes[0, 0].axvline(ratio_mean, color='blue', ls='--', label=f'Mean = {ratio_mean:.3f}')
    axes[0, 0].set_xlabel('Vol Ratio (model / market)')
    axes[0, 0].set_title('Distribution of Vol Ratio')
    axes[0, 0].legend()

    # 5b. Scatter
    axes[0, 1].scatter(df['market_std'], df['model_std'], alpha=0.08, s=4, color='teal')
    mx = max(df['market_std'].quantile(0.99), df['model_std'].quantile(0.99))
    axes[0, 1].plot([0, mx], [0, mx], 'r--', lw=1.5, label='y = x')
    axes[0, 1].set_xlabel('Market Implied σ (pts)')
    axes[0, 1].set_ylabel('Model Predicted σ (pts)')
    axes[0, 1].set_title('Model σ vs Market σ')
    axes[0, 1].legend()

    # 5c. Over game time
    t_groups = df.groupby(df['seconds_remaining'] // 120 * 2)  # 2-min bins
    t_ratio = t_groups['vol_ratio'].mean()
    axes[1, 0].plot(t_ratio.index, t_ratio.values, color='steelblue', lw=1.5)
    axes[1, 0].axhline(1.0, color='red', ls='--', alpha=0.5)
    axes[1, 0].set_xlabel('Minutes Remaining')
    axes[1, 0].set_ylabel('Mean Vol Ratio')
    axes[1, 0].set_title('Vol Ratio Over Game Time')
    axes[1, 0].invert_xaxis()

    # 5d. Both stds over time
    t_model = t_groups['model_std'].mean()
    t_market = t_groups['market_std'].mean()
    axes[1, 1].plot(t_model.index, t_model.values, label='Model σ', color='blue', lw=1.5)
    axes[1, 1].plot(t_market.index, t_market.values, label='Market σ', color='orange', lw=1.5)
    axes[1, 1].set_xlabel('Minutes Remaining')
    axes[1, 1].set_ylabel('Std Dev (points)')
    axes[1, 1].set_title('Model vs Market σ Over Time')
    axes[1, 1].legend()
    axes[1, 1].invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'vol_comparison.png'), dpi=150)
    print(f"\n  📊 Saved charts to {REPORT_DIR}/vol_comparison.png")

    # 5e. Momentum reaction
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    mom_bins = pd.cut(df['momentum_2min'], bins=20)
    mom_ratio = df.groupby(mom_bins, observed=True)['vol_ratio'].mean()
    if not mom_ratio.empty:
        ax2.bar(range(len(mom_ratio)), mom_ratio.values, alpha=0.7, color='steelblue')
        ax2.set_xticks(range(len(mom_ratio)))
        ax2.set_xticklabels([f'{b.mid:.0f}' for b in mom_ratio.index], rotation=45)
        ax2.axhline(1.0, color='red', ls='--')
        ax2.set_xlabel('2-minute Momentum (score diff change)')
        ax2.set_ylabel('Mean Vol Ratio (model / market)')
        ax2.set_title('Vol Ratio by Recent Momentum\n(>1 = model predicts more volatile than market)')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'momentum_vol_reaction.png'), dpi=150)
        print(f"  📊 Saved momentum chart to {REPORT_DIR}/momentum_vol_reaction.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load model
    print("Loading model...")
    model = SpreadDistributionModel(MODEL_PATH)

    # Initialize feature engines
    print("Initializing feature engines (this may take a moment)...")
    team_engine = TeamStatsEngine()
    roster_engine = RosterEngine()

    # Determine which games to process
    if args.game_ids:
        game_ids = args.game_ids
    else:
        game_ids = get_game_ids_with_data(args.date_pattern)

    if args.sample and len(game_ids) > args.sample:
        import random
        random.seed(42)
        game_ids = random.sample(game_ids, args.sample)

    if not game_ids:
        print("No games found with predictions + PBP data!")
        return

    print(f"Found {len(game_ids)} games to analyze")

    # Build comparison data
    df = build_comparison_df(game_ids, model, team_engine, roster_engine, interval=args.snapshot_interval)

    if df.empty:
        print("No comparison data generated!")
        return

    # Run analysis
    run_analysis(df)

    # Save raw data
    out_path = os.path.join(REPORT_DIR, 'vol_comparison_data.csv')
    df.to_csv(out_path, index=False)
    print(f"\n  💾 Saved raw data to {out_path}")


if __name__ == "__main__":
    main()
