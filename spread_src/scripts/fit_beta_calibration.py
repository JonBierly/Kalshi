#!/usr/bin/env python
"""
Fit Beta calibration on game-level observations.

For three price regimes (target fill prices of 25¢, 50¢, 75¢):
  1. Select one ticker per game whose mean fill price is closest to the target
  2. Each game → one independent (p_model, settled_yes) observation
  3. Fit Beta calibration: logit(P_cal) = a*log(p) - b*log(1-p) + c
  4. Show before/after calibration curves and fitted parameters

Usage:
    python -m spread_src.scripts.fit_beta_calibration --start 2026-02-09
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
import argparse
import os

DB_PATH = 'data/nba_data.db'
TARGETS = [25, 50, 75]
COLORS  = ['#e74c3c', '#2980b9', '#27ae60']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start",  type=str, default="2026-02-09")
    p.add_argument("--end",    type=str, default=None)
    p.add_argument("--output", type=str, default="reports/beta_calibration.png")
    return p.parse_args()


# ── data loading ─────────────────────────────────────────────────────────────

def load_trades(start_date=None, end_date=None):
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT game_id, ticker, side, fill_price, model_fair_value, realized_pnl, size
        FROM trades
        WHERE realized_pnl IS NOT NULL AND size > 0
    """
    conds, args = [], []
    if start_date:
        conds.append("DATE(created_at) >= ?"); args.append(start_date)
    if end_date:
        conds.append("DATE(created_at) <= ?"); args.append(end_date)
    if conds:
        query += " AND " + " AND ".join(conds)

    df = pd.read_sql_query(query, conn, params=args)
    conn.close()

    # Derive whether the market settled YES from side + pnl direction
    # buy+profit → YES settled | sell+profit → NO settled
    df['win']         = (df['realized_pnl'] > 0).astype(int)
    df['settled_yes'] = (
        ((df['side'] == 'buy')  & (df['win'] == 1)) |
        ((df['side'] == 'sell') & (df['win'] == 0))
    ).astype(int)

    return df


def build_ticker_summary(df):
    """Collapse to one row per (game_id, ticker)."""
    summary = df.groupby(['game_id', 'ticker']).agg(
        mean_fill      = ('fill_price',      'mean'),
        mean_model_prob= ('model_fair_value', 'mean'),
        settled_yes    = ('settled_yes',      lambda x: int(x.mode()[0])),  # majority vote
        n_trades       = ('realized_pnl',     'count'),
    ).reset_index()

    # p_model: model's P(YES) as a fraction
    summary['p_model'] = summary['mean_model_prob'] / 100.0

    return summary


def select_one_per_game(ticker_df, target_price):
    """For each game pick the ticker whose mean fill price is closest to target."""
    ticker_df = ticker_df.copy()
    ticker_df['dist'] = (ticker_df['mean_fill'] - target_price).abs()
    idx = ticker_df.groupby('game_id')['dist'].idxmin()
    selected = ticker_df.loc[idx].reset_index(drop=True)
    return selected


# ── Beta calibration ─────────────────────────────────────────────────────────

def beta_cal_loss(params, p, y):
    a, b, c = params
    p = np.clip(p, 1e-7, 1 - 1e-7)
    logit_cal = a * np.log(p) - b * np.log(1 - p) + c
    p_cal = np.clip(expit(logit_cal), 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal))


def fit_beta(p, y):
    result = minimize(
        beta_cal_loss,
        x0=[1.0, 1.0, 0.0],
        args=(np.array(p), np.array(y)),
        method='L-BFGS-B',
        bounds=[(0.01, 10), (0.01, 10), (-5, 5)],
    )
    return result.x  # (a, b, c)


def calibrate(p_raw, a, b, c):
    p_raw = np.clip(np.asarray(p_raw, dtype=float), 1e-7, 1 - 1e-7)
    return expit(a * np.log(p_raw) - b * np.log(1 - p_raw) + c)


def brier_score(p, y):
    return np.mean((np.array(p) - np.array(y)) ** 2)


def log_loss(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(np.array(y) * np.log(p) + (1 - np.array(y)) * np.log(1 - p))


# ── calibration summary ───────────────────────────────────────────────────────

def calibration_table(p_raw, p_cal, y, label, n_bins=6):
    """Print bucketed model% vs actual% before and after."""
    bins = np.linspace(0, 1, n_bins + 1)
    print(f"\n  {'Bucket':<14} {'N':>4}  {'Raw model%':>10}  {'Actual%':>9}  {'Raw diff':>9}  "
          f"{'Cal model%':>10}  {'Cal diff':>9}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (np.array(p_raw) >= lo) & (np.array(p_raw) < hi)
        n = mask.sum()
        if n < 3:
            continue
        raw_mean   = np.mean(np.array(p_raw)[mask])
        cal_mean   = np.mean(np.array(p_cal)[mask])
        actual     = np.mean(np.array(y)[mask])
        raw_diff   = actual - raw_mean
        cal_diff   = actual - cal_mean
        flag = " ←" if abs(raw_diff) > 0.08 else ""
        print(f"  {lo:.0%}-{hi:.0%}        {n:>4}  "
              f"{raw_mean:>10.1%}  {actual:>9.1%}  {raw_diff:>+9.1%}  "
              f"{cal_mean:>10.1%}  {cal_diff:>+9.1%}{flag}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading trades from {args.start or 'all time'}…")
    df = load_trades(args.start, args.end)
    ticker_df = build_ticker_summary(df)

    n_games = ticker_df['game_id'].nunique()
    print(f"  {len(df):,} trades  →  {len(ticker_df):,} (game, ticker) pairs  →  {n_games} games")

    fig, axes = plt.subplots(len(TARGETS), 3, figsize=(18, 5 * len(TARGETS)))
    fig.suptitle(
        f"Beta Calibration by Price Regime  |  {args.start or 'all'} →  |  {n_games} games",
        fontsize=14, fontweight='bold'
    )

    all_params = {}

    for row_idx, (target, color) in enumerate(zip(TARGETS, COLORS)):
        selected = select_one_per_game(ticker_df, target)
        p_raw = selected['p_model'].values
        y     = selected['settled_yes'].values

        # Filter to plausible range (ignore degenerate near-0 markets)
        mask  = (p_raw > 0.02) & (p_raw < 0.98)
        p_raw = p_raw[mask]
        y     = y[mask]
        n     = len(p_raw)

        # Fit
        a, b, c = fit_beta(p_raw, y)
        p_cal   = calibrate(p_raw, a, b, c)
        all_params[target] = (a, b, c)

        bs_raw = brier_score(p_raw, y)
        bs_cal = brier_score(p_cal, y)
        ll_raw = log_loss(p_raw, y)
        ll_cal = log_loss(p_cal, y)

        print(f"\n{'='*65}")
        print(f"  TARGET PRICE: {target}¢  |  n={n} games")
        print(f"{'='*65}")
        print(f"  Beta params:   a={a:.3f}  b={b:.3f}  c={c:.3f}")
        print(f"  Brier score:   raw={bs_raw:.4f}  →  cal={bs_cal:.4f}  "
              f"({'better' if bs_cal < bs_raw else 'worse'})")
        print(f"  Log-loss:      raw={ll_raw:.4f}  →  cal={ll_cal:.4f}  "
              f"({'better' if ll_cal < ll_raw else 'worse'})")
        calibration_table(p_raw, p_cal, y, target)

        # ── plots ────────────────────────────────────────────────────────────

        ax_cal  = axes[row_idx, 0]
        ax_hist = axes[row_idx, 1]
        ax_shift= axes[row_idx, 2]

        # 1. Calibration curve
        p_line  = np.linspace(0.02, 0.98, 200)
        p_c_line = calibrate(p_line, a, b, c)

        # Bucketed actuals for scatter
        bins   = np.linspace(0, 1, 9)
        bin_idx = np.digitize(p_raw, bins) - 1
        bucket_raw, bucket_act, bucket_n = [], [], []
        for i in range(len(bins) - 1):
            m = bin_idx == i
            if m.sum() >= 3:
                bucket_raw.append(p_raw[m].mean())
                bucket_act.append(y[m].mean())
                bucket_n.append(m.sum())

        ax_cal.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect', alpha=0.5)
        ax_cal.scatter(bucket_raw, bucket_act, s=[n*3 for n in bucket_n],
                       color=color, alpha=0.8, zorder=5, label='Raw (actual)')
        ax_cal.plot(p_line, p_c_line, color=color, linewidth=2, label='Beta calibrated')
        ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
        ax_cal.set_xlabel('Model probability'); ax_cal.set_ylabel('Actual win rate')
        ax_cal.set_title(f'Target={target}¢: Calibration curve  (n={n})')
        ax_cal.legend(fontsize=8)

        # 2. Distribution of selected market prices
        ax_hist.hist(selected['mean_fill'].values, bins=30, color=color, alpha=0.7, edgecolor='white')
        ax_hist.axvline(target, color='black', linestyle='--', linewidth=1.5,
                        label=f'Target={target}¢')
        ax_hist.set_xlabel('Mean fill price (¢)')
        ax_hist.set_ylabel('# Games')
        ax_hist.set_title(f'Target={target}¢: Selected market prices')
        ax_hist.legend(fontsize=8)

        # 3. Probability shift: raw vs calibrated
        p_plot  = np.linspace(0.05, 0.95, 200)
        p_c_plot = calibrate(p_plot, a, b, c)
        ax_shift.plot(p_plot * 100, p_plot * 100,  'k--', linewidth=1, alpha=0.5, label='No change')
        ax_shift.plot(p_plot * 100, p_c_plot * 100, color=color, linewidth=2,
                      label=f'Beta (a={a:.2f}, b={b:.2f}, c={c:.2f})')
        ax_shift.fill_between(p_plot * 100, p_plot * 100, p_c_plot * 100,
                              alpha=0.15, color=color)
        ax_shift.set_xlabel('Raw model probability (¢)')
        ax_shift.set_ylabel('Calibrated probability (¢)')
        ax_shift.set_title(f'Target={target}¢: Probability shift')
        ax_shift.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved → {args.output}")

    print(f"\n{'='*65}")
    print("  SUMMARY: Beta parameters by price regime")
    print(f"{'='*65}")
    print(f"  {'Target':>8}  {'a':>8}  {'b':>8}  {'c':>8}  interpretation")
    for target, (a, b, c) in all_params.items():
        if a < 1 and b < 1:
            interp = "compress both tails toward 50%"
        elif a > 1 and b > 1:
            interp = "push both tails away from 50%"
        elif b > a:
            interp = "upper tail more compressed than lower"
        else:
            interp = "lower tail more compressed than upper"
        print(f"  {target:>8}¢  {a:>8.3f}  {b:>8.3f}  {c:>8.3f}  {interp}")


if __name__ == "__main__":
    main()
