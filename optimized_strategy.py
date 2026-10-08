"""
Ensemble Strategy with Dynamic Leverage — OPTIMIZED
=====================================================
Multi-asset (SPY / QQQ / XLE) version using the best optimizer CONFIG +
SIGNAL_PARAMS found by the Bayesian optimizer (SPY, 2026-02-25).

Best Sharpe: 0.9244

Instead of being 100% in or 100% out, the strategy sizes its position
dynamically based on:

  1. VOTE CONVICTION  — how many signals agree (8-signal system)
  2. VOLATILITY RISK  — recent realized vol (high vol = reduce size)
  3. TREND STRENGTH   — price momentum confirmation (strong trend = add size)
  4. VIX LEVEL        — absolute fear gauge (very high VIX = reduce exposure)

Run:
    export FRED_API_KEY=your_key_here
    python optimized_strategy.py
"""

import sys
import gc
import json
import urllib.request
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf


import os

# Free key from https://fred.stlouisfed.org/docs/api/api_key.html
# Set it before running:  export FRED_API_KEY=your_key_here
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
if not FRED_API_KEY:
    print("WARNING: FRED_API_KEY not set. Macro signals (ISM, credit, yield curve) will be neutral.")


# ══════════════════════════════════════════════════════════════
# CONFIG  (optimizer best — Sharpe 0.9244)
# ══════════════════════════════════════════════════════════════

CONFIG = {
    "start_date":        "1993-01-01",
    "end_date":          "2024-12-31",
    "starting_capital":  100_000,
    "rebalance_days":    11,
    "n_simulations":     500,
    "transaction_cost":  0.0005,

    # Voting thresholds
    "vote_long":         3,
    "vote_cash":        -2,

    # Leverage limits
    "max_leverage":      1.3,
    "min_leverage":      0.0,
    "leverage_cost":     0.0100,

    # Volatility targeting
    "vol_target":        0.36,
    "vol_lookback":      17,

    # Conviction scaling
    "base_position":     0.90,
    "conviction_boost":  0.30,

    # Rebalance threshold
    "rebalance_threshold": 0.14,

    # Leverage gate
    "leverage_min_score":  2,
}

# Signal thresholds — optimized for Sharpe (SPY, 2026-02-25, best Sharpe 0.9244)
SIGNAL_PARAMS = {
    "hy_z_bear":             0.7500,
    "hy_z_bull":             0.9000,
    "ism_z_bear":            0.3000,
    "ism_z_bull":            0.5500,
    "ma_bear_pct":           0.0150,
    "ma_bull_pct":           0.0050,
    "ma_window":           116,
    "mom_bear_ret":          0.1100,
    "mom_bull_ret":          0.1500,
    "mom_lookback":        159,
    "mom_skip":             10,
    "mom_size_sensitivity":  1.5000,
    "mom_size_window":      69,
    "sent_z_bear":           0.5000,
    "sent_z_bull":           0.2500,
    "vix_mc_bear_median":   29.5032,
    "vix_mc_bull_median":   19.7971,
    "vix_mc_prob_cutoff":    0.1500,
    "vix_mc_prob_threshold": 21.2327,
    "vix_size_neutral":     22.9876,
    "vix_size_slope":        0.0100,
    "vix_ts_bear":           1.5000,
    "vix_ts_bull":          -0.7000,
    "yc_bear":              -0.5000,
    "yc_bull":               1.7000,
}


# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════

def clean_yahoo(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ['_'.join(c).strip() for c in raw.columns.values]
        col = [c for c in raw.columns if 'Close' in c][0]
    else:
        col = 'Close'
    s = raw[col].dropna()
    return pd.Series(np.array(s.values, dtype=float).flatten(), index=s.index)


def fetch_fred_series(series_id, start, end):
    """Pull a FRED series via direct API, forward-fill gaps to business days."""
    try:
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            "?series_id=" + series_id +
            "&observation_start=" + start +
            "&observation_end=" + end +
            "&api_key=" + FRED_API_KEY +
            "&file_type=json"
        )
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        obs = data.get("observations", [])
        if not obs:
            return pd.Series(dtype=float)
        dates  = pd.to_datetime([o["date"] for o in obs])
        values = pd.to_numeric([o["value"] for o in obs], errors="coerce")
        s = pd.Series(values, index=dates)
    except Exception as e:
        print(f"    FRED fetch failed for {series_id}: {e}")
        s = pd.Series(dtype=float)
    daily_idx = pd.date_range(start, end, freq="B")
    return s.reindex(daily_idx).ffill()


def fetch_all(cfg, ticker="SPY"):
    s, e = cfg["start_date"], cfg["end_date"]
    print(f"Fetching data for {ticker}...")
    asset = clean_yahoo(ticker,   s, e); print(f"  {ticker}:  {len(asset)} days")
    vix   = clean_yahoo("^VIX",   s, e); print(f"  VIX:    {len(vix)} days")
    vix3m = clean_yahoo("^VIX3M", s, e); print(f"  VIX3M:  {len(vix3m)} days (from 2007)")
    iwm   = clean_yahoo("IWM",    s, e); print(f"  IWM:    {len(iwm)} days (from 2000)")
    tlt   = clean_yahoo("TLT",    s, e); print(f"  TLT:    {len(tlt)} days (from 2002)")

    # Sentiment signal always compares IWM vs SPY (small cap vs large cap)
    # so we need SPY even when the traded asset is QQQ or XLE
    if ticker != "SPY":
        spy_s = clean_yahoo("SPY", s, e)
    else:
        spy_s = asset

    print("  Fetching FRED data...")
    indpro      = fetch_fred_series("INDPRO",       s, e)
    hy_spread   = fetch_fred_series("BAMLH0A0HYM2", s, e)  # HY spread from 1996
    dgs10       = fetch_fred_series("DGS10",        s, e)
    dgs2        = fetch_fred_series("DGS2",         s, e)
    yield_curve = (dgs10 - dgs2)

    # Common index: traded asset x VIX — everything else slots in as available
    common = asset.index.intersection(vix.index)
    asset = asset.loc[common]
    vix   = vix.loc[common]

    # Optional series: NaN before their inception dates, signals handle NaN as neutral (0)
    vix_ts      = (vix - vix3m.reindex(common))                          # NaN pre-2007
    tlt         = tlt.reindex(common)                                     # NaN pre-2002
    iwm_r       = iwm.reindex(common)                                     # NaN pre-2000
    ism         = indpro.reindex(common).ffill()                           # INDPRO from 1919
    sent        = (iwm_r.pct_change(21) - spy_s.reindex(common).pct_change(21))  # NaN pre-2000
    hy_spread   = hy_spread.reindex(common).ffill()                       # NaN pre-1996
    yield_curve = yield_curve.reindex(common).ffill()

    print(f"  Aligned: {len(common)} days  ({common[0].date()} -> {common[-1].date()})")
    return asset, vix, vix_ts, ism, sent, tlt, hy_spread, yield_curve


# ══════════════════════════════════════════════════════════════
# SIGNALS (returns -1 / 0 / +1)
# ══════════════════════════════════════════════════════════════

def calibrate_ou(v_array):
    v  = np.array(v_array, dtype=float).flatten()
    v  = v[~np.isnan(v)]
    dt = 1.0 / 252.0
    dv, v_lag = np.diff(v), v[:-1]
    A  = np.stack([np.ones(len(v_lag), dtype=float), v_lag], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(A, dv, rcond=None)
    a, b  = float(coeffs[0]), float(coeffs[1])
    kappa = max(-b / dt, 0.01)
    theta = a / (kappa * dt)
    sigma = float(np.std(dv - (a + b * v_lag))) / np.sqrt(dt)
    return kappa, theta, sigma


def signal_vix_mc(vix_hist, vix0, n_sims):
    window = vix_hist[-1260:] if len(vix_hist) > 1260 else vix_hist
    kappa, theta, sigma = calibrate_ou(window)
    if vix0 > 30:   kappa *= 1.5; sigma *= 1.4
    elif vix0 < 15: kappa *= 0.7; sigma *= 0.8
    dt = 1.0/252.0; d = 7
    paths = np.zeros((n_sims, d+1)); paths[:,0] = vix0
    for t in range(1, d+1):
        z = np.random.standard_normal(n_sims)
        paths[:,t] = np.maximum(
            paths[:,t-1] + kappa*(theta-paths[:,t-1])*dt + sigma*np.sqrt(dt)*z, 5.0)
    final = paths[:,-1]
    sp   = SIGNAL_PARAMS
    med  = float(np.median(final))
    prob = float(np.mean(final > sp["vix_mc_prob_threshold"]))
    if prob > sp["vix_mc_prob_cutoff"] or med > sp["vix_mc_bear_median"]: return -1
    if med < sp["vix_mc_bull_median"]:                                     return  1
    return 0


def signal_vix_ts(ts_current):
    sp = SIGNAL_PARAMS
    if ts_current > sp["vix_ts_bear"]: return -1
    if ts_current < sp["vix_ts_bull"]: return  1
    return 0


def signal_ism(ism_hist, ism_current):
    hist = np.array(ism_hist, dtype=float).flatten()
    hist = hist[~np.isnan(hist)]
    if len(hist) < 60: return 0
    mean = float(np.mean(hist[-252:] if len(hist) >= 252 else hist))
    std  = float(np.std( hist[-252:] if len(hist) >= 252 else hist))
    if std == 0: return 0
    sp = SIGNAL_PARAMS
    z = (ism_current - mean) / std
    if z >  sp["ism_z_bull"]: return  1
    if z < -sp["ism_z_bear"]: return -1
    return 0


def signal_sentiment(sent_hist, sent_current):
    hist = np.array(sent_hist, dtype=float).flatten()
    hist = hist[~np.isnan(hist)]
    if len(hist) < 60: return 0
    mean = float(np.mean(hist[-252:] if len(hist) >= 252 else hist))
    std  = float(np.std( hist[-252:] if len(hist) >= 252 else hist))
    if std == 0: return 0
    sp = SIGNAL_PARAMS
    z = (sent_current - mean) / std
    if z >  sp["sent_z_bull"]: return  1
    if z < -sp["sent_z_bear"]: return -1
    return 0


def signal_ma200(spy_hist):
    sp     = SIGNAL_PARAMS
    prices = np.array(spy_hist, dtype=float).flatten()
    prices = prices[~np.isnan(prices)]
    win    = sp["ma_window"]
    if len(prices) < win: return 0
    ma_val = float(np.mean(prices[-win:]))
    pct    = (float(prices[-1]) - ma_val) / ma_val
    if pct >  sp["ma_bull_pct"]: return  1
    if pct < -sp["ma_bear_pct"]: return -1
    return 0


def signal_momentum_12_1(spy_hist):
    sp     = SIGNAL_PARAMS
    prices = np.array(spy_hist, dtype=float).flatten()
    prices = prices[~np.isnan(prices)]
    lb     = sp["mom_lookback"]
    sk     = sp["mom_skip"]
    if len(prices) < lb + sk: return 0
    ret = (float(prices[-sk]) / float(prices[-lb])) - 1.0
    if ret >  sp["mom_bull_ret"]: return  1
    if ret < -sp["mom_bear_ret"]: return -1
    return 0


def signal_credit_spread(spread_hist, spread_now):
    """HY credit spread: wide = stress (bearish), tight = risk-on (bullish)."""
    hist = np.array(spread_hist, dtype=float).flatten()
    hist = hist[~np.isnan(hist)]
    if len(hist) < 60 or np.isnan(spread_now): return 0
    mean = float(np.mean(hist[-252:] if len(hist) >= 252 else hist))
    std  = float(np.std( hist[-252:] if len(hist) >= 252 else hist))
    if std == 0: return 0
    sp = SIGNAL_PARAMS
    z = (spread_now - mean) / std
    if z >  sp["hy_z_bear"]: return -1   # spread spiked: credit stress
    if z < -sp["hy_z_bull"]: return  1   # spread tight: risk-on
    return 0


def signal_yield_curve(curve_now):
    """10Y-2Y yield curve: inverted = bearish, steep = bullish."""
    sp = SIGNAL_PARAMS
    if np.isnan(curve_now): return 0
    if curve_now < sp["yc_bear"]: return -1   # inverted / low: recession signal
    if curve_now > sp["yc_bull"]: return  1   # steep: healthy economy
    return 0


# ══════════════════════════════════════════════════════════════
# CONVICTION & LEVERAGE CALCULATOR
# ══════════════════════════════════════════════════════════════

def calculate_position_size(score, spy_hist, vix_current, cfg):
    """
    Calculates how much of capital to deploy (0.0 to max_leverage).

    Factors:
      1. Vote score      — more agreement = higher base position
      2. Realized vol    — high recent vol = scale down
      3. VIX level       — very high fear = scale down further
      4. Price momentum  — strong uptrend = slight boost

    Returns: position_size (float), breakdown (dict for display)
    """

    # ── 1. Base position from vote score ───────────────────────
    if score <= cfg["vote_cash"]:
        return 0.0, {"vote": 0.0, "vol_scalar": 1.0, "vix_scalar": 1.0,
                     "momentum_scalar": 1.0, "final": 0.0, "score": score}

    extra_votes  = max(score - cfg["vote_long"], 0)
    base_pos     = cfg["base_position"] + extra_votes * cfg["conviction_boost"]

    # ── 2. Volatility scalar ────────────────────────────────────
    prices  = np.array(spy_hist, dtype=float).flatten()
    prices  = prices[~np.isnan(prices)]
    if len(prices) < cfg["vol_lookback"] + 1:
        vol_scalar = 1.0
        realized_vol = 0.15
    else:
        recent_returns = np.diff(np.log(prices[-cfg["vol_lookback"]-1:]))
        realized_vol   = float(np.std(recent_returns) * np.sqrt(252))
        realized_vol   = max(realized_vol, 0.05)
        vol_scalar = cfg["vol_target"] / realized_vol
        vol_scalar = np.clip(vol_scalar, 0.3, 1.8)

    # ── 3. VIX level scalar ─────────────────────────────────────
    sp = SIGNAL_PARAMS
    vix_scalar = np.clip(1.0 - (vix_current - sp["vix_size_neutral"]) * sp["vix_size_slope"], 0.5, 1.1)

    # ── 4. Momentum scalar ──────────────────────────────────────
    win = sp["mom_size_window"]
    if len(prices) >= win:
        ma_val    = float(np.mean(prices[-win:]))
        price_now = float(prices[-1])
        mom_pct   = (price_now - ma_val) / ma_val
        momentum_scalar = np.clip(1.0 + mom_pct * sp["mom_size_sensitivity"], 0.80, 1.20)
    else:
        momentum_scalar = 1.0

    # ── Combine ─────────────────────────────────────────────────
    raw_size   = base_pos * vol_scalar * vix_scalar * momentum_scalar
    final_size = np.clip(raw_size, cfg["min_leverage"], cfg["max_leverage"])

    breakdown = {
        "score":            score,
        "base_pos":         base_pos,
        "realized_vol":     realized_vol,
        "vol_scalar":       vol_scalar,
        "vix_level":        vix_current,
        "vix_scalar":       vix_scalar,
        "momentum_scalar":  momentum_scalar,
        "raw_size":         raw_size,
        "final":            final_size,
    }

    return final_size, breakdown


# ══════════════════════════════════════════════════════════════
# BACKTESTER
# ══════════════════════════════════════════════════════════════

def run_backtest(spy, vix, vix_ts, ism, sent, tlt, hy_spread, yield_curve, cfg):
    dates    = spy.index
    capital  = cfg["starting_capital"]
    tc       = cfg["transaction_cost"]
    n_sims   = cfg["n_simulations"]
    reb      = cfg["rebalance_days"]
    min_hist = 252
    borrow_cost_daily = cfg["leverage_cost"] / 252

    cash          = float(capital)
    spy_shares    = 0.0
    tlt_shares    = 0.0
    current_size  = 0.0
    current_tlt_size = 0.0

    portfolio_values  = []
    position_sizes    = []
    tlt_sizes         = []
    vote_scores       = []
    trade_log         = []
    breakdown_log     = []

    print(f"\nRunning leveraged backtest (8-signal + TLT rotation)...")

    portfolio_value = float(capital)

    for i, date in enumerate(dates):
        price  = float(spy.iloc[i])
        tlt_px = float(tlt.iloc[i]) if not np.isnan(tlt.iloc[i]) else None

        # Apply daily borrowing cost on leveraged portion
        if current_size > 1.0:
            pv_now           = cash + spy_shares * price + (tlt_shares * tlt_px if tlt_px else 0.0)
            leveraged_amount = (current_size - 1.0) * pv_now
            cash            -= leveraged_amount * borrow_cost_daily

        if i >= min_hist and i % reb == 0:

            # -- Compute 8 votes ------------------------------------------
            v1 = signal_vix_mc(vix.values[:i], float(vix.iloc[i]), n_sims)
            v2 = signal_vix_ts(float(vix_ts.iloc[i]) if not np.isnan(vix_ts.iloc[i]) else 0.0)
            ism_cur  = float(ism.iloc[i])  if not np.isnan(ism.iloc[i])  else 0.0
            sent_cur = float(sent.iloc[i]) if not np.isnan(sent.iloc[i]) else 0.0
            v3 = signal_ism(ism.values[:i], ism_cur)
            v4 = signal_sentiment(sent.values[:i], sent_cur)
            v5 = signal_ma200(spy.values[:i])
            v6 = signal_momentum_12_1(spy.values[:i])
            hy_cur = float(hy_spread.iloc[i])  if not np.isnan(hy_spread.iloc[i])  else float('nan')
            yc_cur = float(yield_curve.iloc[i]) if not np.isnan(yield_curve.iloc[i]) else float('nan')
            v7 = signal_credit_spread(hy_spread.values[:i], hy_cur)
            v8 = signal_yield_curve(yc_cur)

            score = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8
            vote_scores.append({"date": date, "score": score,
                                 "v1": v1, "v2": v2, "v3": v3, "v4": v4,
                                 "v5": v5, "v6": v6, "v7": v7, "v8": v8})

            # -- Calculate position size ----------------------------------
            target_size, breakdown = calculate_position_size(
                score, spy.values[:i], float(vix.iloc[i]), cfg)
            breakdown["date"] = date
            breakdown_log.append(breakdown)

            # Gate: leverage only with sufficient conviction
            if score < cfg.get("leverage_min_score", 2):
                target_size = min(target_size, 1.0)

            # -- Rebalance SPY --------------------------------------------
            target_spy_value  = portfolio_value * target_size
            current_spy_value = spy_shares * price
            threshold = cfg.get("rebalance_threshold", 0.06)
            spy_change = abs(target_spy_value - current_spy_value) / (portfolio_value + 1e-10)

            if spy_change > threshold:
                old_size = current_size
                if target_spy_value > current_spy_value:
                    buy_value   = target_spy_value - current_spy_value
                    spy_shares += buy_value / price
                    cash       -= buy_value * (1 + tc)
                    action = "BUY"
                else:
                    sell_value  = current_spy_value - target_spy_value
                    spy_shares -= sell_value / price
                    cash       += sell_value * (1 - tc)
                    action = "SELL" if target_size == 0 else "REDUCE"
                current_size = target_size
                trade_log.append({
                    "date": date, "action": action, "price": price,
                    "size": target_size, "score": score,
                    "vix": float(vix.iloc[i]), "old_size": old_size,
                })

            # -- TLT rotation (bonds when fully out of SPY) ---------------
            target_tlt_size   = 1.0 if (target_size == 0 and tlt_px) else 0.0
            target_tlt_value  = portfolio_value * target_tlt_size
            current_tlt_value = tlt_shares * tlt_px if tlt_px else 0.0
            tlt_change = abs(target_tlt_value - current_tlt_value) / (portfolio_value + 1e-10)

            if tlt_change > threshold and tlt_px:
                if target_tlt_value > current_tlt_value:
                    buy_val     = target_tlt_value - current_tlt_value
                    tlt_shares += buy_val / tlt_px
                    cash       -= buy_val * (1 + tc)
                else:
                    sell_val    = current_tlt_value - target_tlt_value
                    tlt_shares -= sell_val / tlt_px
                    cash       += sell_val * (1 - tc)
                current_tlt_size = target_tlt_size

        tlt_val         = tlt_shares * tlt_px if tlt_px else tlt_shares * 0.0
        portfolio_value = cash + spy_shares * price + tlt_val
        portfolio_values.append(portfolio_value)
        position_sizes.append(current_size)
        tlt_sizes.append(current_tlt_size)

    df = pd.DataFrame({
        "value":    portfolio_values,
        "leverage": position_sizes,
        "tlt_pos":  tlt_sizes,
    }, index=dates)

    votes_df     = pd.DataFrame(vote_scores).set_index("date") if vote_scores else pd.DataFrame()
    trades_df    = pd.DataFrame(trade_log)  if trade_log  else pd.DataFrame()
    breakdown_df = pd.DataFrame(breakdown_log).set_index("date") if breakdown_log else pd.DataFrame()

    print(f"  Total trades/rebalances: {len(trades_df)}")
    if not trades_df.empty:
        buys    = len(trades_df[trades_df["action"] == "BUY"])
        sells   = len(trades_df[trades_df["action"] == "SELL"])
        reduces = len(trades_df[trades_df["action"] == "REDUCE"])
        print(f"  Buys: {buys}  |  Sells: {sells}  |  Reduces: {reduces}")
        if "size" in trades_df.columns:
            print(f"  Avg position size: {trades_df['size'].mean():.2f}x  |  Max leverage used: {trades_df['size'].max():.2f}x")

    return df, trades_df, votes_df, breakdown_df


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════

def compute_metrics(values, index):
    v = np.array(values, dtype=float)
    r = np.diff(v) / (v[:-1] + 1e-10)
    nyears  = len(v) / 252
    cagr    = ((v[-1]/v[0])**(1/nyears) - 1) * 100
    vol     = np.std(r) * np.sqrt(252) * 100
    sharpe  = (np.mean(r)*252) / (np.std(r)*np.sqrt(252) + 1e-10)
    peak    = np.maximum.accumulate(v)
    max_dd  = ((v - peak)/peak).min() * 100
    monthly = pd.Series(v, index=index).resample('ME').last()
    win_rt  = (monthly.pct_change().dropna() > 0).mean() * 100
    calmar  = cagr / abs(max_dd) if max_dd != 0 else 0
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe,
            "MaxDD": max_dd, "WinRate": win_rt, "Calmar": calmar, "Final": v[-1]}


# ══════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════

def plot(df, trades_df, votes_df, breakdown_df, spy, cfg, ticker="SPY"):
    capital = cfg["starting_capital"]
    dates   = df.index
    spy_al  = spy.loc[dates]
    bnh_v   = capital * spy_al.values / spy_al.values[0]
    bnh_m   = compute_metrics(bnh_v, dates)
    lev_v   = df["value"].values
    lev_m   = compute_metrics(lev_v, dates)
    leverage = df["leverage"].values

    fig = plt.figure(figsize=(20, 16), facecolor="#0d1117")
    fig.suptitle(f"Dynamic Leverage Ensemble — OPTIMIZED [{ticker}]",
                 fontsize=19, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3,
                           left=0.07, right=0.97, top=0.95, bottom=0.04)

    ax_port  = fig.add_subplot(gs[0, :])
    ax_lev   = fig.add_subplot(gs[1, :])
    ax_dd    = fig.add_subplot(gs[2, 0])
    ax_votes = fig.add_subplot(gs[2, 1])
    ax_dist  = fig.add_subplot(gs[3, 0])
    ax_tbl   = fig.add_subplot(gs[3, 1])

    # ── Portfolio ─────────────────────────────────────────────
    ax_port.set_facecolor("#161b22")
    ax_port.plot(dates, bnh_v, color="#ffd700", linewidth=2.0,
                 linestyle="--", label=f"Buy & Hold {ticker} (1x)", alpha=0.9)
    ax_port.plot(dates, lev_v, color="#4fc3f7", linewidth=2.2,
                 label=f"Dynamic Leverage Ensemble OPTIMIZED ({ticker})", zorder=5)
    ax_port.fill_between(dates, lev_v, bnh_v,
                          where=lev_v >= bnh_v, alpha=0.15, color="#2ecc71",
                          label="Outperforming B&H")
    ax_port.fill_between(dates, lev_v, bnh_v,
                          where=lev_v < bnh_v,  alpha=0.15, color="#e74c3c",
                          label="Underperforming B&H")
    if not trades_df.empty:
        lev_trades = trades_df[trades_df["size"] > 1.0]
        for _, row in lev_trades.iterrows():
            ax_port.axvline(row["date"], color="#ffd700", linewidth=0.6, alpha=0.4)
    ax_port.axhline(capital, color="#444", linewidth=0.7, linestyle=":")
    ax_port.set_ylabel("Portfolio Value ($)", color="#8b9dc3", fontsize=10)
    ax_port.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
    ax_port.tick_params(colors="#8b9dc3")
    for sp in ax_port.spines.values(): sp.set_color("#30363d")
    ax_port.legend(loc="upper left", fontsize=9, facecolor="#161b22",
                   labelcolor="white", edgecolor="#30363d", ncol=2)
    ax_port.set_title("Portfolio Value  (yellow verticals = leveraged positions)",
                      color="#8b9dc3", fontsize=11, pad=6)

    # ── Leverage over time ────────────────────────────────────
    ax_lev.set_facecolor("#161b22")
    lev_arr = np.array(leverage)
    ax_lev.fill_between(dates, lev_arr, 0,
                         where=lev_arr > 1.0, alpha=0.7, color="#ffd700",
                         label="Leveraged (>1x)")
    ax_lev.fill_between(dates, lev_arr, 0,
                         where=(lev_arr > 0) & (lev_arr <= 1.0), alpha=0.5,
                         color="#4fc3f7", label="Invested (0-1x)")
    ax_lev.fill_between(dates, lev_arr, 0,
                         where=lev_arr == 0, alpha=0.5, color="#e74c3c",
                         label="Cash (0x)")
    ax_lev.plot(dates, lev_arr, color="white", linewidth=0.8, alpha=0.7)
    ax_lev.axhline(1.0, color="#888", linewidth=0.8, linestyle="--", alpha=0.6,
                    label="1x (no leverage)")
    ax_lev.axhline(cfg["max_leverage"], color="#ffd700", linewidth=0.8,
                    linestyle=":", alpha=0.6, label=f"Max ({cfg['max_leverage']}x)")
    ax_lev.set_ylabel("Position Size (x capital)", color="#8b9dc3", fontsize=10)
    ax_lev.set_ylim(-0.05, cfg["max_leverage"] + 0.15)
    ax_lev.tick_params(colors="#8b9dc3")
    for sp in ax_lev.spines.values(): sp.set_color("#30363d")
    ax_lev.legend(loc="upper right", fontsize=8, facecolor="#161b22",
                  labelcolor="white", edgecolor="#30363d", ncol=4)
    ax_lev.set_title("Dynamic Position Size Over Time  (blue=invested, yellow=leveraged, red=cash)",
                     color="#8b9dc3", fontsize=10, pad=6)

    # ── Drawdown ──────────────────────────────────────────────
    ax_dd.set_facecolor("#161b22")
    def dd_fn(v):
        peak = np.maximum.accumulate(v); return (v-peak)/peak*100

    ax_dd.fill_between(dates, dd_fn(bnh_v), 0, alpha=0.2, color="#ffd700")
    ax_dd.fill_between(dates, dd_fn(lev_v), 0, alpha=0.3, color="#4fc3f7")
    ax_dd.plot(dates, dd_fn(bnh_v), color="#ffd700", linewidth=1.2,
               linestyle="--", label=f"B&H  (max {bnh_m['MaxDD']:.1f}%)")
    ax_dd.plot(dates, dd_fn(lev_v), color="#4fc3f7", linewidth=1.5,
               label=f"Dynamic  (max {lev_m['MaxDD']:.1f}%)")
    ax_dd.set_ylabel("Drawdown (%)", color="#8b9dc3", fontsize=9)
    ax_dd.tick_params(colors="#8b9dc3")
    for sp in ax_dd.spines.values(): sp.set_color("#30363d")
    ax_dd.legend(fontsize=8.5, facecolor="#161b22", labelcolor="white", edgecolor="#30363d")
    ax_dd.set_title("Drawdown Comparison", color="#8b9dc3", fontsize=10, pad=6)

    # ── Vote scores ───────────────────────────────────────────
    ax_votes.set_facecolor("#161b22")
    if not votes_df.empty:
        score = votes_df["score"]
        ax_votes.fill_between(score.index, score.values, 0,
                               where=score.values >= 0, alpha=0.6, color="#34d399")
        ax_votes.fill_between(score.index, score.values, 0,
                               where=score.values < 0,  alpha=0.6, color="#e74c3c")
        ax_votes.plot(score.index, score.values, color="white", linewidth=0.5, alpha=0.4)
        ax_votes.axhline( cfg["vote_long"], color="#34d399", linewidth=1.0,
                          linestyle="--", alpha=0.7)
        ax_votes.axhline( cfg["vote_cash"], color="#e74c3c", linewidth=1.0,
                          linestyle="--", alpha=0.7)
        ax_votes.axhline(0, color="#555", linewidth=0.6)
        ax_votes.set_ylabel("Vote Score", color="#8b9dc3", fontsize=9)
        ax_votes.set_yticks([-6,-4,-2,0,2,4,6])
    ax_votes.tick_params(colors="#8b9dc3")
    for sp in ax_votes.spines.values(): sp.set_color("#30363d")
    ax_votes.set_title("Signal Vote Score", color="#8b9dc3", fontsize=10, pad=6)

    # ── Leverage distribution ─────────────────────────────────
    ax_dist.set_facecolor("#161b22")
    nonzero_lev = lev_arr[lev_arr > 0.01]
    if len(nonzero_lev) > 0:
        bins = np.linspace(0, cfg["max_leverage"] + 0.1, 30)
        ax_dist.hist(nonzero_lev, bins=bins, color="#4fc3f7", alpha=0.75,
                     edgecolor="none", density=True)
        ax_dist.axvline(float(np.mean(nonzero_lev)), color="#ffd700", linewidth=1.5,
                        linestyle="--", label=f"Mean: {np.mean(nonzero_lev):.2f}x")
        ax_dist.axvline(1.0, color="#888", linewidth=1.0, linestyle=":",
                        label="1x (no leverage)")
        pct_leveraged = float(np.mean(lev_arr > 1.0) * 100)
        pct_cash      = float(np.mean(lev_arr == 0) * 100)
        ax_dist.text(0.97, 0.95, f"Leveraged: {pct_leveraged:.1f}% of days\nIn cash: {pct_cash:.1f}% of days",
                     transform=ax_dist.transAxes, color="white", fontsize=8,
                     ha='right', va='top', fontfamily='monospace')
    ax_dist.set_xlabel("Position Size (x)", color="#8b9dc3", fontsize=9)
    ax_dist.set_ylabel("Density", color="#8b9dc3", fontsize=9)
    ax_dist.tick_params(colors="#8b9dc3")
    for sp in ax_dist.spines.values(): sp.set_color("#30363d")
    ax_dist.legend(fontsize=8, facecolor="#161b22", labelcolor="white", edgecolor="#30363d")
    ax_dist.set_title("Position Size Distribution", color="#8b9dc3", fontsize=10, pad=6)

    # ── Metrics table ─────────────────────────────────────────
    ax_tbl.set_facecolor("#161b22")
    ax_tbl.axis('off')

    def txt(x, y, s, color="white", size=9, bold=False):
        ax_tbl.text(x, y, s, transform=ax_tbl.transAxes,
                   color=color, fontsize=size,
                   fontweight='bold' if bold else 'normal',
                   fontfamily='monospace', va='top')

    def better(lv, bh, higher_is_better=True):
        try:
            l = float(str(lv).replace('$','').replace('%','').replace(',','').replace('x',''))
            b = float(str(bh).replace('$','').replace('%','').replace(',','').replace('x',''))
            if higher_is_better: return "#2ecc71" if l > b else "#e74c3c"
            else:                return "#2ecc71" if l < b else "#e74c3c"
        except (ValueError, TypeError): return "white"

    y = 0.97
    txt(0.02, y, "PERFORMANCE METRICS", color="#8b9dc3", size=11, bold=True); y -= 0.08
    txt(0.02, y, f"{'Metric':<20}{'Dynamic':>10}{'B&H':>10}", color="#8b9dc3", size=8.5, bold=True); y -= 0.05
    txt(0.02, y, "─"*40, color="#30363d", size=7); y -= 0.06

    rows = [
        ("Final Value",  f"${lev_m['Final']:,.0f}",  f"${bnh_m['Final']:,.0f}",  True),
        ("CAGR",         f"{lev_m['CAGR']:.1f}%",    f"{bnh_m['CAGR']:.1f}%",    True),
        ("Volatility",   f"{lev_m['Vol']:.1f}%",      f"{bnh_m['Vol']:.1f}%",     False),
        ("Sharpe Ratio", f"{lev_m['Sharpe']:.2f}",    f"{bnh_m['Sharpe']:.2f}",   True),
        ("Max Drawdown", f"{lev_m['MaxDD']:.1f}%",    f"{bnh_m['MaxDD']:.1f}%",   False),
        ("Win Rate",     f"{lev_m['WinRate']:.0f}%",  f"{bnh_m['WinRate']:.0f}%", True),
        ("Calmar Ratio", f"{lev_m['Calmar']:.2f}",    f"{bnh_m['Calmar']:.2f}",   True),
    ]

    for label, lv, bh, hib in rows:
        col = better(lv, bh, hib)
        txt(0.02, y, f"{label:<20}", color="#8b9dc3", size=8.5)
        txt(0.55, y, f"{lv:>10}", color=col, size=8.5, bold=True)
        txt(0.78, y, f"{bh:>10}", color="#ffd700", size=8.5)
        y -= 0.07

    txt(0.02, y, "─"*40, color="#30363d", size=7); y -= 0.06
    txt(0.02, y, "LEVERAGE STATS", color="#8b9dc3", size=10, bold=True); y -= 0.06

    if not trades_df.empty and "size" in trades_df.columns:
        avg_lev = trades_df["size"].mean()
        max_lev = trades_df["size"].max()
        pct_lev = float(np.mean(lev_arr > 1.0) * 100)
        pct_csh = float(np.mean(lev_arr == 0.0) * 100)
        txt(0.02, y, f"Avg position size:  {avg_lev:.2f}x", color="white",   size=8.5); y -= 0.06
        txt(0.02, y, f"Max leverage used:  {max_lev:.2f}x", color="#ffd700", size=8.5); y -= 0.06
        txt(0.02, y, f"Days leveraged:     {pct_lev:.1f}%", color="#ffd700", size=8.5); y -= 0.06
        txt(0.02, y, f"Days in cash:       {pct_csh:.1f}%", color="#e74c3c", size=8.5); y -= 0.06
        txt(0.02, y, f"Total rebalances:   {len(trades_df)}",color="white",  size=8.5); y -= 0.06

    txt(0.02, y, "─"*40, color="#30363d", size=7); y -= 0.06
    txt(0.02, y, "CONVICTION FORMULA", color="#8b9dc3", size=9, bold=True); y -= 0.055
    txt(0.02, y, "size = base × vol_scalar",  color="#4fc3f7", size=7.5); y -= 0.05
    txt(0.02, y, "       × vix_scalar",        color="#4fc3f7", size=7.5); y -= 0.05
    txt(0.02, y, "       × momentum_scalar",   color="#4fc3f7", size=7.5); y -= 0.05
    txt(0.02, y, f"vol target: {cfg['vol_target']*100:.0f}%  max: {cfg['max_leverage']}x", color="white", size=7.5)

    out_file = f"optimized_{ticker}_output.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight', facecolor="#0d1117")
    print(f"Chart saved: {out_file}")
    plt.show()
    plt.close(fig)
    gc.collect()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)

    ASSET_STARTS = {
        "SPY": "1993-01-01",
        "QQQ": "1999-03-10",
        "XLE": "1998-12-22",
    }

    for ticker in ["SPY", "QQQ", "XLE"]:
        print(f"\n{'═'*60}")
        print(f"  RUNNING: {ticker}  [OPTIMIZED — Sharpe 0.9244]")
        print(f"{'═'*60}")

        cfg = dict(CONFIG)
        cfg["start_date"] = ASSET_STARTS[ticker]

        asset, vix, vix_ts, ism, sent, tlt, hy_spread, yield_curve = fetch_all(cfg, ticker)

        df, trades_df, votes_df, breakdown_df = run_backtest(
            asset, vix, vix_ts, ism, sent, tlt, hy_spread, yield_curve, cfg)

        capital = cfg["starting_capital"]
        bnh_v   = capital * asset.values / asset.values[0]
        bnh_m   = compute_metrics(bnh_v, asset.index)
        lev_m   = compute_metrics(df["value"].values, df.index)

        print(f"\n{'═'*52}")
        print(f"  FINAL RESULTS: {ticker}  [OPTIMIZED]")
        print(f"{'═'*52}")
        print(f"  {'Metric':<22} {'Dynamic':>12} {'B&H':>12}")
        print(f"  {'─'*48}")
        for label, lk, bk in [
            ("Final Value",  "Final",  "Final"),
            ("CAGR",         "CAGR",   "CAGR"),
            ("Volatility",   "Vol",    "Vol"),
            ("Sharpe Ratio", "Sharpe", "Sharpe"),
            ("Max Drawdown", "MaxDD",  "MaxDD"),
            ("Calmar Ratio", "Calmar", "Calmar"),
        ]:
            lv = lev_m[lk]; bv = bnh_m[bk]
            if lk == "Final":
                print(f"  {label:<22} ${lv:>10,.0f} ${bv:>10,.0f}")
            elif lk in ("CAGR", "Vol", "MaxDD"):
                print(f"  {label:<22} {lv:>11.1f}% {bv:>11.1f}%")
            else:
                print(f"  {label:<22} {lv:>12.2f} {bv:>12.2f}")
        print(f"  {'─'*48}")
        if not trades_df.empty and "size" in trades_df.columns:
            print(f"  {'Total rebalances':<22} {len(trades_df):>12}")
            print(f"  {'Max leverage used':<22} {trades_df['size'].max():>11.2f}x")
            print(f"  {'Avg position size':<22} {trades_df['size'].mean():>11.2f}x")
        print(f"{'═'*52}\n")

        # ── Sub-period breakdown ──────────────────────────────
        periods = [
            ("1994-1999  (tech bubble)",   "1994-01-01", "1999-12-31"),
            ("2000-2009  (two crashes)",   "2000-01-01", "2009-12-31"),
            ("2010-2019  (bull market)",   "2010-01-01", "2019-12-31"),
            ("2020-2024  (COVID + 2022)",  "2020-01-01", "2024-12-31"),
        ]
        print(f"{'═'*70}")
        print(f"  SUB-PERIOD BREAKDOWN: {ticker}")
        print(f"{'═'*70}")
        print(f"  {'Period':<28} {'Dyn CAGR':>9} {'B&H CAGR':>9} {'Dyn MaxDD':>10} {'B&H MaxDD':>10}")
        print(f"  {'─'*66}")
        for label, p_start, p_end in periods:
            mask       = (df.index    >= p_start) & (df.index    <= p_end)
            asset_mask = (asset.index >= p_start) & (asset.index <= p_end)
            if mask.sum() < 50:
                continue
            sub_df    = df.loc[mask]
            sub_asset = asset.loc[asset_mask]
            sub_bnh   = capital * sub_asset.values / sub_asset.values[0]
            dm = compute_metrics(sub_df["value"].values, sub_df.index)
            bm = compute_metrics(sub_bnh, sub_asset.index)
            beat = "+" if dm["CAGR"] > bm["CAGR"] else " "
            print(f"  {label:<28} {dm['CAGR']:>8.1f}% {bm['CAGR']:>8.1f}% {dm['MaxDD']:>9.1f}% {bm['MaxDD']:>9.1f}%  {beat}")
        print(f"{'═'*70}\n")

        plot(df, trades_df, votes_df, breakdown_df, asset, cfg, ticker)


if __name__ == "__main__":
    main()
