"""
Trade Decision Explainer  — Console Output
==========================================
Prints every trade decision as formatted tables directly to the terminal.

  - Date of trade
  - What each of the 8 signals voted (and why)
  - Total score and what it means
  - How big a position was taken and why

Usage:
    export FRED_API_KEY=your_key_here
    python trade_explainer.py
"""

import sys, json, gc, urllib.request, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

import os

# Free key from https://fred.stlouisfed.org/docs/api/api_key.html
# Set it before running:  export FRED_API_KEY=your_key_here
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
if not FRED_API_KEY:
    print("WARNING: FRED_API_KEY not set. Macro signals (ISM, credit, yield curve) will be neutral.")

# ── CONFIG (same optimized settings as optimized_strategy.py) ──
CONFIG = {
    "start_date":          "1993-01-01",
    "end_date":            "2024-12-31",
    "starting_capital":    100_000,
    "rebalance_days":      11,
    "n_simulations":       500,
    "transaction_cost":    0.0005,
    "vote_long":           3,
    "vote_cash":           -2,
    "max_leverage":        1.3,
    "min_leverage":        0.0,
    "leverage_cost":       0.0100,
    "vol_target":          0.36,
    "vol_lookback":        17,
    "base_position":       0.90,
    "conviction_boost":    0.30,
    "rebalance_threshold": 0.14,
    "leverage_min_score":  2,
}

SIGNAL_PARAMS = {
    "hy_z_bear":             0.7500,
    "hy_z_bull":             0.9000,
    "ism_z_bear":            0.3000,
    "ism_z_bull":            0.5500,
    "ma_bear_pct":           0.0150,
    "ma_bull_pct":           0.0050,
    "ma_window":             116,
    "mom_bear_ret":          0.1100,
    "mom_bull_ret":          0.1500,
    "mom_lookback":          159,
    "mom_skip":              10,
    "mom_size_sensitivity":  1.5000,
    "mom_size_window":       69,
    "sent_z_bear":           0.5000,
    "sent_z_bull":           0.2500,
    "vix_mc_bear_median":    29.5032,
    "vix_mc_bull_median":    19.7971,
    "vix_mc_prob_cutoff":    0.1500,
    "vix_mc_prob_threshold": 21.2327,
    "vix_size_neutral":      22.9876,
    "vix_size_slope":        0.0100,
    "vix_ts_bear":           1.5000,
    "vix_ts_bull":          -0.7000,
    "yc_bear":              -0.5000,
    "yc_bull":               1.7000,
}

SIGNAL_NAMES = {
    "v1": "VIX Monte Carlo",
    "v2": "VIX Term Structure",
    "v3": "Economic Activity",
    "v4": "Market Sentiment",
    "v5": "Moving Avg Trend",
    "v6": "Price Momentum",
    "v7": "Credit Spreads",
    "v8": "Yield Curve",
}

SIGNAL_REASONS = {
    "v1": { 1: "VIX sim projects calm — fear gauge staying low",
            0: "VIX sim is neutral",
           -1: "VIX sim projects rising fear / elevated vol ahead"},
    "v2": { 1: "Short VIX < Long VIX (normal) — no near-term fear",
            0: "VIX term structure is flat",
           -1: "Short VIX > Long VIX (inverted) — near-term fear spike"},
    "v3": { 1: "Industrial production above recent trend — expanding",
            0: "Industrial production near trend — neutral",
           -1: "Industrial production below recent trend — slowing"},
    "v4": { 1: "Small-caps (IWM) outperforming SPY — risk appetite healthy",
            0: "Small-cap vs large-cap performance neutral",
           -1: "Small-caps underperforming SPY — fleeing to safety"},
    "v5": { 1: "Price above moving average — uptrend",
            0: "Price near moving average — no clear trend",
           -1: "Price below moving average — downtrend"},
    "v6": { 1: "12-month momentum positive — market rising over past year",
            0: "12-month momentum flat — no directional momentum",
           -1: "12-month momentum negative — market falling over past year"},
    "v7": { 1: "HY credit spreads tight — companies borrowing cheap, risk-on",
            0: "Credit spreads normal — no stress",
           -1: "HY credit spreads wide — credit market stressed, risk-off"},
    "v8": { 1: "Yield curve steep (10Y >> 2Y) — healthy economic outlook",
            0: "Yield curve flat — no strong signal",
           -1: "Yield curve flat/inverted — recession warning"},
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
    asset = clean_yahoo(ticker, s, e)
    vix   = clean_yahoo("^VIX",   s, e)
    vix3m = clean_yahoo("^VIX3M", s, e)
    iwm   = clean_yahoo("IWM",    s, e)
    tlt   = clean_yahoo("TLT",    s, e)
    spy_s = clean_yahoo("SPY", s, e) if ticker != "SPY" else asset
    print("  Fetching FRED data...")
    indpro      = fetch_fred_series("INDPRO",       s, e)
    hy_spread   = fetch_fred_series("BAMLH0A0HYM2", s, e)
    dgs10       = fetch_fred_series("DGS10",        s, e)
    dgs2        = fetch_fred_series("DGS2",         s, e)
    yield_curve = dgs10 - dgs2
    common  = asset.index.intersection(vix.index)
    asset   = asset.loc[common]
    vix     = vix.loc[common]
    vix_ts  = vix - vix3m.reindex(common)
    tlt     = tlt.reindex(common)
    ism     = indpro.reindex(common).ffill()
    sent    = (iwm.reindex(common).pct_change(21) - spy_s.reindex(common).pct_change(21))
    hy_s    = hy_spread.reindex(common).ffill()
    yc      = yield_curve.reindex(common).ffill()
    print(f"  Aligned: {len(common)} days ({common[0].date()} -> {common[-1].date()})")
    return asset, vix, vix_ts, ism, sent, tlt, hy_s, yc


# ══════════════════════════════════════════════════════════════
# SIGNALS
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


def signal_vix_mc_detail(vix_hist, vix0, n_sims):
    sp     = SIGNAL_PARAMS
    window = vix_hist[-1260:] if len(vix_hist) > 1260 else vix_hist
    kappa, theta, sigma = calibrate_ou(window)
    if vix0 > 30:   kappa *= 1.5; sigma *= 1.4
    elif vix0 < 15: kappa *= 0.7; sigma *= 0.8
    dt = 1.0/252.0
    paths = np.zeros((n_sims, 8)); paths[:,0] = vix0
    for t in range(1, 8):
        z = np.random.standard_normal(n_sims)
        paths[:,t] = np.maximum(
            paths[:,t-1] + kappa*(theta-paths[:,t-1])*dt + sigma*np.sqrt(dt)*z, 5.0)
    final = paths[:,-1]
    med   = float(np.median(final))
    prob  = float(np.mean(final > sp["vix_mc_prob_threshold"]))
    if prob > sp["vix_mc_prob_cutoff"] or med > sp["vix_mc_bear_median"]: vote = -1
    elif med < sp["vix_mc_bull_median"]:                                   vote =  1
    else:                                                                   vote =  0
    return vote, f"now={vix0:.1f}  7d_median={med:.1f}  prob_spike={prob*100:.1f}%  mean_rev={theta:.1f}"


def signal_vix_ts_detail(ts_current):
    sp = SIGNAL_PARAMS
    if   ts_current > sp["vix_ts_bear"]: vote = -1
    elif ts_current < sp["vix_ts_bull"]: vote =  1
    else:                                vote =  0
    return vote, f"spread={ts_current:.2f}  bear>{sp['vix_ts_bear']}  bull<{sp['vix_ts_bull']}"


def signal_ism_detail(ism_hist, ism_current):
    sp   = SIGNAL_PARAMS
    hist = np.array(ism_hist, dtype=float).flatten()
    hist = hist[~np.isnan(hist)]
    if len(hist) < 60: return 0, "insufficient history"
    ref  = hist[-252:] if len(hist) >= 252 else hist
    mean = float(np.mean(ref)); std = float(np.std(ref))
    if std == 0: return 0, "std=0"
    z = (ism_current - mean) / std
    if   z >  sp["ism_z_bull"]: vote =  1
    elif z < -sp["ism_z_bear"]: vote = -1
    else:                        vote =  0
    return vote, f"z={z:.2f}  current={ism_current:.2f}  mean={mean:.2f}"


def signal_sentiment_detail(sent_hist, sent_current):
    sp   = SIGNAL_PARAMS
    hist = np.array(sent_hist, dtype=float).flatten()
    hist = hist[~np.isnan(hist)]
    if len(hist) < 60: return 0, "insufficient history"
    ref  = hist[-252:] if len(hist) >= 252 else hist
    mean = float(np.mean(ref)); std = float(np.std(ref))
    if std == 0: return 0, "std=0"
    z = (sent_current - mean) / std
    if   z >  sp["sent_z_bull"]: vote =  1
    elif z < -sp["sent_z_bear"]: vote = -1
    else:                         vote =  0
    return vote, f"z={z:.2f}  spread={sent_current*100:.2f}%  mean={mean*100:.2f}%"


def signal_ma_detail(prices_hist):
    sp     = SIGNAL_PARAMS
    prices = np.array(prices_hist, dtype=float).flatten()
    prices = prices[~np.isnan(prices)]
    win    = sp["ma_window"]
    if len(prices) < win: return 0, "insufficient history"
    ma_val = float(np.mean(prices[-win:]))
    pct    = (float(prices[-1]) - ma_val) / ma_val
    if   pct >  sp["ma_bull_pct"]: vote =  1
    elif pct < -sp["ma_bear_pct"]: vote = -1
    else:                           vote =  0
    return vote, f"price={prices[-1]:.2f}  ma{win}={ma_val:.2f}  dist={pct*100:+.2f}%"


def signal_momentum_detail(prices_hist):
    sp     = SIGNAL_PARAMS
    prices = np.array(prices_hist, dtype=float).flatten()
    prices = prices[~np.isnan(prices)]
    lb, sk = sp["mom_lookback"], sp["mom_skip"]
    if len(prices) < lb + sk: return 0, "insufficient history"
    ret = (float(prices[-sk]) / float(prices[-lb])) - 1.0
    if   ret >  sp["mom_bull_ret"]: vote =  1
    elif ret < -sp["mom_bear_ret"]: vote = -1
    else:                            vote =  0
    return vote, f"ret={ret*100:+.1f}%  lb={lb}d  skip={sk}d"


def signal_credit_detail(spread_hist, spread_now):
    sp   = SIGNAL_PARAMS
    hist = np.array(spread_hist, dtype=float).flatten()
    hist = hist[~np.isnan(hist)]
    if len(hist) < 60 or np.isnan(spread_now): return 0, "no data yet"
    ref  = hist[-252:] if len(hist) >= 252 else hist
    mean = float(np.mean(ref)); std = float(np.std(ref))
    if std == 0: return 0, "std=0"
    z = (spread_now - mean) / std
    if   z >  sp["hy_z_bear"]: vote = -1
    elif z < -sp["hy_z_bull"]: vote =  1
    else:                        vote =  0
    return vote, f"spread={spread_now:.2f}%  z={z:.2f}  mean={mean:.2f}%"


def signal_yc_detail(curve_now):
    sp = SIGNAL_PARAMS
    if np.isnan(curve_now): return 0, "no data yet"
    if   curve_now < sp["yc_bear"]: vote = -1
    elif curve_now > sp["yc_bull"]: vote =  1
    else:                            vote =  0
    return vote, f"10Y-2Y={curve_now:.2f}%  bear<{sp['yc_bear']}  bull>{sp['yc_bull']}"


def calculate_position_size(score, prices_hist, vix_current, cfg):
    sp = SIGNAL_PARAMS
    if score <= cfg["vote_cash"]:
        return 0.0, 0.0, 1.0, 1.0, 1.0

    extra    = max(score - cfg["vote_long"], 0)
    base_pos = cfg["base_position"] + extra * cfg["conviction_boost"]

    prices = np.array(prices_hist, dtype=float).flatten()
    prices = prices[~np.isnan(prices)]
    lb     = cfg["vol_lookback"]
    if len(prices) >= lb + 1:
        rets         = np.diff(np.log(prices[-lb-1:]))
        realized_vol = max(float(np.std(rets) * np.sqrt(252)), 0.05)
        vol_scalar   = np.clip(cfg["vol_target"] / realized_vol, 0.3, 1.8)
    else:
        realized_vol = 0.15; vol_scalar = 1.0

    vix_scalar = float(np.clip(
        1.0 - (vix_current - sp["vix_size_neutral"]) * sp["vix_size_slope"], 0.5, 1.1))

    w = sp["mom_size_window"]
    if len(prices) >= w:
        mom_pct         = (float(prices[-1]) - float(np.mean(prices[-w:]))) / float(np.mean(prices[-w:]))
        momentum_scalar = float(np.clip(1.0 + mom_pct * sp["mom_size_sensitivity"], 0.80, 1.20))
    else:
        momentum_scalar = 1.0

    raw   = base_pos * vol_scalar * vix_scalar * momentum_scalar
    final = float(np.clip(raw, cfg["min_leverage"], cfg["max_leverage"]))
    if score < cfg.get("leverage_min_score", 2):
        final = min(final, 1.0)
    return final, realized_vol, vol_scalar, vix_scalar, momentum_scalar


# ══════════════════════════════════════════════════════════════
# BACKTEST WITH FULL LOGGING
# ══════════════════════════════════════════════════════════════

def run_detailed_backtest(asset, vix, vix_ts, ism, sent, tlt, hy_spread, yield_curve, cfg):
    dates    = asset.index
    capital  = cfg["starting_capital"]
    tc       = cfg["transaction_cost"]
    n_sims   = cfg["n_simulations"]
    reb      = cfg["rebalance_days"]
    borrow_daily = cfg["leverage_cost"] / 252

    cash = float(capital)
    spy_shares = tlt_shares = current_size = 0.0
    pv = float(capital)
    pv_list   = []
    pos_list  = []
    decisions = []

    print(f"Running detailed backtest ({len(dates)} days)...")

    for i, date in enumerate(dates):
        price  = float(asset.iloc[i])
        tlt_px = float(tlt.iloc[i]) if not np.isnan(tlt.iloc[i]) else None

        if current_size > 1.0:
            cash -= (current_size - 1.0) * pv * borrow_daily

        if i >= 252 and i % reb == 0:
            v1, d1 = signal_vix_mc_detail(vix.values[:i], float(vix.iloc[i]), n_sims)
            v2, d2 = signal_vix_ts_detail(float(vix_ts.iloc[i]) if not np.isnan(vix_ts.iloc[i]) else 0.0)
            v3, d3 = signal_ism_detail(ism.values[:i], float(ism.iloc[i]) if not np.isnan(ism.iloc[i]) else 0.0)
            v4, d4 = signal_sentiment_detail(sent.values[:i], float(sent.iloc[i]) if not np.isnan(sent.iloc[i]) else 0.0)
            v5, d5 = signal_ma_detail(asset.values[:i])
            v6, d6 = signal_momentum_detail(asset.values[:i])
            v7, d7 = signal_credit_detail(hy_spread.values[:i],
                         float(hy_spread.iloc[i]) if not np.isnan(hy_spread.iloc[i]) else float('nan'))
            v8, d8 = signal_yc_detail(float(yield_curve.iloc[i]) if not np.isnan(yield_curve.iloc[i]) else float('nan'))

            votes = [v1, v2, v3, v4, v5, v6, v7, v8]
            score = sum(votes)

            target, realized_vol, vol_s, vix_s, mom_s = calculate_position_size(
                score, asset.values[:i], float(vix.iloc[i]), cfg)

            thresh  = cfg["rebalance_threshold"]
            t_spy   = pv * target
            c_spy   = spy_shares * price
            traded  = abs(t_spy - c_spy) / (pv + 1e-10) > thresh

            action = "HOLD"
            if traded:
                diff = t_spy - c_spy
                if diff > 0:
                    spy_shares += diff / price; cash -= diff * (1 + tc)
                    action = "BUY" if current_size == 0 else "ADD"
                else:
                    spy_shares += diff / price; cash -= diff * (1 - tc)
                    action = "SELL" if target == 0 else "REDUCE"
                current_size = target

            t_tlt = pv * (1.0 if (target == 0 and tlt_px) else 0.0)
            c_tlt = tlt_shares * tlt_px if tlt_px else 0.0
            if abs(t_tlt - c_tlt) / (pv + 1e-10) > thresh and tlt_px:
                diff = t_tlt - c_tlt
                tlt_shares += diff / tlt_px; cash -= diff * (1 + tc if diff > 0 else 1 - tc)

            decisions.append({
                "date":         date,
                "price":        price,
                "vix":          float(vix.iloc[i]),
                "score":        score,
                "votes":        {"v1":v1,"v2":v2,"v3":v3,"v4":v4,"v5":v5,"v6":v6,"v7":v7,"v8":v8},
                "details":      {"v1":d1,"v2":d2,"v3":d3,"v4":d4,"v5":d5,"v6":d6,"v7":d7,"v8":d8},
                "target_size":  target,
                "old_size":     current_size,
                "action":       action,
                "traded":       traded,
                "portfolio_value": pv,
                "realized_vol": realized_vol,
                "vol_scalar":   vol_s,
                "vix_scalar":   vix_s,
                "mom_scalar":   mom_s,
            })

        tlt_val = tlt_shares * tlt_px if tlt_px else 0.0
        pv = cash + spy_shares * price + tlt_val
        pv_list.append(pv)
        pos_list.append(current_size)

    return np.array(pv_list), np.array(pos_list), decisions


# ══════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ══════════════════════════════════════════════════════════════

VOTE_LABEL = { 1: "+1 BULL",  0: " 0 NEUT", -1: "-1 BEAR"}

def print_report(decisions, pv_list, pos_list, asset, cfg, ticker, show_holds=False):
    capital  = cfg["starting_capital"]
    bnh_v    = capital * asset.values / asset.values[0]
    final_pv = pv_list[-1]
    final_bnh = bnh_v[-1]

    all_d    = decisions
    trades   = [d for d in decisions if d["traded"]]
    W = 78  # table width

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "═"*W)
    print(f"  TRADE DECISION REPORT  —  {ticker}  |  {cfg['start_date']} → {cfg['end_date']}")
    print("═"*W)
    print(f"  {'Starting capital:':<28} ${capital:>12,.0f}")
    print(f"  {'Strategy final value:':<28} ${final_pv:>12,.0f}   ({(final_pv/capital - 1)*100:+.1f}%)")
    print(f"  {'Buy & hold final value:':<28} ${final_bnh:>12,.0f}   ({(final_bnh/capital - 1)*100:+.1f}%)")
    print(f"  {'Total rebalance checks:':<28} {len(all_d):>12}")
    print(f"  {'Actual trades (buys/sells):':<28} {len(trades):>12}")
    print(f"  {'Max leverage:':<28} {cfg['max_leverage']:>11.1f}x")
    print(f"  {'Vote threshold (long):':<28} {cfg['vote_long']:>12}")
    print(f"  {'Vote threshold (cash):':<28} {cfg['vote_cash']:>12}")
    print("═"*W)

    # ── All-trades summary table ──────────────────────────────
    print(f"\n  {'#':<5} {'Date':<13} {'Price':>8} {'VIX':>6} {'Score':>7} {'Action':<8} {'Size':>6} {'Portfolio':>12}")
    print("  " + "─"*(W-2))
    for n, d in enumerate(trades, 1):
        sc = d["score"]
        sc_str = f"{'+' if sc>0 else ''}{sc}"
        print(f"  {n:<5} {d['date'].strftime('%Y-%m-%d'):<13} "
              f"${d['price']:>7,.2f} {d['vix']:>6.1f} "
              f"{sc_str:>7} {d['action']:<8} "
              f"{d['target_size']:>5.2f}x ${d['portfolio_value']:>11,.0f}")
    print("  " + "─"*(W-2))

    # ── Detailed card per trade ───────────────────────────────
    show_list = [d for d in decisions if d["traded"] or show_holds]

    print(f"\n\n{'═'*W}")
    print(f"  DETAILED BREAKDOWN  ({len(show_list)} trades)")
    print("═"*W)

    for n, d in enumerate(show_list, 1):
        sc     = d["score"]
        sc_str = f"{'+' if sc>0 else ''}{sc}"
        votes  = d["votes"]
        details = d["details"]
        bulls  = sum(1 for v in votes.values() if v ==  1)
        bears  = sum(1 for v in votes.values() if v == -1)
        neuts  = sum(1 for v in votes.values() if v ==  0)

        # Regime label
        if sc >= cfg["vote_long"]:   regime = "BULL"
        elif sc <= cfg["vote_cash"]: regime = "BEAR/CASH"
        else:                         regime = "NEUTRAL"

        print(f"\n  ── Trade #{n}  {d['date'].strftime('%Y-%m-%d')}  |  {ticker} @ ${d['price']:,.2f}  |  VIX: {d['vix']:.1f}")
        print(f"     Score: {sc_str}/8  |  Regime: {regime}  |  Action: {d['action']}  |  Size: {d['target_size']:.2f}x  |  Portfolio: ${d['portfolio_value']:,.0f}")
        print(f"     Signals: {bulls} bull  {bears} bear  {neuts} neutral")
        print()

        # Signal table
        col_s = 22   # signal name width
        col_v = 9    # vote width
        col_r = W - col_s - col_v - 8  # reason width
        hdr = f"  {'Signal':<{col_s}}{'Vote':<{col_v}}{'Reason / Data'}"
        print(hdr)
        print("  " + "─"*(W-2))
        for key in ["v1","v2","v3","v4","v5","v6","v7","v8"]:
            v      = votes[key]
            name   = SIGNAL_NAMES[key]
            reason = SIGNAL_REASONS[key][v]
            data   = details[key]
            print(f"  {name:<{col_s}}{VOTE_LABEL[v]:<{col_v}}{reason}")
            if data:
                print(f"  {'':<{col_s}}{'':<{col_v}}  → {data}")
        print()

        # Sizing table
        base_pos = cfg["base_position"] + max(sc - cfg["vote_long"], 0) * cfg["conviction_boost"]
        vol_lb_label  = f"Realized vol (trailing {cfg['vol_lookback']}d):"
        vix_lbl       = f"VIX scalar (VIX={d['vix']:.1f}):"
        raw           = base_pos * d["vol_scalar"] * d["vix_scalar"] * d["mom_scalar"]
        raw_lbl       = f"Raw size  ({base_pos:.2f}x{d['vol_scalar']:.2f}x{d['vix_scalar']:.2f}x{d['mom_scalar']:.2f}):"
        cap_lbl       = f"Final size (capped at {cfg['max_leverage']}x):"
        print(f"  {'POSITION SIZING':─<{W-2}}")
        print(f"  {'Base position (score conviction):':<40} {base_pos:.3f}x")
        print(f"  {vol_lb_label:<40} {d['realized_vol']*100:.1f}%  -> vol scalar: {d['vol_scalar']:.3f}x")
        print(f"  {vix_lbl:<40} {d['vix_scalar']:.3f}x")
        print(f"  {'Momentum scalar:':<40} {d['mom_scalar']:.3f}x")
        print(f"  {raw_lbl:<40} {raw:.3f}x")
        print(f"  {cap_lbl:<40} {d['target_size']:.3f}x")
        print()

    print("═"*W)
    print(f"  End of report — {len(show_list)} trades shown for {ticker}")
    print("═"*W + "\n")


# ══════════════════════════════════════════════════════════════
# TRADE CHART
# ══════════════════════════════════════════════════════════════

def plot_trades(decisions, pv_list, pos_list, asset, cfg, ticker):
    trades = [d for d in decisions if d["traded"]]

    dates   = asset.index
    prices  = asset.values
    capital = cfg["starting_capital"]
    bnh_v   = capital * prices / prices[0]

    # --- separate trades by action
    buys    = [d for d in trades if d["action"] in ("BUY",  "ADD")]
    sells   = [d for d in trades if d["action"] in ("SELL", "REDUCE")]

    fig = plt.figure(figsize=(20, 14), facecolor="#0d1117")
    fig.suptitle(f"Trade Entry / Exit Map  —  {ticker}",
                 fontsize=17, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.08,
                           left=0.06, right=0.97, top=0.95, bottom=0.05,
                           height_ratios=[3, 1, 1])

    ax_price = fig.add_subplot(gs[0])
    ax_pos   = fig.add_subplot(gs[1], sharex=ax_price)
    ax_score = fig.add_subplot(gs[2], sharex=ax_price)

    # ── Price + trade markers ──────────────────────────────────
    ax_price.set_facecolor("#161b22")
    ax_price.plot(dates, prices, color="#4fc3f7", linewidth=1.4,
                  label=f"{ticker} price", zorder=3)

    # BUY / ADD  →  green triangle up
    if buys:
        bx = [d["date"]  for d in buys]
        by = [d["price"] for d in buys]
        ax_price.scatter(bx, by, marker="^", color="#2ecc71", s=80,
                         zorder=5, label="BUY / ADD")

    # SELL / REDUCE  →  red triangle down
    if sells:
        sx = [d["date"]  for d in sells]
        sy = [d["price"] for d in sells]
        ax_price.scatter(sx, sy, marker="v", color="#e74c3c", s=80,
                         zorder=5, label="SELL / REDUCE")

    ax_price.set_ylabel("Price ($)", color="#8b9dc3", fontsize=10)
    ax_price.tick_params(colors="#8b9dc3", labelbottom=False)
    for sp in ax_price.spines.values(): sp.set_color("#30363d")
    ax_price.legend(loc="upper left", fontsize=9, facecolor="#161b22",
                    labelcolor="white", edgecolor="#30363d")
    ax_price.set_title(f"Price with trade markers  |  {len(buys)} buys  {len(sells)} sells",
                       color="#8b9dc3", fontsize=10, pad=4)

    # ── Position size over time ────────────────────────────────
    ax_pos.set_facecolor("#161b22")
    pos_arr = np.array(pos_list)
    ax_pos.fill_between(dates, pos_arr, 0,
                         where=pos_arr > 1.0, alpha=0.7, color="#ffd700", label="Leveraged")
    ax_pos.fill_between(dates, pos_arr, 0,
                         where=(pos_arr > 0) & (pos_arr <= 1.0),
                         alpha=0.5, color="#4fc3f7", label="Invested")
    ax_pos.fill_between(dates, pos_arr, 0,
                         where=pos_arr == 0, alpha=0.5, color="#e74c3c", label="Cash")
    ax_pos.plot(dates, pos_arr, color="white", linewidth=0.6, alpha=0.5)
    ax_pos.axhline(1.0, color="#555", linewidth=0.8, linestyle="--")
    ax_pos.set_ylabel("Position", color="#8b9dc3", fontsize=9)
    ax_pos.set_ylim(-0.05, cfg["max_leverage"] + 0.1)
    ax_pos.tick_params(colors="#8b9dc3", labelbottom=False)
    for sp in ax_pos.spines.values(): sp.set_color("#30363d")
    ax_pos.legend(loc="upper right", fontsize=8, facecolor="#161b22",
                  labelcolor="white", edgecolor="#30363d", ncol=3)

    # ── Vote score over time ───────────────────────────────────
    ax_score.set_facecolor("#161b22")
    score_dates = [d["date"]  for d in decisions]
    score_vals  = [d["score"] for d in decisions]
    sv = np.array(score_vals)
    sd = score_dates
    ax_score.fill_between(sd, sv, 0, where=sv >= 0, alpha=0.6, color="#34d399")
    ax_score.fill_between(sd, sv, 0, where=sv <  0, alpha=0.6, color="#e74c3c")
    ax_score.plot(sd, sv, color="white", linewidth=0.5, alpha=0.3)
    ax_score.axhline(cfg["vote_long"], color="#34d399", linewidth=1.0,
                      linestyle="--", alpha=0.7, label=f"Long threshold ({cfg['vote_long']})")
    ax_score.axhline(cfg["vote_cash"], color="#e74c3c", linewidth=1.0,
                      linestyle="--", alpha=0.7, label=f"Cash threshold ({cfg['vote_cash']})")
    ax_score.axhline(0, color="#555", linewidth=0.6)
    ax_score.set_ylabel("Score", color="#8b9dc3", fontsize=9)
    ax_score.set_yticks([-6,-4,-2,0,2,4,6])
    ax_score.tick_params(colors="#8b9dc3")
    for sp in ax_score.spines.values(): sp.set_color("#30363d")
    ax_score.legend(loc="upper right", fontsize=8, facecolor="#161b22",
                    labelcolor="white", edgecolor="#30363d", ncol=2)

    plt.setp(ax_price.get_xticklabels(), visible=False)
    plt.setp(ax_pos.get_xticklabels(),   visible=False)

    out_file = f"trade_chart_{ticker}.png"
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

    # Change this list to run other tickers
    for ticker in ["SPY"]:
        print(f"\n{'='*60}")
        print(f"  Trade Explainer: {ticker}")
        print(f"{'='*60}")

        cfg = dict(CONFIG)
        cfg["start_date"] = ASSET_STARTS[ticker]

        asset, vix, vix_ts, ism, sent, tlt, hy_spread, yield_curve = fetch_all(cfg, ticker)

        pv_list, pos_list, decisions = run_detailed_backtest(
            asset, vix, vix_ts, ism, sent, tlt, hy_spread, yield_curve, cfg)

        # show_holds=True to also print HOLD decisions (where no trade was made)
        print_report(decisions, pv_list, pos_list, asset, cfg, ticker, show_holds=False)

        # Trade chart: price + buy/sell markers + position size + vote score
        plot_trades(decisions, pv_list, pos_list, asset, cfg, ticker)


if __name__ == "__main__":
    main()
