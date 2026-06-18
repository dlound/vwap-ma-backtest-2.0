"""
╔══════════════════════════════════════════════════════════════════╗
║    VWAP + Moving Average Trend-Following Intraday Backtester    ║
║    3 Independent Systems: 5 MA | 20 MA | Combined              ║
╚══════════════════════════════════════════════════════════════════╝

  ┌───────────────────────────────────────────────────────────┐
  │  CONFIGURATION  —  edit these to customise your run       │
  └───────────────────────────────────────────────────────────┘
"""

TICKER          = "SPY"        # Any ticker available on yfinance
START_DATE      = "2026-04-20" # YYYY-MM-DD
END_DATE        = "2026-06-16" # YYYY-MM-DD

ACCOUNT_BALANCE = 10_000       # Starting balance per system
FIXED_NOTIONAL  = 100          # Fixed $ per trade (set to None to use full account)

# Moving average periods to test as independent systems
MA_PERIODS      = [5, 20]

# Stop loss: lowest/highest of previous N completed candles
SL_LOOKBACK     = 5

OUTPUT_DIR      = "."

"""
  ┌───────────────────────────────────────────────────────────┐
  │  No edits needed below this line                          │
  └───────────────────────────────────────────────────────────┘
"""

import sys, warnings
warnings.filterwarnings("ignore")

def _ensure(pkg, import_name=None):
    import importlib, subprocess
    try:
        importlib.import_module(import_name or pkg)
    except ModuleNotFoundError:
        print(f"  Installing {pkg} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                               "--quiet", "--break-system-packages"])

for p, a in [("yfinance","yfinance"),("pandas","pandas"),
             ("numpy","numpy"),("matplotlib","matplotlib")]:
    _ensure(p, a)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import time as dtime
import yfinance as yf

MARKET_OPEN  = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
COLORS       = ["#3498db", "#e67e22", "#9b59b6", "#27ae60",
                "#e74c3c", "#1abc9c"]

# ── data ─────────────────────────────────────────────────────────────────────
def download_data(ticker, start, end):
    print(f"\n  Downloading 5-min data for {ticker} ({start} → {end}) …")
    raw = yf.download(ticker, start=start, end=end,
                      interval="5m", auto_adjust=True, progress=False)
    if raw.empty:
        sys.exit(f"  ERROR: No data for {ticker}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index)
    if raw.index.tzinfo is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert("America/New_York")
    raw = raw.between_time(MARKET_OPEN, MARKET_CLOSE)
    raw = raw.dropna(subset=["Open","High","Low","Close"])
    raw = raw[raw["Volume"] > 0]
    n   = raw.index.normalize().nunique()
    print(f"  → {len(raw):,} 5-min bars across {n} trading days.")
    return raw


def add_indicators(df):
    """Add daily-reset VWAP and simple MAs for each period in MA_PERIODS."""
    df = df.copy()
    df["_date"] = df.index.date
    df["_tp"]   = (df["High"] + df["Low"] + df["Close"]) / 3
    df["_tpv"]  = df["_tp"] * df["Volume"]
    df["_ctpv"] = df.groupby("_date")["_tpv"].cumsum()
    df["_cvol"] = df.groupby("_date")["Volume"].cumsum()
    df["VWAP"]  = df["_ctpv"] / df["_cvol"]
    df.drop(columns=["_date","_tp","_tpv","_ctpv","_cvol"], inplace=True)
    for p in MA_PERIODS:
        df[f"MA{p}"] = df["Close"].rolling(p).mean()
    return df


# ── core backtest engine ──────────────────────────────────────────────────────
def run_system(df, ma_period, tp_version):
    """
    One independent system for a single MA period and TP version.

    tp_version : "swing"  → exit when price returns to pre-pullback swing high/low
                 "eod"    → exit at 4:00 PM candle close

    Returns (trades, final_balance)
    """
    ma_col  = f"MA{ma_period}"
    account = ACCOUNT_BALANCE
    trades  = []

    # We need per-candle array access — work with numpy for speed
    bars = df.reset_index()
    n    = len(bars)

    # Pre-extract arrays
    times   = bars["Datetime"].values if "Datetime" in bars.columns \
              else bars.iloc[:, 0].values
    opens   = bars["Open"].values
    highs   = bars["High"].values
    lows    = bars["Low"].values
    closes  = bars["Close"].values
    vwaps   = bars["VWAP"].values
    mas     = bars[ma_col].values
    dates   = np.array([pd.Timestamp(t).date() for t in times])

    i = max(ma_period, SL_LOOKBACK) + 1   # start after enough history

    while i < n - 1:
        # ── skip if MA not yet valid ──────────────────────────────────────────
        if np.isnan(mas[i]):
            i += 1; continue

        # ── market bias ───────────────────────────────────────────────────────
        close_i = closes[i]
        vwap_i  = vwaps[i]
        ma_i    = mas[i]
        ma_prev = mas[i - 1] if i > 0 else np.nan

        if np.isnan(ma_prev):
            i += 1; continue

        above_vwap = close_i > vwap_i
        below_vwap = close_i < vwap_i

        # ── detect pullback touch ─────────────────────────────────────────────
        # Long: candle low touches or crosses below MA, then closes ABOVE MA
        #       AND close > prev close
        long_touch  = (above_vwap and
                       lows[i] <= ma_i and
                       close_i > ma_i and
                       close_i > closes[i - 1])

        # Short: candle high touches or crosses above MA, then closes BELOW MA
        #        AND close < prev close
        short_touch = (below_vwap and
                       highs[i] >= ma_i and
                       close_i < ma_i and
                       close_i < closes[i - 1])

        if not long_touch and not short_touch:
            i += 1; continue

        direction = "long" if long_touch else "short"

        # ── entry at open of NEXT candle ──────────────────────────────────────
        entry_idx = i + 1
        if entry_idx >= n:
            i += 1; continue
        if dates[entry_idx] != dates[i]:     # don't carry over to next day
            i += 1; continue

        entry_price = opens[entry_idx]
        entry_ts    = times[entry_idx]

        # ── stop loss: prev SL_LOOKBACK completed candles ─────────────────────
        sl_window = slice(max(0, i - SL_LOOKBACK), i)
        if direction == "long":
            stop_price = lows[sl_window].min()
        else:
            stop_price = highs[sl_window].max()

        risk = abs(entry_price - stop_price)
        if risk <= 0:
            i += 1; continue

        # ── take profit (Version 1: highest/lowest high across pullback candles) ─
        # Pullback = the consecutive sequence of candles ending at bar i where
        # each candle closes lower than the previous (long) or higher (short).
        # TP = highest high across ALL those pullback candles (long)
        #    = lowest  low  across ALL those pullback candles (short)
        tp_price = None
        if tp_version == "swing":
            if direction == "long":
                # walk backward while each close is below the prior close
                j = i
                while j > 1 and dates[j] == dates[i] and closes[j] < closes[j - 1]:
                    j -= 1
                # pullback spans bars j..i (inclusive)
                pb_window = slice(j, i + 1)
                tp_price  = highs[pb_window].max()
                # must be above entry price to be a valid target
                if tp_price <= entry_price:
                    tp_price = None
            else:
                # walk backward while each close is above the prior close
                j = i
                while j > 1 and dates[j] == dates[i] and closes[j] > closes[j - 1]:
                    j -= 1
                pb_window = slice(j, i + 1)
                tp_price  = lows[pb_window].min()
                if tp_price >= entry_price:
                    tp_price = None

        # ── simulate trade forward ────────────────────────────────────────────
        exit_price = exit_ts = exit_reason = None
        today      = dates[entry_idx]

        for j in range(entry_idx + 1, n):
            if dates[j] != today:           # crossed into next day → EOD exit
                # exit at last bar of today
                last_today = j - 1
                exit_price  = closes[last_today]
                exit_ts     = times[last_today]
                exit_reason = "eod"
                break

            is_last = (j == n - 1) or (dates[j + 1] != today) or \
                      (pd.Timestamp(times[j]).time() >= MARKET_CLOSE)

            if direction == "long":
                sl_hit = lows[j]  <= stop_price
                tp_hit = (tp_price is not None) and highs[j] >= tp_price
            else:
                sl_hit = highs[j] >= stop_price
                tp_hit = (tp_price is not None) and lows[j]  <= tp_price

            # intrabar conflict → stop first
            if sl_hit and tp_hit:
                exit_price, exit_reason = stop_price, "stop"
            elif sl_hit:
                exit_price, exit_reason = stop_price, "stop"
            elif tp_hit:
                exit_price, exit_reason = tp_price,   "target"
            elif is_last or tp_version == "eod":
                if is_last:
                    exit_price, exit_reason = closes[j], "eod"

            if exit_price is not None:
                exit_ts = times[j]
                break

        if exit_price is None or exit_ts is None:
            i = entry_idx + 1; continue

        # ── P&L ───────────────────────────────────────────────────────────────
        notional = FIXED_NOTIONAL if FIXED_NOTIONAL is not None else account
        ret      = ((exit_price - entry_price) / entry_price
                    if direction == "long"
                    else (entry_price - exit_price) / entry_price)
        pnl      = notional * ret
        account += pnl
        duration = (pd.Timestamp(exit_ts) - pd.Timestamp(entry_ts)
                    ).total_seconds() / 60

        trades.append({
            "date":          str(dates[entry_idx]),
            "entry_time":    str(pd.Timestamp(entry_ts)),
            "exit_time":     str(pd.Timestamp(exit_ts)),
            "direction":     direction,
            "ma_period":     ma_period,
            "tp_version":    tp_version,
            "entry_price":   round(entry_price,  4),
            "stop_price":    round(stop_price,   4),
            "tp_price":      round(tp_price, 4) if tp_price else None,
            "exit_price":    round(exit_price,   4),
            "notional":      round(notional,     2),
            "pnl":           round(pnl,          4),
            "return_pct":    round(ret * 100,    4),
            "account_after": round(account,      2),
            "exit_reason":   exit_reason,
            "duration_min":  round(duration,     1),
            "risk":          round(risk,         4),
            "r_multiple":    round(pnl / (notional * risk / entry_price), 4)
                             if risk > 0 else 0,
            "year":          pd.Timestamp(entry_ts).year,
        })

        # resume scan from bar AFTER the exit bar (don't re-enter on same bar)
        exit_bar_idx = np.searchsorted(
            [pd.Timestamp(t) for t in times],
            pd.Timestamp(exit_ts)
        )
        i = max(exit_bar_idx + 1, entry_idx + 1)

    return trades, account


# ── statistics ────────────────────────────────────────────────────────────────
def compute_stats(trades, final_balance, label=""):
    if not trades:
        return {"label": label, "total_trades": 0}

    pnls   = np.array([t["pnl"] for t in trades])
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    longs  = [t for t in trades if t["direction"] == "long"]
    shorts = [t for t in trades if t["direction"] == "short"]
    gp     = wins.sum()        if len(wins)   else 0.0
    gl     = abs(losses.sum()) if len(losses) else 0.0

    acct   = np.array([ACCOUNT_BALANCE] + [t["account_after"] for t in trades])
    peak   = np.maximum.accumulate(acct)
    dd     = acct - peak
    max_dd = dd.min()
    dd_pct = (max_dd / peak[np.argmin(dd)]) * 100

    def wr(sub):
        return round(sum(1 for t in sub if t["pnl"] > 0) / len(sub) * 100, 1) \
               if sub else 0.0

    notional = FIXED_NOTIONAL if FIXED_NOTIONAL else ACCOUNT_BALANCE

    return {
        "label":            label,
        "total_trades":     len(trades),
        "wins":             len(wins),
        "losses":           len(losses),
        "win_rate":         round(len(wins) / len(trades) * 100, 1),
        "avg_winner":       round(wins.mean(),   2) if len(wins)   else 0.0,
        "avg_loser":        round(losses.mean(), 2) if len(losses) else 0.0,
        "total_pnl":        round(pnls.sum(),    2),
        "total_return_pct": round((final_balance - ACCOUNT_BALANCE)
                                  / ACCOUNT_BALANCE * 100, 2),
        "final_balance":    round(final_balance, 2),
        "profit_factor":    round(gp / gl if gl else float("inf"), 3),
        "max_drawdown":     round(max_dd, 2),
        "max_dd_pct":       round(dd_pct, 2),
        "avg_duration":     round(np.mean([t["duration_min"] for t in trades]), 1),
        "avg_r":            round(np.mean([t["r_multiple"] for t in trades]), 3),
        "long_trades":      len(longs),  "short_trades": len(shorts),
        "long_wr":          wr(longs),   "short_wr":     wr(shorts),
        "long_pnl":         round(sum(t["pnl"] for t in longs),  2),
        "short_pnl":        round(sum(t["pnl"] for t in shorts), 2),
        "stop_exits":       sum(1 for t in trades if t["exit_reason"] == "stop"),
        "target_exits":     sum(1 for t in trades if t["exit_reason"] == "target"),
        "eod_exits":        sum(1 for t in trades if t["exit_reason"] == "eod"),
    }


def print_stats(s):
    if s["total_trades"] == 0:
        print("    No trades generated."); return
    print(f"    Trades         : {s['total_trades']}  "
          f"({s['wins']}W / {s['losses']}L)")
    print(f"    Win rate       : {s['win_rate']}%")
    print(f"    Avg winner     : ${s['avg_winner']:,.2f}")
    print(f"    Avg loser      : ${s['avg_loser']:,.2f}")
    print(f"    Profit factor  : {s['profit_factor']:.3f}")
    print(f"    Total P&L      : ${s['total_pnl']:,.2f}")
    print(f"    Total return   : {s['total_return_pct']:+.2f}%")
    print(f"    Final balance  : ${s['final_balance']:,.2f}")
    print(f"    Max drawdown   : ${s['max_drawdown']:,.2f}  "
          f"({s['max_dd_pct']:+.2f}%)")
    print(f"    Avg duration   : {s['avg_duration']} min")
    print(f"    Avg R multiple : {s['avg_r']:.3f}R")
    print(f"    Exits → Stop: {s['stop_exits']}  "
          f"Target: {s['target_exits']}  EOD: {s['eod_exits']}")
    print(f"    Long  : {s['long_trades']} trades | "
          f"WR {s['long_wr']}% | P&L ${s['long_pnl']:,.2f}")
    print(f"    Short : {s['short_trades']} trades | "
          f"WR {s['short_wr']}% | P&L ${s['short_pnl']:,.2f}")


def yearly_breakdown(trades):
    if not trades: return pd.DataFrame()
    df2 = pd.DataFrame(trades)
    def agg(g):
        p  = g["pnl"].values
        w  = (p > 0).sum()
        gl = abs(p[p < 0].sum())
        sb = g["account_after"].iloc[0] - g["pnl"].iloc[0]  # balance before first trade
        eb = g["account_after"].iloc[-1]
        return pd.Series({
            "trades":        len(g),
            "win_rate":      round(w / len(g) * 100, 1),
            "total_pnl":     round(p.sum(), 2),
            "year_return_%": round((eb - sb) / sb * 100, 2) if sb else 0.0,
            "profit_factor": round(p[p>0].sum() / gl if gl else float("inf"), 3),
        })
    return df2.groupby("year").apply(agg, include_groups=False)


# ── charting ──────────────────────────────────────────────────────────────────
def plot_results(all_results, out_dir):
    """
    all_results: dict keyed by label →
        {"trades": [...], "final_balance": float, "stats": dict}
    """
    labels = list(all_results.keys())
    n      = len(labels)

    # layout: one equity curve per system + 1 combined bar chart
    cols = 2
    rows = (n + 1) // cols + 1
    fig  = plt.figure(figsize=(16, 5 * rows))
    fig.patch.set_facecolor("#1a1a2e")
    gs   = gridspec.GridSpec(rows, cols, figure=fig,
                             hspace=0.5, wspace=0.32,
                             top=0.92, bottom=0.05,
                             left=0.07, right=0.96)

    axes = []
    for r in range(rows - 1):
        for c in range(cols):
            axes.append(fig.add_subplot(gs[r, c]))

    for idx, (label, res) in enumerate(all_results.items()):
        if idx >= len(axes): break
        ax     = axes[idx]
        ax.set_facecolor("#0f0f23")
        col    = COLORS[idx % len(COLORS)]
        trades = res["trades"]
        s      = res["stats"]

        if not trades:
            ax.text(0.5, 0.5, "No trades", ha="center", va="center",
                    color="gray", transform=ax.transAxes)
            ax.set_title(label, color=col, fontsize=9, fontweight="bold")
            continue

        acct = np.array([ACCOUNT_BALANCE] +
                        [t["account_after"] for t in trades])
        peak = np.maximum.accumulate(acct)
        dd   = acct - peak

        ax.fill_between(range(len(acct)), acct, peak, where=(dd < 0),
                        color="#e74c3c", alpha=0.25, label="Drawdown")
        ax.plot(acct, color=col, linewidth=1.6, label="Balance")
        ax.axhline(ACCOUNT_BALANCE, color="#888", linewidth=0.8,
                   linestyle="--", label="Start")

        notional_label = (f"${FIXED_NOTIONAL}/trade"
                          if FIXED_NOTIONAL else "Full acct")
        info = (f"T:{s['total_trades']}  WR:{s['win_rate']}%  "
                f"PF:{s['profit_factor']}  "
                f"Ret:{s['total_return_pct']:+.1f}%  "
                f"Final:${s['final_balance']:,.0f}  [{notional_label}]")

        ax.set_title(label, color=col, fontsize=9, fontweight="bold")
        ax.set_xlabel("Trade #", fontsize=7, color="#aaa")
        ax.set_ylabel("Account Balance ($)", fontsize=7, color="#aaa")
        ax.tick_params(labelsize=6, colors="#aaa")
        ax.spines[:].set_color("#333")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.text(0.02, 0.04, info, transform=ax.transAxes,
                fontsize=6.0, color="#eee",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#1e1e3a",
                          alpha=0.9, edgecolor="#555"))
        ax.legend(fontsize=6, loc="upper left",
                  facecolor="#1a1a2e", edgecolor="#555", labelcolor="#ccc")

    # ── bottom row: comparison bar charts ────────────────────────────────────
    ax_left  = fig.add_subplot(gs[rows - 1, 0])
    ax_right = fig.add_subplot(gs[rows - 1, 1])

    for ax in [ax_left, ax_right]:
        ax.set_facecolor("#0f0f23")
        ax.tick_params(labelsize=6, colors="#aaa")
        ax.spines[:].set_color("#333")

    finals = [res["stats"].get("final_balance", ACCOUNT_BALANCE)
              for res in all_results.values()]
    wrs    = [res["stats"].get("win_rate", 0) for res in all_results.values()]
    short_labels = [l.replace("Version","V").replace("MA ","MA")
                    for l in labels]

    bar_colors = ["#27ae60" if v >= ACCOUNT_BALANCE else "#e74c3c"
                  for v in finals]
    b1 = ax_left.bar(range(n), finals, color=bar_colors, alpha=0.85,
                     edgecolor="#555", linewidth=0.8)
    ax_left.axhline(ACCOUNT_BALANCE, color="#aaa", linewidth=1.0,
                    linestyle="--")
    ax_left.set_xticks(range(n))
    ax_left.set_xticklabels(short_labels, fontsize=6, color="#ccc",
                             rotation=15, ha="right")
    ax_left.set_title("Final Balance by System", color="#fff",
                      fontsize=9, fontweight="bold")
    ax_left.set_ylabel("Balance ($)", fontsize=7, color="#aaa")
    ax_left.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    max_f = max(finals) if finals else ACCOUNT_BALANCE
    for bar, val in zip(b1, finals):
        pct = (val - ACCOUNT_BALANCE) / ACCOUNT_BALANCE * 100
        ax_left.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max_f * 0.01,
                     f"${val:,.0f}\n({pct:+.1f}%)",
                     ha="center", va="bottom", fontsize=5.5,
                     color="#eee", linespacing=1.4)

    wr_colors = ["#27ae60" if w >= 50 else "#e74c3c" for w in wrs]
    b2 = ax_right.bar(range(n), wrs, color=wr_colors, alpha=0.85,
                      edgecolor="#555", linewidth=0.8)
    ax_right.axhline(50, color="#aaa", linewidth=1.0, linestyle="--")
    ax_right.set_xticks(range(n))
    ax_right.set_xticklabels(short_labels, fontsize=6, color="#ccc",
                              rotation=15, ha="right")
    ax_right.set_title("Win Rate by System", color="#fff",
                       fontsize=9, fontweight="bold")
    ax_right.set_ylabel("Win Rate (%)", fontsize=7, color="#aaa")
    for bar, val in zip(b2, wrs):
        ax_right.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.5,
                      f"{val:.1f}%", ha="center", va="bottom",
                      fontsize=6, color="#eee")

    notional_str = (f"${FIXED_NOTIONAL} fixed/trade"
                    if FIXED_NOTIONAL
                    else f"Full ${ACCOUNT_BALANCE:,} account")
    fig.suptitle(
        f"VWAP + MA Trend-Following  —  {TICKER}  "
        f"({START_DATE} → {END_DATE})\n"
        f"Candles: 5-min  |  Sizing: {notional_str}  |  "
        f"SL lookback: {SL_LOOKBACK} bars  |  "
        f"Swing TP: full pullback range",
        color="#fff", fontsize=10, fontweight="bold", y=0.975
    )

    path = Path(out_dir) / f"vwap_ma_equity_{TICKER}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    return str(path)


# ── CSV export ────────────────────────────────────────────────────────────────
def save_csv(all_results, out_dir):
    rows = []
    for label, res in all_results.items():
        for t in res["trades"]:
            rows.append({**t, "system": label})
    if not rows:
        return ""
    path = Path(out_dir) / f"vwap_ma_trades_{TICKER}.csv"
    pd.DataFrame(rows).sort_values("entry_time").to_csv(path, index=False)
    return str(path)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    notional_str = (f"${FIXED_NOTIONAL} fixed per trade"
                    if FIXED_NOTIONAL
                    else f"Full ${ACCOUNT_BALANCE:,} account (compounding)")
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║    VWAP + MA Trend-Following Intraday Backtester                ║
╠══════════════════════════════════════════════════════════════════╣
║  Ticker      : {TICKER:<51} ║
║  Date range  : {START_DATE} → {END_DATE:<39} ║
║  MA periods  : {str(MA_PERIODS):<51} ║
║  Sizing      : {notional_str:<51} ║
║  SL lookback : {SL_LOOKBACK} bars (prev N candles for stop){'':<23} ║
║  Swing TP    : highest/lowest high across full pullback range{'':<9} ║
╚══════════════════════════════════════════════════════════════════╝""")

    df = download_data(TICKER, START_DATE, END_DATE)
    df = add_indicators(df)

    # Systems: one per MA period × TP version, plus per-version combined
    tp_versions = ["swing", "eod"]
    all_results = {}

    for tp_ver in tp_versions:
        tp_label = "TP: Pullback Extreme" if tp_ver == "swing" else "TP: End of Day"
        combined_trades = []
        combined_account = ACCOUNT_BALANCE   # track separately

        for ma_p in MA_PERIODS:
            label = f"MA{ma_p} | {tp_label}"
            print(f"\n  Running: {label} …", end=" ", flush=True)
            trades, final_bal = run_system(df, ma_p, tp_ver)
            print(f"{len(trades)} trades  |  Final: ${final_bal:,.2f}")

            stats = compute_stats(trades, final_bal, label)
            all_results[label] = {
                "trades":        trades,
                "final_balance": final_bal,
                "stats":         stats,
            }
            combined_trades.extend(trades)

        # combined for this TP version (re-compute account curve on merged trades)
        if combined_trades:
            comb_sorted = sorted(combined_trades, key=lambda t: t["entry_time"])
            comb_account = ACCOUNT_BALANCE
            for t in comb_sorted:
                comb_account += t["pnl"]
                t = dict(t)   # don't mutate original
            comb_stats = compute_stats(
                comb_sorted, comb_account, f"Combined | {tp_label}")
            all_results[f"Combined | {tp_label}"] = {
                "trades":        comb_sorted,
                "final_balance": comb_account,
                "stats":         comb_stats,
            }

    # ── print results ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  RESULTS")
    print("═" * 65)

    for label, res in all_results.items():
        print(f"\n  ▶  {label}")
        print_stats(res["stats"])
        yb = yearly_breakdown(res["trades"])
        if not yb.empty:
            print(f"\n    Year-by-year:")
            print(yb.to_string())

    # ── overall combined across everything ────────────────────────────────────
    print("\n" + "═" * 65)
    print("  GRAND SUMMARY — Final Balances")
    print("═" * 65)
    for label, res in all_results.items():
        s = res["stats"]
        if s["total_trades"] == 0:
            print(f"  {label:<40} : No trades")
            continue
        print(f"  {label:<40} : "
              f"${res['final_balance']:>10,.2f}  "
              f"({s['total_return_pct']:+.2f}%)  "
              f"WR:{s['win_rate']}%  "
              f"PF:{s['profit_factor']}")

    chart_path = plot_results(all_results, OUTPUT_DIR)
    print(f"\n  Chart → {chart_path}")

    csv_path = save_csv(all_results, OUTPUT_DIR)
    if csv_path:
        print(f"  CSV   → {csv_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main()