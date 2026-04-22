"""
=============================================================
  Primetrade.ai Assignment — analysis.py
  Trader Performance vs Bitcoin Fear & Greed Sentiment
=============================================================
  HOW TO RUN:
    1. Put your CSV files in the same folder as this script
       - historical_data.csv
       - fear_greed_index.csv
    2. Open terminal in that folder
    3. Run: python analysis.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

# ── Create output folder for charts ──────────────────────────────────────────
os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("  PRIMETRADE.AI — TRADER SENTIMENT ANALYSIS")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading data...")

trades = pd.read_csv("data/historical_data.csv")
fg     = pd.read_csv("data/fear_greed_index.csv")

print(f"  ✓ Trades loaded     : {len(trades):,} rows")
print(f"  ✓ Fear/Greed loaded : {len(fg):,} rows")


# ─────────────────────────────────────────────────────────────
# STEP 2: CLEAN & PREPARE TRADES
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Cleaning trades data...")

# Rename columns to be code-friendly (strip spaces, lowercase)
trades.columns = [c.strip().lower().replace(" ", "_") for c in trades.columns]

# The date is inside "Timestamp IST" column which looks like "02-12-2024 22:50"
trades["timestamp_ist"] = pd.to_datetime(
    trades["timestamp_ist"], format="%d-%m-%Y %H:%M", errors="coerce"
)

# Extract just the date (no time), so we can join with daily Fear/Greed
trades["date"] = trades["timestamp_ist"].dt.date
trades["date"] = pd.to_datetime(trades["date"])

# Make sure Closed PnL is numeric
trades["closed_pnl"] = pd.to_numeric(trades["closed_pnl"], errors="coerce").fillna(0)
trades["size_usd"]   = pd.to_numeric(trades["size_usd"],   errors="coerce").fillna(0)
trades["fee"]        = pd.to_numeric(trades["fee"],         errors="coerce").fillna(0)

# Net PnL = Closed PnL minus fees
trades["net_pnl"] = trades["closed_pnl"] - trades["fee"]

# Win flag: 1 if net_pnl > 0, else 0
trades["is_win"] = (trades["net_pnl"] > 0).astype(int)

# Only keep rows where a position was actually closed (Closed PnL != 0)
closed_trades = trades[trades["closed_pnl"] != 0].copy()
print(f"  ✓ Closed trades (PnL ≠ 0): {len(closed_trades):,}")


# ─────────────────────────────────────────────────────────────
# STEP 3: CLEAN & PREPARE FEAR/GREED
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Cleaning Fear & Greed data...")

fg["date"]           = pd.to_datetime(fg["date"])
fg["value"]          = pd.to_numeric(fg["value"], errors="coerce")
fg["classification"] = fg["classification"].str.strip()

# Standardise labels to 5 buckets for charts
def simplify_label(label):
    mapping = {
        "Extreme Fear": "Extreme Fear",
        "Fear":         "Fear",
        "Neutral":      "Neutral",
        "Greed":        "Greed",
        "Extreme Greed":"Extreme Greed",
    }
    return mapping.get(label, label)

fg["sentiment"] = fg["classification"].apply(simplify_label)
print(f"  ✓ Unique sentiments : {fg['sentiment'].unique().tolist()}")


# ─────────────────────────────────────────────────────────────
# STEP 4: MERGE ON DATE
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Merging datasets on date...")

merged = closed_trades.merge(
    fg[["date", "value", "sentiment"]],
    on="date",
    how="inner"
)
print(f"  ✓ Merged rows : {len(merged):,}")


# ─────────────────────────────────────────────────────────────
# STEP 5: ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Running analysis...\n")

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
COLORS = {
    "Extreme Fear": "#d62728",
    "Fear":         "#ff7f0e",
    "Neutral":      "#bcbd22",
    "Greed":        "#2ca02c",
    "Extreme Greed":"#17becf",
}

# ── A. PnL by Sentiment ───────────────────────────────────────
pnl_by_sentiment = (
    merged.groupby("sentiment")["net_pnl"]
    .agg(["mean", "median", "sum", "count"])
    .reindex(SENTIMENT_ORDER)
    .dropna()
)
pnl_by_sentiment.columns = ["Mean PnL", "Median PnL", "Total PnL", "Trade Count"]
print("── A. Average Net PnL by Sentiment ──")
print(pnl_by_sentiment.to_string())

# ── B. Win Rate by Sentiment ──────────────────────────────────
win_rate = (
    merged.groupby("sentiment")["is_win"]
    .mean()
    .reindex(SENTIMENT_ORDER)
    .dropna()
    .mul(100)
    .round(2)
)
print("\n── B. Win Rate (%) by Sentiment ──")
print(win_rate.to_string())

# ── C. Trade Volume by Sentiment ──────────────────────────────
volume = (
    merged.groupby("sentiment")["size_usd"]
    .agg(["mean", "sum"])
    .reindex(SENTIMENT_ORDER)
    .dropna()
)
volume.columns = ["Avg Trade Size USD", "Total Volume USD"]
print("\n── C. Trade Volume by Sentiment ──")
print(volume.to_string())

# ── D. Daily PnL trend with rolling sentiment ─────────────────
daily = (
    merged.groupby("date")
    .agg(daily_pnl=("net_pnl", "sum"), fg_value=("value", "first"))
    .reset_index()
    .sort_values("date")
)
daily["rolling_pnl"] = daily["daily_pnl"].rolling(7).mean()

# ── E. Pearson correlation: FG value vs net_pnl ───────────────
corr, pval = stats.pearsonr(merged["value"], merged["net_pnl"])
print(f"\n── E. Pearson Correlation (FG value vs Net PnL) ──")
print(f"  r = {corr:.4f}  |  p-value = {pval:.4f}")
if pval < 0.05:
    print("  → Statistically significant relationship!")
else:
    print("  → Not statistically significant at 95% confidence.")

# ── F. Best & Worst single-day accounts ──────────────────────
top_days = merged.groupby(["date", "sentiment"])["net_pnl"].sum().reset_index()
top_days = top_days.sort_values("net_pnl", ascending=False)
print("\n── F. Top 5 Best Days ──")
print(top_days.head(5).to_string(index=False))
print("\n── F. Top 5 Worst Days ──")
print(top_days.tail(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# STEP 6: CHARTS
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Generating charts...")

plt.style.use("dark_background")
FONT = {"fontfamily": "monospace"}

# ── Chart 1: Mean PnL by Sentiment (bar) ─────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sentiments = pnl_by_sentiment.index.tolist()
means      = pnl_by_sentiment["Mean PnL"].values
bar_colors = [COLORS.get(s, "#888") for s in sentiments]
bars = ax.bar(sentiments, means, color=bar_colors, edgecolor="white", linewidth=0.5)
ax.axhline(0, color="white", linewidth=0.8, linestyle="--")
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"${val:.2f}", ha="center", va="bottom", fontsize=9, color="white")
ax.set_title("Average Net PnL per Trade by Market Sentiment", fontsize=14, **FONT, pad=15)
ax.set_xlabel("Sentiment", **FONT)
ax.set_ylabel("Avg Net PnL (USD)", **FONT)
ax.set_facecolor("#111")
fig.patch.set_facecolor("#0d0d0d")
plt.tight_layout()
plt.savefig("charts/1_mean_pnl_by_sentiment.png", dpi=150)
plt.close()
print("  ✓ charts/1_mean_pnl_by_sentiment.png")

# ── Chart 2: Win Rate by Sentiment ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
win_sentiments = win_rate.index.tolist()
win_vals       = win_rate.values
bar_colors2    = [COLORS.get(s, "#888") for s in win_sentiments]
bars2 = ax.bar(win_sentiments, win_vals, color=bar_colors2, edgecolor="white", linewidth=0.5)
ax.axhline(50, color="yellow", linewidth=1, linestyle="--", label="50% baseline")
for bar, val in zip(bars2, win_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="white")
ax.set_title("Win Rate (%) by Market Sentiment", fontsize=14, **FONT, pad=15)
ax.set_xlabel("Sentiment", **FONT)
ax.set_ylabel("Win Rate (%)", **FONT)
ax.set_ylim(0, 100)
ax.legend()
ax.set_facecolor("#111")
fig.patch.set_facecolor("#0d0d0d")
plt.tight_layout()
plt.savefig("charts/2_win_rate_by_sentiment.png", dpi=150)
plt.close()
print("  ✓ charts/2_win_rate_by_sentiment.png")

# ── Chart 3: Trade Count by Sentiment (pie) ──────────────────
fig, ax = plt.subplots(figsize=(8, 8))
counts  = pnl_by_sentiment["Trade Count"]
pie_colors = [COLORS.get(s, "#888") for s in counts.index]
wedges, texts, autotexts = ax.pie(
    counts, labels=counts.index, autopct="%1.1f%%",
    colors=pie_colors, startangle=140,
    textprops={"color": "white", "fontsize": 11}
)
for at in autotexts:
    at.set_fontsize(10)
ax.set_title("Distribution of Trades by Sentiment", fontsize=14, **FONT, pad=15)
fig.patch.set_facecolor("#0d0d0d")
plt.tight_layout()
plt.savefig("charts/3_trade_distribution.png", dpi=150)
plt.close()
print("  ✓ charts/3_trade_distribution.png")

# ── Chart 4: Daily PnL + FG value over time ──────────────────
fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()
ax1.fill_between(daily["date"], daily["daily_pnl"], alpha=0.3, color="#00bfff", label="Daily PnL")
ax1.plot(daily["date"], daily["rolling_pnl"], color="#00bfff", linewidth=2, label="7-day Rolling PnL")
ax2.plot(daily["date"], daily["fg_value"], color="#ffa500", linewidth=1.5, alpha=0.8, linestyle="--", label="FG Index")
ax1.axhline(0, color="white", linewidth=0.5, linestyle=":")
ax1.set_title("Daily Net PnL vs Fear & Greed Index Over Time", fontsize=14, **FONT, pad=15)
ax1.set_xlabel("Date", **FONT)
ax1.set_ylabel("Net PnL (USD)", color="#00bfff", **FONT)
ax2.set_ylabel("Fear & Greed Value (0-100)", color="#ffa500", **FONT)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.set_facecolor("#111")
fig.patch.set_facecolor("#0d0d0d")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charts/4_daily_pnl_vs_fg.png", dpi=150)
plt.close()
print("  ✓ charts/4_daily_pnl_vs_fg.png")

# ── Chart 5: PnL distribution by sentiment (box plot) ────────
fig, ax = plt.subplots(figsize=(12, 6))
available_sentiments = [s for s in SENTIMENT_ORDER if s in merged["sentiment"].unique()]
data_to_plot = [merged[merged["sentiment"] == s]["net_pnl"].values for s in available_sentiments]
bp = ax.boxplot(data_to_plot, labels=available_sentiments, patch_artist=True,
                medianprops=dict(color="white", linewidth=2))
for patch, s in zip(bp["boxes"], available_sentiments):
    patch.set_facecolor(COLORS.get(s, "#888"))
    patch.set_alpha(0.7)
ax.axhline(0, color="yellow", linewidth=1, linestyle="--")
ax.set_title("PnL Distribution by Sentiment (Box Plot)", fontsize=14, **FONT, pad=15)
ax.set_xlabel("Sentiment", **FONT)
ax.set_ylabel("Net PnL (USD)", **FONT)
ax.set_facecolor("#111")
fig.patch.set_facecolor("#0d0d0d")
plt.tight_layout()
plt.savefig("charts/5_pnl_boxplot.png", dpi=150)
plt.close()
print("  ✓ charts/5_pnl_boxplot.png")

# ── Chart 6: Scatter — FG value vs Net PnL ───────────────────
fig, ax = plt.subplots(figsize=(10, 6))
scatter_colors = [COLORS.get(s, "#888") for s in merged["sentiment"]]
ax.scatter(merged["value"], merged["net_pnl"], c=scatter_colors, alpha=0.4, s=15)
m, b = np.polyfit(merged["value"], merged["net_pnl"], 1)
x_line = np.linspace(merged["value"].min(), merged["value"].max(), 100)
ax.plot(x_line, m*x_line + b, color="white", linewidth=2, label=f"Trend (r={corr:.3f})")
ax.axhline(0, color="yellow", linewidth=0.8, linestyle="--")
ax.set_title("Fear & Greed Value vs Net PnL per Trade", fontsize=14, **FONT, pad=15)
ax.set_xlabel("Fear & Greed Index Value (0=Extreme Fear, 100=Extreme Greed)", **FONT)
ax.set_ylabel("Net PnL (USD)", **FONT)
ax.legend()
ax.set_facecolor("#111")
fig.patch.set_facecolor("#0d0d0d")
# Add sentiment zone labels
for label, (lo, hi) in zip(
    ["Ext. Fear", "Fear", "Neutral", "Greed", "Ext. Greed"],
    [(0,25),(25,45),(45,55),(55,75),(75,100)]
):
    ax.axvspan(lo, hi, alpha=0.05, color=list(COLORS.values())[
        ["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"].index(
            {"Ext. Fear":"Extreme Fear","Fear":"Fear","Neutral":"Neutral",
             "Greed":"Greed","Ext. Greed":"Extreme Greed"}[label]
        )
    ])
plt.tight_layout()
plt.savefig("charts/6_scatter_fg_vs_pnl.png", dpi=150)
plt.close()
print("  ✓ charts/6_scatter_fg_vs_pnl.png")

# ── Save summary stats to CSV for report.py ──────────────────
pnl_by_sentiment.to_csv("charts/summary_pnl.csv")
win_rate.to_frame("win_rate").to_csv("charts/summary_winrate.csv")
daily.to_csv("charts/summary_daily.csv", index=False)
merged.to_csv("charts/merged_data.csv", index=False)

print("\n" + "=" * 60)
print("  ANALYSIS COMPLETE!")
print("  ✓ All charts saved to the /charts folder")
print("  ✓ Now run: python report.py")
print("=" * 60)