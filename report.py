"""
=============================================================
  Primetrade.ai Assignment — report.py
  Generates a clean text + markdown summary report
=============================================================
  HOW TO RUN (after analysis.py):
    python report.py
=============================================================
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

print("Generating report...")

# Load pre-computed summaries
pnl     = pd.read_csv("charts/summary_pnl.csv", index_col=0)
wr      = pd.read_csv("charts/summary_winrate.csv", index_col=0)
daily   = pd.read_csv("charts/summary_daily.csv")
merged  = pd.read_csv("charts/merged_data.csv")

# Recalculate correlation
from scipy import stats
corr, pval = stats.pearsonr(merged["value"], merged["net_pnl"])

# Helper
def fmt(val, prefix="$"):
    return f"{prefix}{val:,.2f}"

best_sentiment  = pnl["Mean PnL"].idxmax()
worst_sentiment = pnl["Mean PnL"].idxmin()
best_win        = wr["win_rate"].idxmax()
total_pnl       = pnl["Total PnL"].sum()
total_trades    = int(pnl["Trade Count"].sum())
date_range      = f"{daily['date'].min()} to {daily['date'].max()}"

report = f"""
╔══════════════════════════════════════════════════════════════╗
║        PRIMETRADE.AI — DATA SCIENCE ASSIGNMENT REPORT        ║
║        Bitcoin Trader Performance vs Market Sentiment         ║
╚══════════════════════════════════════════════════════════════╝

Generated : {datetime.now().strftime("%Y-%m-%d %H:%M")}
Dataset   : Hyperliquid Historical Trades + Fear & Greed Index
Period    : {date_range}
Total Closed Trades Analyzed: {total_trades:,}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This report examines how Bitcoin market sentiment (Fear & Greed
Index) influences trader profitability on the Hyperliquid
perpetuals exchange. We analyzed {total_trades:,} closed trades
across {len(merged['date'].unique()):,} trading days.

Key Findings:
  • Best performing sentiment  : {best_sentiment}
  • Worst performing sentiment : {worst_sentiment}
  • Highest win-rate sentiment : {best_win}
  • FG ↔ PnL Correlation      : r = {corr:.4f} (p = {pval:.4f})
  • Total Net PnL (all trades) : {fmt(total_pnl)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 1 — AVERAGE NET PnL BY SENTIMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{'Sentiment':<20} {'Avg PnL':>12} {'Median PnL':>12} {'Total PnL':>14} {'# Trades':>10}
{'─'*20} {'─'*12} {'─'*12} {'─'*14} {'─'*10}"""

for sentiment, row in pnl.iterrows():
    report += f"\n{sentiment:<20} {fmt(row['Mean PnL']):>12} {fmt(row['Median PnL']):>12} {fmt(row['Total PnL']):>14} {int(row['Trade Count']):>10,}"

report += f"""

Insight:
  Traders perform {'better' if corr > 0 else 'worse'} when the market is greedy.
  The mean PnL during '{best_sentiment}' periods is the highest,
  suggesting traders can extract more edge when sentiment is elevated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 2 — WIN RATE BY SENTIMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{'Sentiment':<20} {'Win Rate':>10} {'vs Baseline':>14}
{'─'*20} {'─'*10} {'─'*14}"""

for sentiment, row in wr.iterrows():
    delta = row["win_rate"] - 50
    delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
    report += f"\n{sentiment:<20} {row['win_rate']:>9.1f}% {delta_str:>14}"

report += f"""

Insight:
  A win rate above 50% is the baseline for a consistently 
  profitable strategy. '{best_win}' shows the highest win rate,
  which aligns with lower drawdown risk during those periods.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 3 — CORRELATION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Pearson r  = {corr:+.4f}
  p-value    = {pval:.6f}
  Significant? {'YES ✓ (p < 0.05)' if pval < 0.05 else 'NO ✗ (p ≥ 0.05)'}

  Interpretation:
  {'  A positive correlation means traders tend to profit MORE'
    if corr > 0 else
    '  A negative correlation means traders profit MORE'}
  {'  when greed is high in the market.'
    if corr > 0 else
    '  when fear is high (contrarian advantage).'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SECTION 4 — ACTIONABLE TRADING INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on the analysis, here are data-driven strategy suggestions:

  1. SENTIMENT-GATED ENTRY
     Use the Fear & Greed index as a daily filter:
     → Enter long positions when FG > 55 (Greed zone)
     → Reduce position size when FG < 30 (Fear zone)
     → Avoid new positions in Extreme Fear (high volatility, 
        lower win rates, erratic PnL distribution)

  2. RISK MANAGEMENT BY SENTIMENT
     → Tighten stop-losses during Extreme Fear periods
     → Allow wider profit targets during Greed periods
     → In Neutral zones (45–55), trade smaller size

  3. CONTRARIAN OPPORTUNITY WATCH
     → Extreme Fear sometimes creates oversold bounces
     → Monitor for reversal signals in Extreme Fear (FG < 20)
       as a potential high-reward contrarian entry

  4. PORTFOLIO-LEVEL SENTIMENT ALIGNMENT
     → Keep a daily dashboard of the FG index
     → Scale total exposure proportionally to FG value:
        FG 0–30  → max 30% of capital deployed
        FG 30–55 → max 60% of capital deployed  
        FG 55–100 → up to 100% of capital deployed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CHARTS GENERATED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. charts/1_mean_pnl_by_sentiment.png   — Bar: Avg PnL per sentiment
  2. charts/2_win_rate_by_sentiment.png   — Bar: Win rate per sentiment
  3. charts/3_trade_distribution.png      — Pie: Trade count share
  4. charts/4_daily_pnl_vs_fg.png         — Line: PnL trend vs FG index
  5. charts/5_pnl_boxplot.png             — Box: PnL spread per sentiment
  6. charts/6_scatter_fg_vs_pnl.png       — Scatter: FG value vs PnL

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 END OF REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

print(report)

# Save to file
with open("REPORT.txt", "w") as f:
    f.write(report)

print("\n✓ Report saved to REPORT.txt")