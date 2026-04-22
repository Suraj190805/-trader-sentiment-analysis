# 📊 Trader Sentiment Analysis

A data-driven project that analyzes the relationship between **market sentiment (Fear & Greed Index)** and **trader performance** using historical trading data.

---

## 🚀 Overview

This project explores how market sentiment impacts trader behavior and profitability. By combining sentiment data with real trading records, the goal is to uncover patterns that can help in making smarter trading decisions.

---

## 📂 Datasets Used

1. **Bitcoin Fear & Greed Index**
   - Columns: Date, Classification (Fear/Greed)

2. **Historical Trader Data (Hyperliquid)**
   - Columns include:
     - Account
     - Execution Price
     - Size (Tokens & USD)
     - Side (Buy/Sell)
     - Timestamp
     - Closed PnL
     - Direction

---

## ⚙️ Tech Stack

- **Python 3**
- **Pandas** → Data manipulation & analysis  
- **NumPy** → Numerical operations  
- **Matplotlib** → Data visualization  

---

## 📈 Key Analysis Performed

- Data cleaning and preprocessing  
- Merging sentiment data with trading data  
- Analysis of:
  - Trader profitability vs sentiment
  - Buy/Sell behavior in Fear vs Greed markets
- Visualization of trends and patterns  

---

## 🔍 Insights

- Traders tend to behave differently during **Fear vs Greed phases**
- Certain sentiment conditions show higher/lower profitability trends
- Market sentiment can influence trading decisions significantly  

---

## 🗂️ Project Structure
```
├── data/
│   ├── historical_data.csv
│   ├── fear_greed_index.csv
│
├── analysis.py        # Main analysis script
├── report.py          # (Optional insights/report generation)
├── requirements.txt   # Dependencies
├── uploads/           # Images / outputs
└── README.md
```
---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python3 analysis.py
python3 report.py
