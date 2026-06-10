# Nassau Candy Distributor
## Factory Reallocation & Shipping Optimization Recommendation System

---

## Project Files
- `Nassau_Candy_Distributor.csv` — Dataset (place in same folder)
- `nassau_analysis.py` — Full EDA + ML analysis (run this first)
- `app.py` — Interactive Streamlit dashboard

---

## How to Run

### Step 1 — Install dependencies
```
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### Step 2 — Run EDA Analysis
```
python nassau_analysis.py
```
This prints all analysis results and saves `nassau_eda_dashboard.png`

### Step 3 — Launch the Dashboard
```
streamlit run app.py
```
Open browser at http://localhost:8501

---

## Dashboard Modules
1. **Overview & EDA** — KPIs, charts, region/factory analysis
2. **Predictive Model** — 3 ML models compared, predict lead time
3. **Scenario Simulator** — Reassign any product to any factory
4. **Recommendations** — Ranked reassignment suggestions
5. **Risk & Impact** — Alerts, profit analysis, priority scatter

---

## Key Findings
- **Hair Toffee** has the highest lead time (1455 days avg)
- **Standard Class** is paradoxically faster than First Class
- **Lickable Wallpaper & Everlasting Gobstopper** = high profit + high lead time → prime reassignment candidates
- **Gulf region** has the best delivery performance
- **Lot's O' Nuts** handles 56% of all orders
