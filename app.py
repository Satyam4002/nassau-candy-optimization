import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy – Factory Optimization",
    page_icon="🍬",
    layout="wide"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory"
}

FACTORY_COORDS = {
    "Lot's O' Nuts":    (32.881893, -111.768036),
    "Wicked Choccy's":  (32.076176, -81.088371),
    "Sugar Shack":      (48.119140, -96.181150),
    "Secret Factory":   (41.446333, -90.565487),
    "The Other Factory":(35.117500, -89.971107)
}

ALL_FACTORIES = list(FACTORY_COORDS.keys())

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Nassau Candy Distributor.csv")
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
    df['Lead Time']  = (df['Ship Date'] - df['Order Date']).dt.days
    df['Factory']    = df['Product Name'].map(FACTORY_MAP)
    df['Margin %']   = (df['Gross Profit'] / df['Sales'] * 100).round(2)
    return df

@st.cache_resource
def train_model(df):
    le_dict = {}
    dfe = df.copy()
    for col in ['Ship Mode','Region','Product Name','Factory','Division','Country/Region']:
        le = LabelEncoder()
        dfe[col+'_enc'] = le.fit_transform(dfe[col])
        le_dict[col] = le

    features = ['Ship Mode_enc','Region_enc','Product Name_enc','Factory_enc','Division_enc','Units','Country/Region_enc']
    X = dfe[features]; y = dfe['Lead Time']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression':   LinearRegression(),
        'Random Forest':        RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting':    GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    results = {}
    trained = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        results[name] = dict(
            RMSE=round(np.sqrt(mean_squared_error(y_test, p)), 2),
            MAE =round(mean_absolute_error(y_test, p), 2),
            R2  =round(r2_score(y_test, p), 4)
        )
        trained[name] = m

    best_name = min(results, key=lambda x: results[x]['RMSE'])
    return trained[best_name], le_dict, results, features

df = load_data()
best_model, le_dict, model_results, feature_cols = train_model(df)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Willy_Wonka_%26_the_Chocolate_Factory_logo.svg/320px-Willy_Wonka_%26_the_Chocolate_Factory_logo.svg.png",
                 width=200, use_column_width=True)
st.sidebar.title("🍬 Nassau Candy")
st.sidebar.markdown("**Factory Optimization System**")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "📊 Overview & EDA",
    "🤖 Predictive Model",
    "🔁 Scenario Simulator",
    "🏆 Recommendations",
    "⚠️ Risk & Impact"
])

st.sidebar.divider()
st.sidebar.caption("Data: 10,194 orders | 2024–2025")
st.sidebar.caption("15 products | 5 factories | 4 regions")

# ─────────────────────────────────────────────
# PAGE 1: OVERVIEW & EDA
# ─────────────────────────────────────────────
if page == "📊 Overview & EDA":
    st.title("📊 Nassau Candy – Operational Overview")
    st.markdown("Exploratory analysis of orders, shipping, and profitability across factories and regions.")

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Orders",    f"{len(df):,}")
    col2.metric("Total Sales",     f"${df['Sales'].sum():,.0f}")
    col3.metric("Total Profit",    f"${df['Gross Profit'].sum():,.0f}")
    col4.metric("Avg Lead Time",   f"{df['Lead Time'].mean():.0f} days")
    col5.metric("Avg Margin",      f"{df['Margin %'].mean():.1f}%")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Orders & Sales by Region")
        reg = df.groupby('Region').agg(Orders=('Row ID','count'), Sales=('Sales','sum')).reset_index()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(reg['Region'], reg['Orders'], color=['#4e79a7','#f28e2b','#e15759','#76b7b2'])
        ax.set_ylabel("Order Count"); ax.set_title("Orders by Region")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                    str(int(bar.get_height())), ha='center', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        st.subheader("Lead Time by Ship Mode")
        sm = df.groupby('Ship Mode')['Lead Time'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ['#59a14f' if v == sm.min() else '#e15759' if v == sm.max() else '#4e79a7' for v in sm]
        ax.barh(sm.index, sm.values, color=colors)
        ax.set_xlabel("Avg Lead Time (days)"); ax.set_title("Avg Lead Time by Ship Mode")
        for i, v in enumerate(sm.values):
            ax.text(v+1, i, f"{v:.0f}d", va='center', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Factory Workload & Avg Lead Time")
        fac = df.groupby('Factory').agg(Orders=('Row ID','count'), AvgLT=('Lead Time','mean')).reset_index()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax2 = ax.twinx()
        x = range(len(fac))
        ax.bar(x, fac['Orders'], color='#4e79a7', alpha=0.7, label='Orders')
        ax2.plot(x, fac['AvgLT'], 'o-', color='#e15759', lw=2, label='Avg Lead Time')
        ax.set_xticks(list(x)); ax.set_xticklabels(fac['Factory'], rotation=20, ha='right', fontsize=8)
        ax.set_ylabel("Orders"); ax2.set_ylabel("Avg Lead Time (days)")
        ax.set_title("Factory: Volume vs Lead Time")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_d:
        st.subheader("Profit Margin by Product")
        prod = df.groupby('Product Name')['Margin %'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        colors = ['#59a14f' if v >= prod.mean() else '#e15759' for v in prod.values]
        ax.barh(prod.index, prod.values, color=colors)
        ax.set_xlabel("Avg Margin %"); ax.set_title("Profit Margin by Product")
        ax.axvline(prod.mean(), color='gray', linestyle='--', alpha=0.7, label=f'Avg {prod.mean():.1f}%')
        ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("📋 Lead Time by Product & Factory")
    pt = df.groupby(['Product Name','Factory'])['Lead Time'].mean().reset_index()
    pt.columns = ['Product','Factory','Avg Lead Time (days)']
    pt = pt.sort_values('Avg Lead Time (days)', ascending=False)
    pt['Avg Lead Time (days)'] = pt['Avg Lead Time (days)'].round(0).astype(int)
    st.dataframe(pt, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# PAGE 2: PREDICTIVE MODEL
# ─────────────────────────────────────────────
elif page == "🤖 Predictive Model":
    st.title("🤖 Lead Time Prediction Model")
    st.markdown("Three regression models were trained and evaluated to predict shipping lead time.")

    # Model comparison table
    st.subheader("Model Comparison")
    mdf = pd.DataFrame(model_results).T.reset_index()
    mdf.columns = ['Model','RMSE','MAE','R²']
    best_row = mdf.loc[mdf['RMSE'].idxmin(), 'Model']

    def highlight_best(row):
        return ['background-color: #d4edda' if row['Model'] == best_row else '' for _ in row]

    st.dataframe(mdf.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)
    st.success(f"✅ Best Model: **{best_row}** (lowest RMSE) — selected for simulation")

    st.divider()
    st.subheader("🔮 Predict Lead Time for a Single Order")

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_product  = st.selectbox("Product", sorted(df['Product Name'].unique()))
        sel_factory  = st.selectbox("Factory (Current)", ALL_FACTORIES,
                                    index=ALL_FACTORIES.index(FACTORY_MAP.get(sel_product, ALL_FACTORIES[0])))
    with col2:
        sel_region   = st.selectbox("Destination Region", sorted(df['Region'].unique()))
        sel_shipmode = st.selectbox("Ship Mode", sorted(df['Ship Mode'].unique()))
    with col3:
        sel_units    = st.slider("Units", 1, 14, 3)
        sel_country  = st.selectbox("Country", ['United States','Canada'])

    if st.button("Predict Lead Time", type="primary"):
        row = {
            'Ship Mode_enc':     le_dict['Ship Mode'].transform([sel_shipmode])[0],
            'Region_enc':        le_dict['Region'].transform([sel_region])[0],
            'Product Name_enc':  le_dict['Product Name'].transform([sel_product])[0],
            'Factory_enc':       le_dict['Factory'].transform([sel_factory])[0],
            'Division_enc':      le_dict['Division'].transform([df.loc[df['Product Name']==sel_product,'Division'].iloc[0]])[0],
            'Units':             sel_units,
            'Country/Region_enc':le_dict['Country/Region'].transform([sel_country])[0],
        }
        X_pred = pd.DataFrame([row])[feature_cols]
        pred   = best_model.predict(X_pred)[0]
        actual = df[df['Product Name']==sel_product]['Lead Time'].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Lead Time", f"{pred:.0f} days")
        c2.metric("Product Avg Lead Time", f"{actual:.0f} days")
        delta = pred - actual
        c3.metric("Δ vs Average", f"{delta:+.0f} days",
                  delta_color="inverse" if delta > 0 else "normal")

# ─────────────────────────────────────────────
# PAGE 3: SCENARIO SIMULATOR
# ─────────────────────────────────────────────
elif page == "🔁 Scenario Simulator":
    st.title("🔁 Factory Reallocation Scenario Simulator")
    st.markdown("Simulate what happens if a product is moved to a different factory. Compare current vs alternative lead times and profit impact.")

    col1, col2 = st.columns(2)
    with col1:
        sim_product  = st.selectbox("Select Product to Reassign", sorted(df['Product Name'].unique()))
        sim_region   = st.selectbox("Target Region", sorted(df['Region'].unique()))
        sim_shipmode = st.selectbox("Ship Mode", sorted(df['Ship Mode'].unique()))
    with col2:
        sim_units   = st.slider("Units", 1, 14, 3)
        sim_country = st.selectbox("Country", ['United States','Canada'])
        opt_priority = st.slider("Optimization Priority: Speed ◄──► Profit", 0, 100, 50,
                                 help="0 = Prioritize speed, 100 = Prioritize profit")

    current_factory = FACTORY_MAP.get(sim_product, ALL_FACTORIES[0])

    if st.button("▶ Run Scenario Simulation", type="primary"):
        division = df.loc[df['Product Name']==sim_product,'Division'].iloc[0]
        sim_results = []

        for factory in ALL_FACTORIES:
            row = {
                'Ship Mode_enc':     le_dict['Ship Mode'].transform([sim_shipmode])[0],
                'Region_enc':        le_dict['Region'].transform([sim_region])[0],
                'Product Name_enc':  le_dict['Product Name'].transform([sim_product])[0],
                'Factory_enc':       le_dict['Factory'].transform([factory])[0],
                'Division_enc':      le_dict['Division'].transform([division])[0],
                'Units':             sim_units,
                'Country/Region_enc':le_dict['Country/Region'].transform([sim_country])[0],
            }
            X_pred  = pd.DataFrame([row])[feature_cols]
            pred_lt = best_model.predict(X_pred)[0]
            avg_profit = df[df['Product Name']==sim_product]['Gross Profit'].mean()

            sim_results.append({
                'Factory': factory,
                'Predicted Lead Time': round(pred_lt),
                'Avg Profit/Order': round(avg_profit, 2),
                'Is Current': factory == current_factory
            })

        sim_df = pd.DataFrame(sim_results).sort_values('Predicted Lead Time')
        current_lt = sim_df.loc[sim_df['Is Current'], 'Predicted Lead Time'].values[0]
        sim_df['Lead Time Reduction'] = (current_lt - sim_df['Predicted Lead Time']).round(0).astype(int)
        sim_df['Status'] = sim_df['Factory'].apply(lambda x: '⭐ Current' if x == current_factory else '🔁 Alternative')

        st.subheader("Simulation Results")
        st.dataframe(
            sim_df[['Status','Factory','Predicted Lead Time','Lead Time Reduction','Avg Profit/Order']],
            use_container_width=True, hide_index=True
        )

        st.subheader("Lead Time Comparison")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = ['#e15759' if r['Is Current'] else '#4e79a7' for _, r in sim_df.iterrows()]
        bars = ax.bar(sim_df['Factory'], sim_df['Predicted Lead Time'], color=colors)
        red_patch  = mpatches.Patch(color='#e15759', label='Current Factory')
        blue_patch = mpatches.Patch(color='#4e79a7', label='Alternative')
        ax.legend(handles=[red_patch, blue_patch])
        ax.set_ylabel("Predicted Lead Time (days)"); ax.set_title(f"Lead Time by Factory for: {sim_product}")
        ax.set_xticklabels(sim_df['Factory'], rotation=20, ha='right', fontsize=9)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    str(int(bar.get_height())), ha='center', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        best_alt = sim_df[~sim_df['Is Current']].iloc[0]
        if best_alt['Lead Time Reduction'] > 0:
            st.success(f"✅ Moving **{sim_product}** to **{best_alt['Factory']}** could reduce lead time by **{best_alt['Lead Time Reduction']} days**.")
        else:
            st.info(f"ℹ️ Current factory **{current_factory}** already provides competitive lead times for this product.")

# ─────────────────────────────────────────────
# PAGE 4: RECOMMENDATIONS
# ─────────────────────────────────────────────
elif page == "🏆 Recommendations":
    st.title("🏆 Factory Reassignment Recommendations")
    st.markdown("Ranked list of products that would benefit most from factory reallocation.")

    region_filter   = st.selectbox("Filter by Region", ['All'] + sorted(df['Region'].unique().tolist()))
    shipmode_filter = st.selectbox("Filter by Ship Mode", ['All'] + sorted(df['Ship Mode'].unique().tolist()))

    recommendations = []

    for product in df['Product Name'].unique():
        current_factory = FACTORY_MAP.get(product)
        division = df.loc[df['Product Name']==product,'Division'].iloc[0]
        regions  = [region_filter] if region_filter != 'All' else df['Region'].unique().tolist()
        shipmodes= [shipmode_filter] if shipmode_filter != 'All' else df['Ship Mode'].unique().tolist()

        best_lt = None; best_factory = None

        for region in regions:
            for smode in shipmodes:
                current_row = {
                    'Ship Mode_enc':     le_dict['Ship Mode'].transform([smode])[0],
                    'Region_enc':        le_dict['Region'].transform([region])[0],
                    'Product Name_enc':  le_dict['Product Name'].transform([product])[0],
                    'Factory_enc':       le_dict['Factory'].transform([current_factory])[0],
                    'Division_enc':      le_dict['Division'].transform([division])[0],
                    'Units': 3,
                    'Country/Region_enc':le_dict['Country/Region'].transform(['United States'])[0],
                }
                current_lt = best_model.predict(pd.DataFrame([current_row])[feature_cols])[0]

                for factory in ALL_FACTORIES:
                    if factory == current_factory: continue
                    row = dict(current_row); row['Factory_enc'] = le_dict['Factory'].transform([factory])[0]
                    lt  = best_model.predict(pd.DataFrame([row])[feature_cols])[0]
                    if best_lt is None or lt < best_lt:
                        best_lt = lt; best_factory = factory; best_current_lt = current_lt

        avg_profit = df[df['Product Name']==product]['Gross Profit'].mean()
        reduction  = best_current_lt - best_lt

        recommendations.append({
            'Product':         product,
            'Current Factory': current_factory,
            'Recommended Factory': best_factory,
            'Lead Time Reduction (days)': round(reduction),
            'Avg Profit/Order ($)': round(avg_profit, 2),
            'Priority Score': round(reduction * 0.7 + avg_profit * 0.3, 1)
        })

    rdf = pd.DataFrame(recommendations).sort_values('Priority Score', ascending=False)
    rdf['Rank'] = range(1, len(rdf)+1)

    st.subheader("Top Reassignment Recommendations")
    display_df = rdf[['Rank','Product','Current Factory','Recommended Factory',
                       'Lead Time Reduction (days)','Avg Profit/Order ($)','Priority Score']]

    def color_rows(row):
        if row['Rank'] <= 3:
            return ['background-color: #fff3cd'] * len(row)
        return [''] * len(row)

    st.dataframe(display_df.style.apply(color_rows, axis=1), use_container_width=True, hide_index=True)
    st.caption("🟡 Top 3 highlighted | Priority Score = 70% Lead Time Reduction + 30% Profit")

    st.divider()
    st.subheader("Lead Time Reduction Potential")
    top5 = rdf.head(5)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(top5['Product'], top5['Lead Time Reduction (days)'],
                   color=['#f28e2b','#f28e2b','#f28e2b','#4e79a7','#4e79a7'])
    ax.set_xlabel("Days Saved"); ax.set_title("Top 5 Products by Lead Time Reduction Potential")
    for bar in bars:
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{bar.get_width():.0f}d", va='center', fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────
# PAGE 5: RISK & IMPACT
# ─────────────────────────────────────────────
elif page == "⚠️ Risk & Impact":
    st.title("⚠️ Risk & Profit Impact Panel")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 High Lead Time Alerts")
        threshold = st.slider("Alert Threshold (days)", 1300, 1500, 1380)
        high_lt = df.groupby(['Product Name','Factory','Region'])['Lead Time'].mean().reset_index()
        high_lt.columns = ['Product','Factory','Region','Avg Lead Time']
        alerts = high_lt[high_lt['Avg Lead Time'] >= threshold].sort_values('Avg Lead Time', ascending=False)
        if len(alerts) > 0:
            st.error(f"🚨 {len(alerts)} route-product combinations exceed {threshold} days!")
            st.dataframe(alerts.head(10), use_container_width=True, hide_index=True)
        else:
            st.success("✅ No routes exceed the alert threshold.")

    with col2:
        st.subheader("💰 Profit Impact by Factory")
        fac_profit = df.groupby('Factory').agg(
            Total_Profit=('Gross Profit','sum'),
            Orders=('Row ID','count'),
            Avg_Margin=('Margin %','mean')
        ).reset_index().sort_values('Total_Profit', ascending=False)

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        bars = ax.bar(fac_profit['Factory'], fac_profit['Total_Profit'],
                      color=['#59a14f','#4e79a7','#f28e2b','#e15759','#76b7b2'])
        ax.set_ylabel("Total Gross Profit ($)"); ax.set_title("Profit by Factory")
        ax.set_xticklabels(fac_profit['Factory'], rotation=20, ha='right', fontsize=8)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                    f"${bar.get_height():,.0f}", ha='center', fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("📉 Products with High Lead Time BUT High Profit (Reassignment Priority)")
    combined = df.groupby('Product Name').agg(
        Avg_Lead_Time=('Lead Time','mean'),
        Avg_Profit=('Gross Profit','mean'),
        Factory=('Factory','first')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    scatter = ax.scatter(combined['Avg_Lead_Time'], combined['Avg_Profit'],
                         s=120, c=range(len(combined)), cmap='tab10', zorder=3)
    for _, row in combined.iterrows():
        ax.annotate(row['Product Name'], (row['Avg_Lead_Time'], row['Avg_Profit']),
                    fontsize=7, xytext=(4, 4), textcoords='offset points')
    ax.axvline(combined['Avg_Lead_Time'].mean(), color='red', linestyle='--', alpha=0.5, label='Avg Lead Time')
    ax.axhline(combined['Avg_Profit'].mean(), color='blue', linestyle='--', alpha=0.5, label='Avg Profit')
    ax.set_xlabel("Avg Lead Time (days)"); ax.set_ylabel("Avg Profit/Order ($)")
    ax.set_title("Lead Time vs Profit — Products in Top-Right = High Priority for Reassignment")
    ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()
    st.caption("🔴 Top-right quadrant = High lead time + High profit → reassignment priority")

    st.divider()
    st.subheader("📊 Monthly Order & Profit Trend")
    df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Month').agg(Orders=('Row ID','count'), Profit=('Gross Profit','sum')).reset_index()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax2 = ax.twinx()
    ax.bar(monthly['Month'], monthly['Orders'], color='#4e79a7', alpha=0.6, label='Orders')
    ax2.plot(monthly['Month'], monthly['Profit'], 'o-', color='#e15759', lw=2, label='Profit')
    ax.set_xlabel("Month"); ax.set_ylabel("Orders"); ax2.set_ylabel("Gross Profit ($)")
    ax.set_title("Monthly Orders & Profit Trend")
    n = len(monthly); step = max(1, n//12)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels([monthly['Month'].iloc[i] for i in range(0, n, step)], rotation=45, ha='right', fontsize=7)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='upper left')
    plt.tight_layout(); st.pyplot(fig); plt.close()
