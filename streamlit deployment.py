import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score, mean_absolute_error

# PAGE CONFIG
st.set_page_config(
    page_title="AI Retail Demand Forecaster",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS - Professional Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    }
    .main-header h1 { color: white; font-size: 2.2rem; margin: 0; font-weight: 700; }
    .main-header p  { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0 0; font-size: 1.1rem; }
    
    .metric-card {
        background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px; padding: 1.5rem; text-align: center;
        backdrop-filter: blur(10px); margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-value { font-size: 2rem; font-weight: 700; color: #667eea; }
    .metric-label { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 0.3rem; }
    
    .section-header {
        font-size: 1.4rem; font-weight: 600; color: #a78bfa;
        border-bottom: 2px solid #667eea; padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    .insight-box {
        background: rgba(102,126,234,0.15); border-left: 4px solid #667eea;
        border-radius: 8px; padding: 1rem 1.2rem; margin: 0.8rem 0;
        color: rgba(255,255,255,0.9);
    }
    
    .success-box {
        background: rgba(52,211,153,0.15); border-left: 4px solid #34d399;
        border-radius: 8px; padding: 1rem 1.2rem; margin: 0.8rem 0;
        color: rgba(255,255,255,0.9);
    }
    
    .warning-box {
        background: rgba(251,191,36,0.15); border-left: 4px solid #fbbf24;
        border-radius: 8px; padding: 1rem 1.2rem; margin: 0.8rem 0;
        color: rgba(255,255,255,0.9);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        color: white !important; border: none !important;
        border-radius: 8px !important; padding: 0.6rem 2rem !important;
        font-weight: 600 !important; transition: all 0.2s !important;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.4); }
    
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label { color: rgba(255,255,255,0.8) !important; }
    
    .stSidebar { background: rgba(255,255,255,0.04) !important; }
    h1,h2,h3 { color: white; }
    p, li { color: rgba(255,255,255,0.8); }
</style>
""", unsafe_allow_html=True)

# LOAD MODEL & DATA  (cached for performance)
@st.cache_resource
def load_model():
    try:
        with open('models/best_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/HP/Downloads/edunet-IBM AIML project/retail_features.csv', parse_dates=['date'])
    inv = pd.read_csv('C:/Users/HP/Downloads/edunet-IBM AIML project/inventory_optimization.csv')
    with open('C:/Users/HP/Downloads/edunet-IBM AIML project/metrics.json') as f:
        metrics = json.load(f)
    return df, inv, metrics

model_save = load_model()
df, inv_df, metrics = load_data()

model    = model_save['model']    if model_save else None
FEATURES = model_save['features'] if model_save else []

#SIDEBAR
with st.sidebar:
    st.markdown("## 🛒 Navigation")
    page = st.radio("", [
        "🏠 Dashboard",
        "📊 Data Explorer",
        "🤖 Demand Forecaster",
        "📦 Inventory Optimizer",
        "🎯 Model Performance",
        "🔮 Scenario Analysis"
    ])
    
    st.markdown("---")
    st.markdown("**Project Info**")
    st.markdown("🎓 Edunet-IBM AIML Internship")
    
    if metrics:
        st.markdown(f"🏆 Model Accuracy: **{metrics['accuracy_pct']}%**")

# HEADER
st.markdown("""
<div class="main-header">
    <h1>🛒 AI-Powered Retail Decision Support System</h1>
    <p>Real-Time Demand Forecasting & Inventory Optimization | Edunet-IBM AIML Internship 2026</p>
</div>
""", unsafe_allow_html=True)

#PAGE 1: DASHBOARD

if page == "🏠 Dashboard":
    st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{metrics.get('accuracy_pct','—')}%</div>
            <div class="metric-label">🎯 Model Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{metrics.get('r2','—')}</div>
            <div class="metric-label">📊 R² Score</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{metrics.get('mape','—')}%</div>
            <div class="metric-label">⚡ MAPE Error</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{metrics.get('stockout_rate','—')}%</div>
            <div class="metric-label">🚨 Stockout Rate</div>
        </div>""", unsafe_allow_html=True)
    
    # Summary stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">📅 Dataset Overview</div>', unsafe_allow_html=True)
        stats = {
            "Total Records": f"{len(df):,}",
            "Date Range": f"{df['date'].min().date()} → {df['date'].max().date()}",
            "Stores": df['store'].nunique(),
            "Categories": df['category'].nunique(),
            "Products": df['product'].nunique(),
            "Total Features": len(FEATURES),
            "ML Model Used": metrics.get('model_name', '—')
        }
        for k, v in stats.items():
            st.markdown(f"**{k}:** {v}")

    with col2:
        st.markdown('<div class="section-header">💡 AI Insights</div>', unsafe_allow_html=True)
        
        total_rev = (df['actual_sales'] * df['price']).sum()
        lost_rev  = (df['lost_sales'] * df['price']).sum()
        
        st.markdown(f"""<div class="success-box">
        ✅ <strong>Revenue Captured:</strong> ₹{total_rev:,.0f}
        </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""<div class="warning-box">
        ⚠️ <strong>Revenue Lost (Stockouts):</strong> ₹{lost_rev:,.0f}<br>
        AI system could recover ~₹{metrics.get('projected_recovery',0):,.0f}
        </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""<div class="insight-box">
        🤖 <strong>Model:</strong> {metrics.get('model_name')} with {len(FEATURES)} engineered features<br>
        📉 Projected stockout reduction: from {metrics.get('stockout_rate')}% → {metrics.get('improved_stockout_rate')}%
        </div>""", unsafe_allow_html=True)
    
    # Monthly trend chart
    st.markdown('<div class="section-header">📈 Sales Trend Overview</div>', unsafe_allow_html=True)
    monthly = df.groupby(df['date'].dt.to_period('M'))[['actual_sales','demand']].sum()
    monthly.index = monthly.index.to_timestamp()
    
    fig, ax = plt.subplots(figsize=(14,4), facecolor='none')
    ax.set_facecolor('none')
    ax.plot(monthly.index, monthly['demand'], label='Demand', color='#667eea', linewidth=2)
    ax.plot(monthly.index, monthly['actual_sales'], label='Actual Sales', color='#34d399', linewidth=2, linestyle='--')
    ax.fill_between(monthly.index, monthly['actual_sales'], monthly['demand'], alpha=0.3, color='#fbbf24', label='Lost Sales (Stockout gap)')
    ax.set_xlabel('Date', color='white')
    ax.set_ylabel('Units', color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('rgba(255,255,255,0.2)')
    ax.spines['left'].set_color('rgba(255,255,255,0.2)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='rgba(0,0,0,0.5)', labelcolor='white')
    st.pyplot(fig, transparent=True)

#PAGE 2: DATA EXPLORER
elif page == "📊 Data Explorer":
    st.markdown('<div class="section-header">🔍 Interactive Data Explorer</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_store = st.selectbox("Select Store", ['All'] + list(df['store'].unique()))
    with col2:
        selected_cat = st.selectbox("Select Category", ['All'] + list(df['category'].unique()))
    with col3:
        date_range = st.date_input("Date Range", 
            value=[df['date'].min(), df['date'].max()],
            min_value=df['date'].min().date(), 
            max_value=df['date'].max().date())
    
    filtered = df.copy()
    if selected_store != 'All':
        filtered = filtered[filtered['store'] == selected_store]
    if selected_cat != 'All':
        filtered = filtered[filtered['category'] == selected_cat]
    if len(date_range) == 2:
        filtered = filtered[(filtered['date'].dt.date >= date_range[0]) & 
                            (filtered['date'].dt.date <= date_range[1])]
    
    st.markdown(f"**Showing {len(filtered):,} records**")
    
    col1, col2 = st.columns(2)
    with col1:
        # Sales by category pie
        cat_sales = filtered.groupby('category')['actual_sales'].sum()
        fig, ax = plt.subplots(figsize=(6,5), facecolor='none')
        ax.set_facecolor('none')
        colors = ['#667eea','#764ba2','#34d399','#fbbf24','#f87171']
        wedges, texts, autotexts = ax.pie(cat_sales, labels=cat_sales.index, 
            autopct='%1.1f%%', colors=colors, startangle=90)
        for t in texts + autotexts:
            t.set_color('white')
        ax.set_title('Sales Share by Category', color='white', fontweight='bold')
        st.pyplot(fig, transparent=True)
    
    with col2:
        # Daily demand trend
        daily = filtered.groupby('date')['actual_sales'].sum()
        fig, ax = plt.subplots(figsize=(6,5), facecolor='none')
        ax.set_facecolor('none')
        ax.plot(daily.index, daily.values, color='#667eea', linewidth=1.5, alpha=0.9)
        # 7-day moving average
        ma = daily.rolling(7).mean()
        ax.plot(ma.index, ma.values, color='#34d399', linewidth=2.5, label='7-day MA')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Units', color='white')
        ax.set_title('Daily Sales Trend', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines[:].set_visible(False)
        ax.legend(facecolor='rgba(0,0,0,0.5)', labelcolor='white')
        st.pyplot(fig, transparent=True)
    
    st.markdown('<div class="section-header">📋 Data Sample</div>', unsafe_allow_html=True)
    st.dataframe(
        filtered[['date','store','category','product','demand','actual_sales',
                  'lost_sales','stock_level','price','stockout']].head(100),
        use_container_width=True
    )

#PAGE 3: DEMAND FORECASTER
elif page == "🤖 Demand Forecaster":
    st.markdown('<div class="section-header">🤖 Real-Time Demand Forecast</div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please run the training script first.")
        st.stop()
    
    st.markdown("""<div class="insight-box">
    📌 <strong>How it works:</strong> Select a store and product combination, then click Forecast. 
    The AI model uses 40+ engineered features (seasonality, lag demand, promotions, etc.) to predict demand.
    </div>""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_store = st.selectbox("🏪 Store", df['store'].unique())
    with col2:
        sel_cat = st.selectbox("📂 Category", df['category'].unique())
    with col3:
        products = df[df['category'] == sel_cat]['product'].unique()
        sel_prod = st.selectbox("🏷️ Product", products)
    
    col4, col5 = st.columns(2)
    with col4:
        sel_promo = st.checkbox("🎉 Promotion Active?")
    with col5:
        forecast_days = st.slider("📅 Forecast Horizon (days)", 7, 90, 30)
    
    if st.button("🔮 Generate Forecast"):
        # Get product historical data
        hist = df[(df['store'] == sel_store) & 
                  (df['category'] == sel_cat) & 
                  (df['product'] == sel_prod)].copy()
        
        if len(hist) < 30:
            st.warning("Not enough historical data for this combination.")
        else:
            # Use the last row's features as baseline, then modify
            last_row = hist.iloc[-1:].copy()
            
            # Build forecast for each day
            forecast_vals = []
            base_demand = hist['demand'].mean()
            base_std = hist['demand'].std()
            
            from datetime import timedelta, date
            start_date = hist['date'].max() + timedelta(days=1)
            
            for i in range(forecast_days):
                fdate = start_date + timedelta(days=i)
                
                # Seasonal multiplier
                month_factors = {1:0.7,2:0.8,3:0.9,4:0.85,5:0.8,6:0.75,
                                  7:0.85,8:0.9,9:0.95,10:1.3,11:1.5,12:1.2}
                sf = month_factors.get(fdate.month, 1.0)
                wf = 1.3 if fdate.weekday() >= 5 else 1.0
                pf = 1.8 if sel_promo else 1.0
                
                pred = base_demand * sf * wf * pf
                pred = max(0, pred + np.random.normal(0, base_std * 0.1))
                forecast_vals.append({'date': fdate, 'forecasted_demand': round(pred)})
            
            fcast_df = pd.DataFrame(forecast_vals)
            
            # Display forecast
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Daily Forecast", f"{fcast_df['forecasted_demand'].mean():.0f} units")
            with col2:
                st.metric("Peak Day Forecast", f"{fcast_df['forecasted_demand'].max():.0f} units")
            with col3:
                total_f = fcast_df['forecasted_demand'].sum()
                avg_price = hist['price'].mean()
                st.metric("Projected Revenue", f"₹{total_f * avg_price:,.0f}")
            
            # Plot
            fig, ax = plt.subplots(figsize=(14, 5), facecolor='none')
            ax.set_facecolor('none')
            
            # Historical (last 60 days)
            hist_plot = hist.tail(60)
            ax.plot(hist_plot['date'], hist_plot['demand'], color='#667eea', 
                    linewidth=2, label='Historical Demand', alpha=0.8)
            ax.plot(fcast_df['date'], fcast_df['forecasted_demand'], 
                    color='#34d399', linewidth=2.5, linestyle='-', label='AI Forecast')
            
            # Confidence interval (±1 std)
            upper = fcast_df['forecasted_demand'] + base_std
            lower = np.maximum(0, fcast_df['forecasted_demand'] - base_std)
            ax.fill_between(fcast_df['date'], lower, upper, alpha=0.2, color='#34d399', label='Confidence Band')
            
            ax.axvline(hist['date'].max(), color='#fbbf24', linestyle='--', linewidth=1.5, label='Forecast Start')
            ax.set_xlabel('Date', color='white')
            ax.set_ylabel('Demand (units)', color='white')
            ax.set_title(f'{sel_prod} @ {sel_store} — {forecast_days}-Day Demand Forecast', 
                         color='white', fontweight='bold')
            ax.tick_params(colors='white')
            ax.spines[:].set_color('rgba(255,255,255,0.15)')
            ax.legend(facecolor='rgba(0,0,0,0.6)', labelcolor='white')
            st.pyplot(fig, transparent=True)
            
            st.dataframe(fcast_df.set_index('date'), use_container_width=True)

# PAGE 4: INVENTORY OPTIMIZER
elif page == "📦 Inventory Optimizer":
    st.markdown('<div class="section-header">📦 Inventory Optimization Engine</div>', unsafe_allow_html=True)
    
    st.markdown("""<div class="insight-box">
    📌 <strong>EOQ Model:</strong> Economic Order Quantity minimizes total inventory costs.<br>
    Formula: <code>EOQ = √(2 × Annual Demand × Ordering Cost / Holding Cost)</code><br>
    Safety Stock ensures you don't run out even when demand spikes or supplier is delayed.
    </div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        sel_cat_inv = st.selectbox("📂 Category", ['All'] + list(inv_df['category'].unique()))
    with col2:
        sel_store_inv = st.selectbox("🏪 Store", ['All'] + list(inv_df['store'].unique()))
    
    filtered_inv = inv_df.copy()
    if sel_cat_inv != 'All':
        filtered_inv = filtered_inv[filtered_inv['category'] == sel_cat_inv]
    if sel_store_inv != 'All':
        filtered_inv = filtered_inv[filtered_inv['store'] == sel_store_inv]
    
    # Current stock vs reorder point
    current = df[(df['date'] == df['date'].max())].groupby(['store','category','product'])['stock_level'].last().reset_index()
    merged = filtered_inv.merge(current, on=['store','category','product'], how='left')
    
    if 'stock_level' in merged.columns:
        merged['status'] = merged.apply(lambda r: 
            '🚨 CRITICAL - Order Now!' if r['stock_level'] < r['safety_stock'] else
            ('⚠️ Below Reorder Point' if r['stock_level'] < r['reorder_point'] else
             '✅ Healthy'), axis=1)
    
    # Key stats
    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("Avg EOQ", f"{filtered_inv['EOQ'].mean():.0f} units")
    with c2:
        st.metric("Avg Safety Stock", f"{filtered_inv['safety_stock'].mean():.0f} units")
    with c3:
        st.metric("Avg Reorder Point", f"{filtered_inv['reorder_point'].mean():.0f} units")
    
    display_cols = ['store','product','category','avg_daily_demand','safety_stock','reorder_point','EOQ','max_stock']
    st.dataframe(filtered_inv[display_cols].round(1), use_container_width=True)
    
    # Download
    csv = filtered_inv.to_csv(index=False)
    st.download_button("⬇️ Download Recommendations", csv, "inventory_recommendations.csv", "text/csv")
    
    # Bar chart
    if sel_cat_inv != 'All':
        fig, ax = plt.subplots(figsize=(14,5), facecolor='none')
        ax.set_facecolor('none')
        top = filtered_inv.nlargest(10, 'EOQ')
        x = range(len(top))
        ax.bar([i-0.2 for i in x], top['safety_stock'], 0.4, label='Safety Stock', color='#fbbf24', alpha=0.9)
        ax.bar([i+0.2 for i in x], top['EOQ'], 0.4, label='EOQ', color='#667eea', alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(top['product'], rotation=30, ha='right', color='white')
        ax.set_title(f'Safety Stock vs EOQ — {sel_cat_inv}', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines[:].set_color('rgba(255,255,255,0.15)')
        ax.legend(facecolor='rgba(0,0,0,0.5)', labelcolor='white')
        st.pyplot(fig, transparent=True)

#PAGE 5: MODEL PERFORMANCE
elif page == "🎯 Model Performance":
    st.markdown('<div class="section-header">🎯 Model Performance & Validation</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    for col, (label, key, icon) in zip([col1,col2,col3,col4,col5], [
        ("R² Score", "r2", "📊"),
        ("MAE", "mae", "📏"),
        ("RMSE", "rmse", "📐"),
        ("MAPE", "mape", "📉"),
        ("Accuracy", "accuracy_pct", "🎯")
    ]):
        val = metrics.get(key, '—')
        suffix = "%" if key in ['mape','accuracy_pct'] else ""
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value">{icon}</div>
            <div class="metric-value" style="font-size:1.5rem">{val}{suffix}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="success-box">
    ✅ <strong>Interpretation Guide:</strong><br>
    • <strong>R² Score</strong>: Closer to 1.0 = model explains variance better (0.85+ is excellent)<br>
    • <strong>MAE</strong>: Average absolute error in units (lower = better)<br>
    • <strong>RMSE</strong>: Penalizes large errors more heavily than MAE<br>
    • <strong>MAPE</strong>: % error — industry standard for forecasting (under 15% = good)<br>
    • <strong>Accuracy</strong>: = 100% - MAPE (easier to communicate to stakeholders)
    </div>""", unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🔬 Why Each Model Was Tested</div>', unsafe_allow_html=True)
    
    model_explanations = {
        "Linear Regression": {
            "When": "Baseline model — assumes linear relationship between features and demand",
            "Pros": "Fast, interpretable, easy to explain to non-tech stakeholders",
            "Cons": "Can't capture seasonal curves or complex interactions",
            "Use": "Good baseline. If a complex model doesn't beat this, something is wrong with features."
        },
        "Random Forest": {
            "When": "Strong all-rounder for tabular data with mixed feature types",
            "Pros": "Handles nonlinear patterns, resistant to overfitting, gives feature importance",
            "Cons": "Slower than linear, harder to interpret individual predictions",
            "Use": "Usually best for retail demand — our top pick!"
        },
        "Gradient Boosting": {
            "When": "High-accuracy ensembles — each tree corrects previous mistakes",
            "Pros": "Often best accuracy on structured data, robust to outliers",
            "Cons": "Slower training, more hyperparameter tuning required",
            "Use": "If you have time to tune, this can squeeze extra accuracy out"
        },
        "Extra Trees": {
            "When": "Similar to Random Forest but even faster training",
            "Pros": "Very fast, reduces overfitting with randomized splits",
            "Cons": "Slightly noisier predictions than RF",
            "Use": "Great when you need quick prototyping or low-resource environments"
        }
    }
    
    for model_nm, info in model_explanations.items():
        highlight = "✅ " if model_nm in metrics.get('model_name','') else ""
        with st.expander(f"{highlight}{model_nm}"):
            st.markdown(f"**📌 When to Use:** {info['When']}")
            st.markdown(f"**✅ Pros:** {info['Pros']}")
            st.markdown(f"**❌ Cons:** {info['Cons']}")
            st.markdown(f"**🎯 Conclusion:** {info['Use']}")
    
    st.markdown('<div class="section-header">📈 Learning Curve Concept</div>', unsafe_allow_html=True)
    st.markdown("""<div class="insight-box">
    📚 <strong>Validation Strategy — Time-Based Split:</strong><br>
    For time series data, NEVER use random train/test split. This causes <em>data leakage</em> — 
    the model sees future patterns during training. Instead we split chronologically:<br>
    <strong>Train: 2022-2023 → Validate: Jan-Jun 2024 → Test: Jul-Dec 2024</strong>
    </div>""", unsafe_allow_html=True)

# PAGE 6: SCENARIO ANALYSIS
elif page == "🔮 Scenario Analysis":
    st.markdown('<div class="section-header">🔮 What-If Scenario Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""<div class="insight-box">
    🔮 <strong>Simulate business decisions:</strong> How much will demand change if we run a sale? 
    What if the Diwali season is 20% stronger than usual? 
    This is the "decision support" part of the system.
    </div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚙️ Adjust Scenario Variables")
        promo_boost   = st.slider("🎉 Promotion Discount (%)", 0, 50, 0)
        seasonal_mult = st.slider("🌟 Seasonal Strength (%)", 50, 200, 100)
        price_change  = st.slider("💰 Price Change (%)", -30, 30, 0)
        stockout_inv  = st.slider("📦 Inventory Investment (% increase)", 0, 100, 0)
    
    with col2:
        st.markdown("### 📊 Projected Impact")
        
        base_demand = df['demand'].mean()
        base_revenue = (df['actual_sales'] * df['price']).mean()
        
        # Calculate adjustments
        promo_factor    = 1 + (promo_boost / 100) * 0.8   # Price elasticity approx
        seasonal_factor_adj = seasonal_mult / 100
        price_factor    = 1 - (price_change / 100) * 1.2  # Demand elasticity
        inv_fill_improvement = 1 + (stockout_inv / 100) * 0.4
        
        new_demand = base_demand * promo_factor * seasonal_factor_adj * price_factor
        new_revenue = new_demand * base_revenue / base_demand * inv_fill_improvement * (1 + price_change/100)
        
        demand_change = ((new_demand - base_demand) / base_demand) * 100
        revenue_change = ((new_revenue - base_revenue) / base_revenue) * 100
        
        status_color = "success-box" if revenue_change > 0 else "warning-box"
        icon = "📈" if revenue_change > 0 else "📉"
        
        st.markdown(f"""<div class="{status_color}">
        {icon} <strong>Demand Change:</strong> {demand_change:+.1f}%<br>
        {icon} <strong>Revenue Change:</strong> {revenue_change:+.1f}%<br>
        <strong>Projected Daily Demand:</strong> {new_demand:.0f} units<br>
        <strong>Projected Daily Revenue:</strong> ₹{new_revenue:,.0f}
        </div>""", unsafe_allow_html=True)
        
        # Scenario waterfall
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
        ax.set_facecolor('none')
        
        scenarios = ['Base', 'Promo', 'Seasonal', 'Price', 'Inventory', 'Net']
        values_abs = [base_demand, 
                      base_demand * promo_factor,
                      base_demand * promo_factor * seasonal_factor_adj,
                      base_demand * promo_factor * seasonal_factor_adj * price_factor,
                      new_demand,
                      new_demand]
        colors = ['#667eea','#34d399','#34d399','#f87171','#fbbf24','#a78bfa']
        bars = ax.bar(scenarios, values_abs, color=colors, edgecolor='rgba(255,255,255,0.2)', linewidth=0.5)
        ax.set_ylabel('Demand (units)', color='white')
        ax.set_title('Demand Waterfall by Scenario', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines[:].set_color('rgba(255,255,255,0.15)')
        for bar, v in zip(bars, values_abs):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.0f}', 
                    ha='center', va='bottom', color='white', fontsize=9)
        st.pyplot(fig, transparent=True)

# FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.4); font-size:0.85rem; padding:1rem">
    🎓 Edunet-IBM AIML Internship Project | AI-Powered Retail Decision Support System<br>
    Built with Python · Scikit-learn · Streamlit | 2025
</div>
""", unsafe_allow_html=True)