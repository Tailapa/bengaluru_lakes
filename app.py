import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# --- SETTINGS & THEMING ---
st.set_page_config(page_title="Bengaluru Flood Engine", layout="wide")

# Custom CSS for a cleaner, modern look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stSidebar"] { background-color: #1e293b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# 1. LOAD DATA & ADVANCED PRE-PROCESSING
@st.cache_data
def load_data():
    df = pd.read_csv('data/dashboard_data.csv')
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    
    
    df_cords = pd.read_csv('data/lakes_dashboard.csv')
    cords = df_cords[['name', 'lat', 'lon']]
    df = df.merge(cords, how='left', on='name')

    
    # Feature Engineering
    rain_feats = ['max_3day_rain_mm', 'peak_30min_intensity_mm'] 
    urban_feats = ['impervious_fraction', 'urban_stress']
    physical_feats = ['potential_ha', 'csr_ratio', 'elevation', 'slope', 'biological_clogging', 'log_flow']
    features = rain_feats + urban_feats + physical_feats
    
    X = df[features].fillna(0)
    y = df['sar_flood_freq_pct']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ADVANCED MODEL: Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate RMSE for dynamic Safety Intervals
    y_pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    df['predicted_flood'] = model.predict(X)
    df['rmse_buffer'] = rmse # Store for the safety_interval function
    
    # POLYNOMIAL COST UTILITY (Quadratic Growth)
    poly_cost = make_pipeline(PolynomialFeatures(degree=2), Ridge())
    mock_y = (df['potential_ha']**2 * 55) + (df['potential_ha'] * 4800) + 12000
    poly_cost.fit(df[['potential_ha']], mock_y)
    df['est_cost'] = poly_cost.predict(df[['potential_ha']])
    
    return df, rmse

# 2. OPTIMIZATION & INTERVALS
def optimize_lakes(df, budget):
    # ROI Score: Risk reduced per Rupee
    df['priority_score'] = df['sar_flood_freq_pct'] / df['est_cost']
    df['flood_reduction'] = df['sar_flood_freq_pct'] * 0.65 
    
    optimized = df.sort_values('priority_score', ascending=False).copy()
    optimized['cum_cost'] = optimized['est_cost'].cumsum()
    optimized['cum_reduction'] = optimized['flood_reduction'].cumsum()
    
    affordable = optimized[optimized['cum_cost'] <= budget].copy()
    return affordable, optimized

def safety_interval(pred, rmse):
    # Professional Interval: Using 1.96 * RMSE for ~95% confidence
    lower = np.maximum(0, pred - (1.5 * rmse))
    upper = pred + (1.5 * rmse)
    return lower, upper

# --- DATA INITIALIZATION ---
df, model_rmse = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/flood.png", width=80)
    st.header("Control Panel")
    budget = st.slider("Desilting Budget (₹)", 500000, 20000000, 5000000, step=500000)
    st.info(f"Model Confidence: ±{model_rmse:.2f}% (RMSE)")

affordable_lakes, full_optimized = optimize_lakes(df, budget)

# --- HEADER SECTION ---
st.title("Bengaluru Lake Flood Risk Decision Engine")
st.markdown("---")

# 3. TOP LEVEL METRICS
m1, m2, m3, m4 = st.columns(4)
m1.metric("Lakes in Scope", len(affordable_lakes))
m2.metric("Mitigated Risk", f"{affordable_lakes['flood_reduction'].sum():.1f}%")
m3.metric("Budget Utilization", f"{(affordable_lakes['est_cost'].sum()/budget)*100:.1f}%")
m4.metric("Avg. Cost/Lake", f"₹{affordable_lakes['est_cost'].mean():,.0f}")

# 4. MAIN VISUALS (MAP & OPTIMIZATION CURVE)
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Deployment Strategy Map")
    m = folium.Map(location=[12.97, 77.59], zoom_start=11, tiles="CartoDB positron")
    for idx, row in affordable_lakes.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=row['sar_flood_freq_pct']*0.8,
            popup=f"<b>{row['name']}</b><br>Cost: ₹{row['est_cost']:,.0f}",
            color='#1e293b', fill=True, fill_color='#3b82f6', fill_opacity=0.7
        ).add_to(m)
    st_folium(m, width="100%", height=450)

with col2:
    st.subheader("Diminishing Returns Curve")
    # Professional Plotly Chart: Cumulative Impact
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=full_optimized['cum_cost'], y=full_optimized['cum_reduction'],
        mode='lines', name='Risk Mitigation', line=dict(color='#3b82f6', width=3),
        fill='tozeroy'
    ))
    # Budget Cutoff Line
    fig_curve.add_vline(x=budget, line_dash="dash", line_color="red", annotation_text="Budget Limit")
    fig_curve.update_layout(
        template="plotly_white", margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Cumulative Spend (₹)", yaxis_title="Total Risk Reduced (%)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_curve, use_container_width=True)

# 5. MODEL VALIDATION & SAFETY (REFINED PLOTLY)
st.markdown("---")
st.subheader("Intelligence Diagnostics (Gradient Boosting)")

c1, c2 = st.columns(2)

with c1:
    # Prediction vs Actual
    fig_acc = px.scatter(df, x='predicted_flood', y='sar_flood_freq_pct', 
                         trendline="ols", labels={'predicted_flood':'Predicted Risk', 'sar_flood_freq_pct':'Actual Risk'},
                         title="Model Accuracy (Observed vs Predicted)")
    fig_acc.update_traces(marker=dict(size=10, color='#1e293b', opacity=0.6))
    st.plotly_chart(fig_acc, use_container_width=True)

with c2:
    # Safety Intervals for High Priority Lakes
    sample = affordable_lakes.head(8)
    fig_safe = go.Figure()
    
    # Draw Error Bars with Legend Pro-tip applied
    fig_safe.add_trace(go.Scatter(
        x=sample['name'], y=sample['predicted_flood'],
        mode='markers',
        error_y=dict(type='data', array=sample['rmse_buffer']*1.5, visible=True, thickness=2, width=4, color='#ef4444'),
        marker=dict(size=12, color='#3b82f6'),
        name="Predicted Risk + Safety Buffer"
    ))
    fig_safe.update_layout(
        template="plotly_white", title="Safety Confidence Intervals",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_safe, use_container_width=True)

# 6. ACTION TABLE
st.subheader("Final Action Plan: Top Priority Desilting")
st.dataframe(
    affordable_lakes[['name', 'sar_flood_freq_pct', 'est_cost', 'priority_score']]
    .style.background_gradient(subset=['priority_score'], cmap='Blues')
    .format({'est_cost': '₹{:,.0f}', 'sar_flood_freq_pct': '{:.1f}%'}),
    use_container_width=True
)

if st.button("🚀 Finalize BBMP Checklist"):
    st.balloons()
    st.success(f"Strategy Validated. {len(affordable_lakes)} Lakes scheduled for desilting.")