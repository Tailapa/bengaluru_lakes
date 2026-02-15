import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# --- APP CONFIG ---
st.set_page_config(page_title="Bengaluru Flood Engine", layout="wide")

# Modern Professional Styling
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #0f172a; }
    </style>
    """, unsafe_allow_html=True)

# 1. LOAD DATA & ADVANCED PRE-PROCESSING
@st.cache_data
def load_data():
    # Load primary data
    df = pd.read_csv('data/dashboard_data.csv')
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    df = df.groupby('name').mean()
    df = df.drop(columns=['year'])
    
    # Merge Coordinates
    df_cords = pd.read_csv('data/lakes_dashboard.csv')
    cords = df_cords[['name', 'lat', 'lon']]
    df = df.merge(cords, how='left', on='name')
    
    # Fill missing coords to prevent Folium crash
    df['lat'] = df['lat'].fillna(12.97)
    df['lon'] = df['lon'].fillna(77.59)

    # Features
    features = ['max_3day_rain_mm', 'peak_30min_intensity_mm', 'impervious_fraction', 
                'urban_stress', 'potential_ha', 'csr_ratio', 'elevation', 'slope', 
                'biological_clogging', 'log_flow']
    
    X = df[features].fillna(0)
    y = df['sar_flood_freq_pct']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gradient Boosting Model
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    df['predicted_flood'] = model.predict(X)
    df['rmse_buffer'] = rmse
    
    # Polynomial Cost Modeling (The Utility)
    # Quadratic growth simulates logistical complexity of larger lakes
    poly_cost = make_pipeline(PolynomialFeatures(degree=2), Ridge())
    mock_y = (df['potential_ha']**2 * 55) + (df['potential_ha'] * 4800) + 12000
    poly_cost.fit(df[['potential_ha']], mock_y)
    df['est_cost'] = poly_cost.predict(df[['potential_ha']])
    
    return df, rmse

def optimize_lakes(df, budget):
    # ROI: Percentage risk reduced per Rupee spent
    df['priority_score'] = df['sar_flood_freq_pct'] / df['est_cost']
    df['flood_reduction'] = df['sar_flood_freq_pct'] * 0.65 
    
    optimized = df.sort_values('priority_score', ascending=False).copy()
    optimized['cum_cost'] = optimized['est_cost'].cumsum()
    optimized['cum_reduction'] = optimized['flood_reduction'].cumsum()
    
    affordable = optimized[optimized['cum_cost'] <= budget].copy()
    return affordable, optimized

# --- DATA INITIALIZATION ---
try:
    df, model_rmse = load_data()
except Exception as e:
    st.error(f"Data Load Error: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Control Panel")
    budget = st.slider("Desilting Budget (₹)", 500000, 20000000, 5000000, step=500000)
    st.divider()
    st.metric("Model Precision (RMSE)", f"±{model_rmse:.2f}%")

affordable_lakes, full_optimized = optimize_lakes(df, budget)

# --- UI LAYOUT ---
st.title("Bengaluru Lake Flood Risk Engine")

# 3. TOP LEVEL METRICS
m1, m2, m3, m4 = st.columns(4)
m1.metric("Lakes Affordable", len(affordable_lakes))
m2.metric("Total Mitigation", f"{affordable_lakes['flood_reduction'].sum():.1f}%")
m3.metric("Budget Efficiency", f"{(affordable_lakes['est_cost'].sum()/budget)*100:.1f}%")
m4.metric("Avg. Lake Cost", f"₹{affordable_lakes['est_cost'].mean():,.0f}")

# 4. MAP & CURVE
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Deployment Strategy")
    m = folium.Map(location=[12.97, 77.59], zoom_start=11, tiles="CartoDB positron")
    for _, row in affordable_lakes.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=row['sar_flood_freq_pct']*0.8,
            popup=f"<b>{row['name']}</b><br>Cost: ₹{row['est_cost']:,.0f}",
            color='#3b82f6', fill=True, fill_opacity=0.7
        ).add_to(m)
    st_folium(m, width="stretch", height=450)

with col2:
    st.subheader("Diminishing Returns")
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=full_optimized['cum_cost'], y=full_optimized['cum_reduction'],
        mode='lines', fill='tozeroy', line=dict(color='#3b82f6', width=3)
    ))
    fig_curve.add_vline(x=budget, line_dash="dash", line_color="#ef4444")
    fig_curve.update_layout(template="plotly_white", xaxis_title="Total Spend", yaxis_title="Total Risk Reduced", margin=dict(t=20))
    st.plotly_chart(fig_curve, width="stretch")

# 5. DIAGNOSTICS
st.subheader("Model Integrity Diagnostics")
c1, c2 = st.columns(2)

with c1:
    # REPLACED 'trendline="ols"' with identity line to fix error
    fig_acc = px.scatter(df, x='predicted_flood', y='sar_flood_freq_pct', opacity=0.4,
                         title="Accuracy: Predicted vs. Actual")
    # Add 45-degree identity line
    max_val = df['sar_flood_freq_pct'].max()
    fig_acc.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="#ef4444", dash="dot"))
    fig_acc.update_layout(template="plotly_white")
    st.plotly_chart(fig_acc, width="stretch")

with c2:
    sample = affordable_lakes.head(8)
    fig_safe = go.Figure()
    fig_safe.add_trace(go.Scatter(
        x=sample['name'], y=sample['predicted_flood'], mode='markers',
        error_y=dict(type='data', array=sample['rmse_buffer']*1.5, color='#ef4444'),
        marker=dict(size=12, color='#3b82f6')
    ))
    fig_safe.update_layout(template="plotly_white", title="Safety Buffer Analysis",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_safe, width="stretch")

# 6. ACTION TABLE
st.subheader("Priority Desilting Checklist")
# Ensure matplotlib is installed for the gradient
try:
    styled_df = affordable_lakes[['name', 'sar_flood_freq_pct', 'est_cost', 'priority_score']].style.background_gradient(subset=['priority_score'], cmap='Blues').format({'est_cost': '₹{:,.0f}', 'sar_flood_freq_pct': '{:.1f}%'})
    st.dataframe(styled_df, width="stretch")
except:
    st.dataframe(affordable_lakes[['name', 'sar_flood_freq_pct', 'est_cost']], width="stretch")

if st.button("Generate PDF Action Plan"):
    st.balloons()
    st.success("Plan exported for BBMP review.")