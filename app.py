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

# Modern Professional Styling (CSS)
st.markdown("""
    <style>
    /* Hides the GitHub icon and the main menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hides the footer (Made with Streamlit) */
    footer {visibility: hidden;}
    
    /* Optional: If you want to keep the menu but hide ONLY the 'View Source' button */
    .viewerBadge_link__1S137 {display: none;} 
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #3b82f6 !important;
    }
    [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .stSlider [data-baseweb="slider"] > div > div {
        background-color: #3b82f6 !important;
    }
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
        color: #ffffff !important;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 1rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #10b981 !important;
        font-weight: 800 !important;
    }
            
    /* Target the button containing the help icon inside a metric */
    [data-testid="stMetric"] [data-testid="stMetricHelp"] button svg {
        fill: #ef4444 !important; /* Change to your desired color */
        color: #ef4444 !important;
    }

    /* Target the path specifically if the SVG fill isn't responding */
    [data-testid="stMetric"] [data-testid="stMetricHelp"] button svg path {
        fill: #ef4444 !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# 1. LOAD DATA & ADVANCED PRE-PROCESSING
@st.cache_data
def load_data():
    # Load primary data
    df_raw = pd.read_csv('data/dashboard_data.csv')
    df_raw = df_raw.loc[:, ~df_raw.columns.str.contains('Unnamed')]
    
    # Aggregate by lake name to prevent duplicates (averaging features)
    df = df_raw.groupby('name').mean(numeric_only=True).reset_index()
    if 'year' in df.columns:
        df = df.drop(columns=['year'])
    
    # Merge Coordinates (Single record per lake)
    df_cords = pd.read_csv('data/lakes_dashboard.csv')
    cords = df_cords[['name', 'lat', 'lon']].drop_duplicates(subset=['name'])
    df = df.merge(cords, how='left', on='name')
    
    # Fill missing coords
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
    
    # Feature Importance for Decision Transparency
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    
    # Polynomial Cost Modeling
    poly_cost = make_pipeline(PolynomialFeatures(degree=2), Ridge())
    mock_y = (df['potential_ha']**2 * 55) + (df['potential_ha'] * 4800) + 12000
    poly_cost.fit(df[['potential_ha']], mock_y)
    df['est_cost'] = poly_cost.predict(df[['potential_ha']])
    
    return df, rmse, importance

def optimize_lakes(df, budget):
    # ROI: Priority Score scaled for readability
    df['priority_score'] = (df['sar_flood_freq_pct'] / df['est_cost']) * 10000
    df['flood_reduction'] = df['sar_flood_freq_pct'] * 0.65 
    
    optimized = df.sort_values('priority_score', ascending=False).copy()
    optimized['cum_cost'] = optimized['est_cost'].cumsum()
    optimized['cum_reduction'] = optimized['flood_reduction'].cumsum()
    
    affordable = optimized[optimized['cum_cost'] <= budget].copy()
    return affordable, optimized

# --- DATA INITIALIZATION ---
try:
    df, model_rmse, feat_importance = load_data()
except Exception as e:
    st.error(f"Data Load Error: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Control Panel")
    budget = st.slider("Revival Budget (₹)", 500000, 20000000, 5000000, step=500000)
    st.divider()
    st.metric(label="Model Precision (RMSE)", 
              value=f"± {model_rmse:.2f}%",
              help="Root Mean Square Error: The average difference between the model's predicted flood risk and the actual observed risk.")
    
    # st.write("Top Risk Drivers")
    # st.bar_chart(feat_importance.head(5))

    st.markdown("<p style='color: #ffffff; font-size: 14px; font-weight: bold; margin-bottom: 0px;'>Top Risk Drivers</p>", unsafe_allow_html=True)
    
    # 1. Define the Mapping Dictionary
    label_map = {
        "peak_30min_intensity_mm": "Rainfall Intensity",
        "potential_ha": "Surface Area",
        "sar_flood_freq_pct": "Flood Risk",
        "urban_stress": "Urban Pressure",
        "impervious_fraction": "Concrete Cover",
        "priority_score": "Priority Rank",
        "biological_clogging": "Weed Encroachment"
    }

    # 2. Prepare and Rename Data
    top_feats = feat_importance.head(5).reset_index()
    top_feats.columns = ['Feature', 'Importance']
    
    # Apply the human-readable names
    top_feats['Feature'] = top_feats['Feature'].replace(label_map)

    # 3. Create the Clean Plotly Chart
    fig = px.bar(
        top_feats, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        template='plotly_dark'
    )

    fig.update_traces(
        marker_color='#94a3b8', 
        marker_line_color='rgba(0,0,0,0)',
        hovertemplate='%{x:.2f}<extra></extra>'
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',  
        margin=dict(l=0, r=10, t=10, b=0), 
        height=220, # Slightly taller for longer labels
        xaxis=dict(showgrid=False, showticklabels=False, title=''), 
        yaxis=dict(
            showgrid=False, 
            title='', 
            tickfont=dict(color="#ffffff", size=11),
            autorange="reversed" # Keeps the highest driver at the top
        ), 
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

affordable_lakes, full_optimized = optimize_lakes(df, budget)

# --- UI LAYOUT ---
st.title("Bengaluru Lake Restoration: Decision Support Dashboard")

# 3. TOP LEVEL METRICS

# Calculate Total Risk across the entire dataset
total_starting_risk = df['sar_flood_freq_pct'].sum()

# Calculate Total Risk Reduced from selected lakes
total_reduction = affordable_lakes['flood_reduction'].sum()

# Calculate the Mitigation Percentage
# This represents: (Risk Removed / Total Possible Risk)
mitigation_ratio = (total_reduction / total_starting_risk) * 100 if total_starting_risk > 0 else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    label="Lakes Affordable", 
    value=len(affordable_lakes),
    help="The number of lakes that can be fully desilted within the current budget allocation based on estimated costs."
)

m2.metric(
    label="Risk Mitigation Share", 
    value=f"{mitigation_ratio:.1f}%", 
    help="Percentage of city-wide flood risk mitigated by the selected budget. Calculated as (Total Risk Reduced / Total Possible Risk)."
)

m3.metric(
    label="Budget Efficiency", 
    value=f"{(affordable_lakes['est_cost'].sum()/budget)*100:.1f}%",
    help="The percentage of the total allocated budget that can be successfully utilized by our targeted priority lakes."
)

m4.metric(
    label="Average Lake Cost", 
    value=f"₹ {affordable_lakes['est_cost'].mean():,.0f}",
    help="The mean estimated cost for reviving a single lake within our targeted list of lakes."
)

# 4. MAP & CURVE
col1, col2 = st.columns([1.5, 1])

import folium
from streamlit_folium import st_folium

with col1:
    st.subheader("Geospatial Lake Identification Strategy")
    m = folium.Map(location=[12.97, 77.59], zoom_start=11, tiles="CartoDB positron")
    
    # 1. Add Legend HTML/CSS
    legend_html = '''
     <div style="
     position: fixed; 
     bottom: 20px; left: 20px; width: 170px; height: 100px; 
     background-color: white; border:2px solid grey; z-index:9999; font-size:12px;
     border-radius: 6px; padding: 10px;
     ">
     <b>Map Legend</b><br>
     <i class="fa fa-circle" style="color:#ef4444"></i>&nbsp; High Flood Risk (>20%)<br>
     <i class="fa fa-circle" style="color:#22c55e"></i>&nbsp; High Priority Rank (>4)<br>
     <i class="fa fa-circle" style="color:#3b82f6"></i>&nbsp; Standard Priority
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    for _, row in affordable_lakes.iterrows():
        # 2. Dynamic color logic based on your new rules
        if row['sar_flood_freq_pct'] > 20:
            marker_color = "#ef4444"  # Red
        elif row['priority_score'] > 4:
            marker_color = "#22c55e"  # Green
        else:
            marker_color = "#3b82f6"  # Default Blue
        
        popup_html = f"""
            <div style="font-family: sans-serif; width: 200px;">
                <h4 style="margin-bottom: 5px;">{row['name']}</h4>
                <hr style="margin: 5px 0;">
                <table style="width: 100%; font-size: 12px;">
                    <tr><td><b>Risk Score:</b></td><td style="text-align:right;">{row['sar_flood_freq_pct']:.1f}%</td></tr>
                    <tr><td><b>Priority Rank:</b></td><td style="text-align:right;">{row['priority_score']:.2f}</td></tr>
                    <tr style="color: #3b82f6;"><td><b>Est. Cost:</b></td><td style="text-align:right;">₹{row['est_cost']:,.0f}</td></tr>
                </table>
            </div>
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=row['sar_flood_freq_pct'] * 0.7,
            popup=folium.Popup(popup_html, max_width=250),
            color=marker_color, 
            fill=True, 
            fill_opacity=0.6,
            weight=2
        ).add_to(m)
        
    st_folium(m, width=700, height=450)

with col2:
    st.subheader("Diminishing Returns Analysis")
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=full_optimized['cum_cost'], y=full_optimized['cum_reduction'],
        mode='lines', fill='tozeroy', line=dict(color='#3b82f6', width=3)
    ))
    fig_curve.add_vline(x=budget, line_dash="dash", line_color="#ef4444")
    fig_curve.update_layout(
        template="plotly_white", 
        xaxis_title="Cumulative Spend (₹)", 
        yaxis_title="Total Mitigation (%)",
        margin=dict(t=20, l=10, r=10, b=10)
    )
    st.plotly_chart(fig_curve, width="stretch")

# 5. DIAGNOSTICS
st.subheader("Model Integrity Diagnostics")
c1, c2 = st.columns(2)

with c1:
    fig_acc = px.scatter(df, x='predicted_flood', y='sar_flood_freq_pct', opacity=0.5,
                         title="Accuracy: Predicted vs. Actual Risk")
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
    fig_safe.update_layout(template="plotly_white", title="Safety Confidence Intervals")
    st.plotly_chart(fig_safe, width="stretch")

# 6. ACTION TABLE
st.subheader("Priority Lakes (low hanging fruits)")

# Process table data
final_display = affordable_lakes[['name', 'sar_flood_freq_pct', 'est_cost', 'priority_score']].drop_duplicates(subset=['name']).reset_index(drop=True)
final_display = final_display.rename(columns={
    'name': 'Lake Name',
    'sar_flood_freq_pct': 'Flood Risk Percent',
    'est_cost': 'Estimated Cost',
    'priority_score': 'Priority Rank'
})

st.dataframe(
    final_display,
    column_config={
        "Priority Rank": st.column_config.ProgressColumn(
            "Priority Rank",
            help="Higher value indicates higher restoration priority",
            format="%.2f",
            min_value=0,
            max_value=5, # or final_display["Priority Rank"].max()
            color="blue" # Matches your 'Blues' cmap preference
        ),
        "Estimated Cost": st.column_config.NumberColumn(
            "Estimated Cost",
            format="₹ %d",
        ),
        "Flood Risk Percent": st.column_config.NumberColumn(
            "Flood Risk",
            format="%.1f%%",
        )
    },
    hide_index=True,
    use_container_width=True
)

if st.button("Finalize Action Plan"):
    st.balloons()
    st.success("Yet to be formulated")