import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
import sqlite3
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Professional page configuration
st.set_page_config(
    page_title="AI4I Industrial Analytics Platform",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling - Dark Industrial Theme
st.markdown("""
<style>
    /* Dark Industrial Theme Variables */
    :root {
        --primary-blue: #3b82f6;
        --bg-dark: #1e293b;
        --bg-secondary: #334155;
        --text-light: #f1f5f9;
        --accent-green: #059669;
        --warning-orange: #d97706;
        --critical-red: #dc2626;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: var(--text-light);
        margin-bottom: 1.5rem;
        border-bottom: 3px solid var(--primary-blue);
        padding-bottom: 0.8rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-light);
        margin: 1rem 0 0.5rem 0;
        border-left: 4px solid var(--primary-blue);
        padding-left: 1rem;
        background: rgba(59, 130, 246, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 4px;
    }
    .metric-container {
        background: linear-gradient(135deg, var(--primary-blue) 0%, #1e40af 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .alert-critical {
        background: linear-gradient(135deg, var(--critical-red) 0%, #991b1b 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 6px solid #7f1d1d;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(220, 38, 38, 0.3);
    }
    .alert-warning {
        background: linear-gradient(135deg, var(--warning-orange) 0%, #92400e 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 6px solid #78350f;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(217, 119, 6, 0.3);
    }
    .alert-normal {
        background: linear-gradient(135deg, var(--accent-green) 0%, #047857 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 6px solid #065f46;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(5, 150, 105, 0.3);
    }
    .equipment-card {
        background: var(--bg-secondary);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        color: var(--text-light);
    }
    .sidebar .metric-container {
        background: var(--bg-secondary);
        color: var(--text-light);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-left: 4px solid var(--primary-blue);
        margin: 0.5rem 0;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Dark theme enhancements */
    .stMetric {
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Custom plotly dark theme integration */
    .js-plotly-plot {
        background-color: var(--bg-dark) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data persistence
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.last_refresh = datetime.now()
    
# Professional sidebar
st.sidebar.markdown("### Control Center")
st.sidebar.markdown("---")

# System status in sidebar
st.sidebar.markdown("**System Overview**")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Equipment", "25", "Active")
with col2:
    st.metric("Alerts", "7", "Open")

st.sidebar.metric("Efficiency", "94.2%", "+2.1%")
st.sidebar.metric("Uptime", "99.7%", "Target: 99.5%")

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox("Select Module", [
    "Executive Dashboard",
    "Real-Time Operations", 
    "Predictive Intelligence",
    "Asset Management",
    "Quality Analytics",
    "Maintenance Operations",
    "Financial Analysis",
    "Advanced Analytics",
    "Anomaly Detection",
    "Performance Optimization",
    "Risk Assessment",
    "Compliance Monitoring"
])

# Advanced filters
st.sidebar.markdown("**Analysis Controls**")
time_range = st.sidebar.selectbox("Time Range", 
    ["Last Hour", "Last 4 Hours", "Last 24 Hours", "Last 7 Days", "Last 30 Days"])
facility_filter = st.sidebar.multiselect("Facility", 
    ["All Facilities", "Plant A", "Plant B", "Plant C"], default=["All Facilities"])
equipment_class = st.sidebar.selectbox("Equipment Class", 
    ["All Classes", "Critical", "Production", "Auxiliary", "Safety"])

if st.sidebar.button("Refresh Data"):
    st.session_state.last_refresh = datetime.now()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Last Updated:** {st.session_state.last_refresh.strftime('%H:%M:%S')}")

# Advanced data generation functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_comprehensive_data():
    """Generate comprehensive industrial data"""
    np.random.seed(42)
    
    # Equipment fleet
    equipment_types = ['Centrifugal Pump', 'Motor Drive', 'Compressor', 'Heat Exchanger', 
                      'Turbine', 'Generator', 'Conveyor', 'Reactor', 'Separator', 'Valve Assembly']
    
    equipment_data = []
    for i in range(50):  # Larger fleet
        eq_id = f"EQ-{i+1:04d}"
        eq_type = np.random.choice(equipment_types)
        
        # Realistic operating parameters
        base_temp = 320 + np.random.normal(0, 20)
        base_pressure = 15 + np.random.normal(0, 3)  # bar
        base_flow = 100 + np.random.normal(0, 15)    # m3/h
        base_power = 50 + np.random.normal(0, 10)    # kW
        
        # Equipment condition
        age_years = np.random.randint(1, 15)
        condition_score = max(0, 100 - age_years * 3 - np.random.normal(0, 10))
        
        # Operational status
        if condition_score < 60:
            status = "Critical"
            availability = np.random.uniform(0.7, 0.85)
        elif condition_score < 75:
            status = "Warning"
            availability = np.random.uniform(0.85, 0.95)
        else:
            status = "Normal"
            availability = np.random.uniform(0.95, 0.99)
        
        equipment_data.append({
            'Equipment_ID': eq_id,
            'Type': eq_type,
            'Location': f"Zone {chr(65 + i//10)}",
            'Temperature': base_temp,
            'Pressure': base_pressure,
            'Flow_Rate': base_flow,
            'Power_Consumption': base_power,
            'Vibration': 20 + np.random.normal(0, 5),
            'Condition_Score': condition_score,
            'Status': status,
            'Availability': availability,
            'Age_Years': age_years,
            'Last_Maintenance': datetime.now() - timedelta(days=np.random.randint(1, 365)),
            'MTBF_Hours': np.random.randint(2000, 8000),
            'Operating_Hours': np.random.randint(1000, 8760),
            'Efficiency': min(98, availability * 100 + np.random.normal(0, 2))
        })
    
    return pd.DataFrame(equipment_data)

@st.cache_data(ttl=300)
def generate_time_series_data(hours=168):  # 7 days
    """Generate detailed time series data"""
    timestamps = pd.date_range(start=datetime.now() - timedelta(hours=hours), 
                              end=datetime.now(), freq='H')
    
    time_series = []
    equipment_ids = [f"EQ-{i+1:04d}" for i in range(10)]  # Focus on 10 equipment
    
    for eq_id in equipment_ids:
        base_temp = 320 + np.random.normal(0, 10)
        base_pressure = 15 + np.random.normal(0, 2)
        
        for i, ts in enumerate(timestamps):
            # Add daily and weekly patterns
            daily_pattern = 10 * np.sin(2 * np.pi * i / 24)
            weekly_pattern = 5 * np.sin(2 * np.pi * i / (24 * 7))
            noise = np.random.normal(0, 2)
            
            time_series.append({
                'Equipment_ID': eq_id,
                'Timestamp': ts,
                'Temperature': base_temp + daily_pattern + weekly_pattern + noise,
                'Pressure': base_pressure + daily_pattern/2 + noise/2,
                'Vibration': 20 + daily_pattern/3 + noise,
                'Power': 50 + daily_pattern + noise,
                'Flow_Rate': 100 + daily_pattern*2 + noise*2
            })
    
    return pd.DataFrame(time_series)

@st.cache_data(ttl=300)
def generate_maintenance_data():
    """Generate maintenance history and schedule"""
    maintenance_types = ['Preventive', 'Predictive', 'Corrective', 'Emergency', 'Overhaul']
    technicians = ['Anderson, J.', 'Rodriguez, M.', 'Chen, L.', 'Johnson, R.', 'Smith, K.']
    
    maintenance_data = []
    for i in range(100):
        equipment_id = f"EQ-{np.random.randint(1, 51):04d}"
        maintenance_type = np.random.choice(maintenance_types, 
                                          p=[0.4, 0.25, 0.2, 0.1, 0.05])
        
        # Past maintenance
        if i < 70:
            date = datetime.now() - timedelta(days=np.random.randint(1, 365))
            status = np.random.choice(['Completed', 'Completed', 'Completed', 'Cancelled'], 
                                    p=[0.85, 0.1, 0.03, 0.02])
        else:  # Future maintenance
            date = datetime.now() + timedelta(days=np.random.randint(1, 90))
            status = np.random.choice(['Scheduled', 'Planned', 'Pending Approval'])
        
        duration = np.random.randint(2, 24)  # hours
        cost = np.random.randint(500, 15000)  # USD
        
        maintenance_data.append({
            'Work_Order': f"WO-{i+1:06d}",
            'Equipment_ID': equipment_id,
            'Type': maintenance_type,
            'Scheduled_Date': date,
            'Duration_Hours': duration,
            'Technician': np.random.choice(technicians),
            'Status': status,
            'Cost_USD': cost,
            'Priority': np.random.choice(['Low', 'Medium', 'High', 'Critical'], 
                                       p=[0.3, 0.4, 0.25, 0.05]),
            'Description': f"{maintenance_type} maintenance on {equipment_id}"
        })
    
    return pd.DataFrame(maintenance_data)

def perform_anomaly_detection(data):
    """Advanced anomaly detection using Isolation Forest"""
    # Select numeric columns for anomaly detection - match time series column names
    numeric_cols = ['Temperature', 'Pressure', 'Vibration', 'Power', 'Flow_Rate']
    available_cols = [col for col in numeric_cols if col in data.columns]
    
    if len(available_cols) == 0:
        # Fallback if no matching columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        available_cols = numeric_cols[:5]  # Take first 5 numeric columns
    
    X = data[available_cols].dropna()
    
    if len(X) == 0:
        # Return dummy data if no valid rows
        return np.array([1] * len(data)), np.array([0] * len(data))
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest for anomaly detection
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(X_scaled)
    
    # Add anomaly scores
    anomaly_scores = iso_forest.decision_function(X_scaled)
    
    # Pad results to match original data length
    if len(anomalies) < len(data):
        padding_length = len(data) - len(anomalies)
        anomalies = np.concatenate([anomalies, np.array([1] * padding_length)])
        anomaly_scores = np.concatenate([anomaly_scores, np.array([0] * padding_length)])
    
    return anomalies, anomaly_scores

# Load comprehensive data
equipment_df = generate_comprehensive_data()
time_series_df = generate_time_series_data()
maintenance_df = generate_maintenance_data()

# Advanced analytics calculations
total_equipment = len(equipment_df)
critical_equipment = len(equipment_df[equipment_df['Status'] == 'Critical'])
avg_availability = equipment_df['Availability'].mean()
total_power = equipment_df['Power_Consumption'].sum()
avg_efficiency = equipment_df['Efficiency'].mean()

# Main content based on selected page
if page == "Executive Dashboard":
    st.markdown('<h1 class="main-header">AI4I Industrial Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Executive KPIs
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Fleet Size", f"{total_equipment}", "Assets")
    with col2:
        st.metric("Availability", f"{avg_availability:.1%}", "+1.2%")
    with col3:
        st.metric("Overall Efficiency", f"{avg_efficiency:.1f}%", "+0.8%")
    with col4:
        st.metric("Critical Assets", f"{critical_equipment}", f"-{2} vs last week")
    with col5:
        st.metric("Power Consumption", f"{total_power:.0f} kW", "-3.2%")
    with col6:
        annual_savings = 2.3
        st.metric("Annual Savings", f"${annual_savings:.1f}M", "+12%")
    
    # Executive level charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Asset Health Distribution</div>', unsafe_allow_html=True)
        
        # Asset health by condition score
        health_bins = pd.cut(equipment_df['Condition_Score'], 
                           bins=[0, 60, 75, 90, 100], 
                           labels=['Critical', 'Poor', 'Good', 'Excellent'])
        health_counts = health_bins.value_counts()
        
        fig_health = px.pie(values=health_counts.values, names=health_counts.index,
                           color_discrete_map={'Critical': '#dc2626', 'Poor': '#d97706', 
                                             'Good': '#059669', 'Excellent': '#2563eb'})
        fig_health.update_layout(height=400)
        st.plotly_chart(fig_health, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">Equipment Performance Matrix</div>', unsafe_allow_html=True)
        
        # Performance vs Age analysis
        fig_performance = px.scatter(equipment_df, x='Age_Years', y='Efficiency', 
                                   color='Status', size='Power_Consumption',
                                   hover_data=['Equipment_ID', 'Type'],
                                   color_discrete_map={'Normal': '#059669', 'Warning': '#d97706', 'Critical': '#dc2626'})
        fig_performance.update_layout(height=400)
        st.plotly_chart(fig_performance, use_container_width=True)
    
    # Advanced analytics section
    st.markdown('<div class="section-header">Operational Intelligence</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Mean Time Between Failures")
        avg_mtbf = equipment_df['MTBF_Hours'].mean()
        mtbf_by_type = equipment_df.groupby('Type')['MTBF_Hours'].mean().sort_values(ascending=False)
        
        fig_mtbf = px.bar(x=mtbf_by_type.values, y=mtbf_by_type.index, orientation='h')
        fig_mtbf.update_layout(height=350, xaxis_title="MTBF (Hours)")
        st.plotly_chart(fig_mtbf, use_container_width=True)
    
    with col2:
        st.subheader("Power Consumption Analysis")
        power_by_zone = equipment_df.groupby('Location')['Power_Consumption'].sum()
        
        fig_power = px.bar(x=power_by_zone.index, y=power_by_zone.values)
        fig_power.update_layout(height=350, yaxis_title="Power (kW)")
        st.plotly_chart(fig_power, use_container_width=True)
    
    with col3:
        st.subheader("Availability Trends")
        # Create weekly availability trend
        weeks = pd.date_range(start='2024-01-01', end='2024-09-23', freq='W')
        availability_trend = 0.94 + 0.02 * np.sin(np.linspace(0, 4*np.pi, len(weeks))) + np.random.normal(0, 0.01, len(weeks))
        
        fig_avail = px.line(x=weeks, y=availability_trend)
        fig_avail.update_layout(height=350, yaxis_title="Availability %")
        st.plotly_chart(fig_avail, use_container_width=True)
    
    # Critical equipment alerts
    st.markdown('<div class="section-header">Critical Equipment Alerts</div>', unsafe_allow_html=True)
    
    critical_eq = equipment_df[equipment_df['Status'] == 'Critical'].head(5)
    for _, eq in critical_eq.iterrows():
        st.markdown(f"""
        <div class="alert-critical">
            <strong>{eq['Equipment_ID']} - {eq['Type']}</strong><br>
            Condition Score: {eq['Condition_Score']:.1f}% | Availability: {eq['Availability']:.1%}<br>
            Location: {eq['Location']} | Age: {eq['Age_Years']} years
        </div>
        """, unsafe_allow_html=True)

elif page == "Real-Time Operations":
    st.markdown('<h1 class="main-header">Real-Time Operations Center</h1>', unsafe_allow_html=True)
    
    # Real-time equipment selection
    col1, col2 = st.columns([3, 1])
    
    with col2:
        selected_equipment = st.selectbox("Monitor Equipment", equipment_df['Equipment_ID'].tolist())
        
        # Equipment details
        eq_details = equipment_df[equipment_df['Equipment_ID'] == selected_equipment].iloc[0]
        
        st.markdown("**Equipment Details**")
        st.write(f"Type: {eq_details['Type']}")
        st.write(f"Location: {eq_details['Location']}")
        st.write(f"Age: {eq_details['Age_Years']} years")
        st.write(f"Status: {eq_details['Status']}")
        
        # Current readings
        st.metric("Temperature", f"{eq_details['Temperature']:.1f}°C")
        st.metric("Pressure", f"{eq_details['Pressure']:.1f} bar")
        st.metric("Vibration", f"{eq_details['Vibration']:.1f} mm/s")
        st.metric("Power", f"{eq_details['Power_Consumption']:.1f} kW")
    
    with col1:
        # Real-time trend analysis
        eq_time_data = time_series_df[time_series_df['Equipment_ID'] == selected_equipment]
        
        # Create subplots for multiple parameters
        fig_trends = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature Trend', 'Pressure Trend', 'Vibration Trend', 'Power Consumption'),
            vertical_spacing=0.1
        )
        
        # Temperature
        fig_trends.add_trace(
            go.Scatter(x=eq_time_data['Timestamp'], y=eq_time_data['Temperature'], 
                      name="Temperature", line=dict(color='#dc2626')),
            row=1, col=1
        )
        
        # Pressure
        fig_trends.add_trace(
            go.Scatter(x=eq_time_data['Timestamp'], y=eq_time_data['Pressure'], 
                      name="Pressure", line=dict(color='#2563eb')),
            row=1, col=2
        )
        
        # Vibration
        fig_trends.add_trace(
            go.Scatter(x=eq_time_data['Timestamp'], y=eq_time_data['Vibration'], 
                      name="Vibration", line=dict(color='#059669')),
            row=2, col=1
        )
        
        # Power
        fig_trends.add_trace(
            go.Scatter(x=eq_time_data['Timestamp'], y=eq_time_data['Power'], 
                      name="Power", line=dict(color='#d97706')),
            row=2, col=2
        )
        
        fig_trends.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_trends, use_container_width=True)
    
    # Fleet overview
    st.markdown('<div class="section-header">Fleet Status Overview</div>', unsafe_allow_html=True)
    
    # Real-time status grid
    status_summary = equipment_df.groupby(['Location', 'Status']).size().unstack(fill_value=0)
    
    fig_heatmap = px.imshow(status_summary.values, 
                           x=status_summary.columns, 
                           y=status_summary.index,
                           aspect="auto",
                           color_continuous_scale="RdYlGn_r")
    fig_heatmap.update_layout(title="Equipment Status by Location")
    st.plotly_chart(fig_heatmap, use_container_width=True)

elif page == "Predictive Intelligence":
    st.markdown('<h1 class="main-header">Predictive Intelligence Engine</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">Failure Prediction Model</div>', unsafe_allow_html=True)
        
        # Advanced prediction model simulation
        equipment_risk = equipment_df.copy()
        
        # Calculate risk scores based on multiple factors
        equipment_risk['Risk_Score'] = (
            (100 - equipment_risk['Condition_Score']) * 0.4 +
            (equipment_risk['Age_Years'] / 15 * 100) * 0.3 +
            ((equipment_risk['Temperature'] - 320) / 50 * 100).clip(0, 100) * 0.2 +
            (equipment_risk['Vibration'] / 30 * 100) * 0.1
        ).clip(0, 100)
        
        # Risk categorization
        equipment_risk['Risk_Category'] = pd.cut(equipment_risk['Risk_Score'], 
                                               bins=[0, 25, 50, 75, 100],
                                               labels=['Low', 'Medium', 'High', 'Critical'])
        
        # Failure probability over time
        risk_timeline = equipment_risk.groupby('Risk_Category').size()
        
        fig_risk = px.bar(x=risk_timeline.index, y=risk_timeline.values,
                         color=risk_timeline.index,
                         color_discrete_map={'Low': '#059669', 'Medium': '#d97706', 
                                           'High': '#dc2626', 'Critical': '#7c2d12'})
        fig_risk.update_layout(title="Equipment Risk Distribution")
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Feature importance analysis
        st.subheader("Model Feature Importance")
        features = ['Condition Score', 'Age', 'Temperature', 'Vibration', 'Pressure', 'Operating Hours']
        importance = [0.35, 0.25, 0.18, 0.12, 0.06, 0.04]
        
        fig_importance = px.bar(x=importance, y=features, orientation='h')
        fig_importance.update_layout(height=300)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">Prediction Controls</div>', unsafe_allow_html=True)
        
        # Interactive prediction
        selected_eq = st.selectbox("Select Equipment for Analysis", equipment_df['Equipment_ID'].tolist())
        
        eq_data = equipment_df[equipment_df['Equipment_ID'] == selected_eq].iloc[0]
        
        # Parameter inputs
        temperature = st.slider("Temperature (°C)", 250.0, 400.0, float(eq_data['Temperature']))
        vibration = st.slider("Vibration (mm/s)", 10.0, 50.0, float(eq_data['Vibration']))
        condition = st.slider("Condition Score (%)", 0.0, 100.0, float(eq_data['Condition_Score']))
        
        # Calculate prediction
        risk_factors = [
            (100 - condition) * 0.4,
            (eq_data['Age_Years'] / 15 * 100) * 0.3,
            max(0, (temperature - 320) / 50 * 100) * 0.2,
            (vibration / 30 * 100) * 0.1
        ]
        
        total_risk = sum(risk_factors)
        failure_prob = min(total_risk / 100, 0.95)
        
        st.metric("Failure Probability", f"{failure_prob:.1%}")
        
        if failure_prob > 0.7:
            st.error("CRITICAL: Immediate maintenance required")
        elif failure_prob > 0.5:
            st.warning("HIGH RISK: Schedule maintenance soon")
        elif failure_prob > 0.3:
            st.info("MODERATE: Monitor closely")
        else:
            st.success("LOW RISK: Normal operation")
        
        # Recommended actions
        st.subheader("Recommendations")
        if failure_prob > 0.7:
            st.write("• Stop equipment immediately")
            st.write("• Schedule emergency maintenance")
            st.write("• Inspect critical components")
        elif failure_prob > 0.5:
            st.write("• Schedule maintenance within 7 days")
            st.write("• Increase monitoring frequency")
            st.write("• Prepare replacement parts")
        else:
            st.write("• Continue normal operation")
            st.write("• Follow standard maintenance schedule")

elif page == "Advanced Analytics":
    st.markdown('<h1 class="main-header">Advanced Analytics Suite</h1>', unsafe_allow_html=True)
    
    # Anomaly detection
    st.markdown('<div class="section-header">Anomaly Detection Analysis</div>', unsafe_allow_html=True)
    
    # Perform anomaly detection
    latest_data = time_series_df.groupby('Equipment_ID').last().reset_index()
    anomalies, anomaly_scores = perform_anomaly_detection(latest_data)
    
    # Add anomaly information to dataframe
    latest_data['Anomaly'] = anomalies
    latest_data['Anomaly_Score'] = anomaly_scores
    latest_data['Is_Anomaly'] = latest_data['Anomaly'] == -1
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly visualization
        fig_anomaly = px.scatter_3d(latest_data, x='Temperature', y='Pressure', z='Vibration',
                                   color='Is_Anomaly', hover_data=['Equipment_ID'],
                                   color_discrete_map={True: '#dc2626', False: '#059669'})
        fig_anomaly.update_layout(title="3D Anomaly Detection")
        st.plotly_chart(fig_anomaly, use_container_width=True)
    
    with col2:
        # Anomaly scores distribution
        fig_scores = px.histogram(latest_data, x='Anomaly_Score', nbins=20)
        fig_scores.update_layout(title="Anomaly Score Distribution")
        st.plotly_chart(fig_scores, use_container_width=True)
    
    # Statistical analysis
    st.markdown('<div class="section-header">Statistical Process Control</div>', unsafe_allow_html=True)
    
    # Control charts for key parameters
    selected_param = st.selectbox("Select Parameter for Control Chart", 
                                 ['Temperature', 'Pressure', 'Vibration', 'Power'])
    
    # Calculate control limits
    param_data = time_series_df.groupby('Timestamp')[selected_param].mean()
    mean_val = param_data.mean()
    std_val = param_data.std()
    
    ucl = mean_val + 3 * std_val  # Upper Control Limit
    lcl = mean_val - 3 * std_val  # Lower Control Limit
    
    fig_control = go.Figure()
    fig_control.add_trace(go.Scatter(x=param_data.index, y=param_data.values, 
                                   mode='lines+markers', name=selected_param))
    fig_control.add_hline(y=mean_val, line_dash="solid", line_color="green", 
                         annotation_text="Mean")
    fig_control.add_hline(y=ucl, line_dash="dash", line_color="red", 
                         annotation_text="UCL")
    fig_control.add_hline(y=lcl, line_dash="dash", line_color="red", 
                         annotation_text="LCL")
    
    fig_control.update_layout(title=f"{selected_param} Control Chart", height=400)
    st.plotly_chart(fig_control, use_container_width=True)

# Continue with other pages...
elif page == "Maintenance Operations":
    st.markdown('<h1 class="main-header">Maintenance Operations Center</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">Maintenance Schedule</div>', unsafe_allow_html=True)
        
        # Filter maintenance data
        upcoming_maintenance = maintenance_df[
            (maintenance_df['Scheduled_Date'] >= datetime.now()) & 
            (maintenance_df['Status'].isin(['Scheduled', 'Planned']))
        ].head(20)
        
        # Enhanced maintenance table
        st.dataframe(upcoming_maintenance[['Work_Order', 'Equipment_ID', 'Type', 
                                         'Scheduled_Date', 'Duration_Hours', 
                                         'Technician', 'Priority', 'Cost_USD']], 
                    use_container_width=True)
        
        # Maintenance cost analysis
        st.subheader("Cost Analysis by Maintenance Type")
        cost_by_type = maintenance_df.groupby('Type')['Cost_USD'].agg(['sum', 'mean', 'count'])
        
        fig_cost = px.bar(x=cost_by_type.index, y=cost_by_type['sum'])
        fig_cost.update_layout(title="Total Maintenance Cost by Type", yaxis_title="Cost (USD)")
        st.plotly_chart(fig_cost, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">Maintenance Metrics</div>', unsafe_allow_html=True)
        
        # Key maintenance metrics
        total_cost = maintenance_df['Cost_USD'].sum()
        avg_duration = maintenance_df['Duration_Hours'].mean()
        completed_orders = len(maintenance_df[maintenance_df['Status'] == 'Completed'])
        
        st.metric("Total Annual Cost", f"${total_cost:,.0f}")
        st.metric("Avg Duration", f"{avg_duration:.1f} hours")
        st.metric("Completed Orders", f"{completed_orders}")
        
        # Maintenance type distribution
        type_dist = maintenance_df['Type'].value_counts()
        fig_type = px.pie(values=type_dist.values, names=type_dist.index)
        st.plotly_chart(fig_type, use_container_width=True)

elif page == "Financial Analysis":
    st.markdown('<h1 class="main-header">Financial Impact Analysis</h1>', unsafe_allow_html=True)
    
    # ROI calculations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="section-header">Cost Breakdown</div>', unsafe_allow_html=True)
        
        # Annual cost categories
        cost_categories = {
            'Maintenance': 1200000,
            'Energy': 2800000,
            'Downtime': 450000,
            'Labor': 1800000,
            'Parts': 650000
        }
        
        fig_costs = px.pie(values=list(cost_categories.values()), 
                          names=list(cost_categories.keys()))
        st.plotly_chart(fig_costs, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">ROI Analysis</div>', unsafe_allow_html=True)
        
        # Predictive maintenance ROI
        traditional_cost = 7500000
        predictive_cost = 6200000
        investment = 800000
        annual_savings = traditional_cost - predictive_cost
        roi = (annual_savings - investment) / investment * 100
        payback_months = investment / (annual_savings / 12)
        
        st.metric("Annual Savings", f"${annual_savings:,.0f}")
        st.metric("ROI", f"{roi:.1f}%")
        st.metric("Payback Period", f"{payback_months:.1f} months")
        
        # ROI visualization
        roi_data = pd.DataFrame({
            'Year': [1, 2, 3, 4, 5],
            'Cumulative_Savings': [annual_savings * i - investment for i in range(1, 6)]
        })
        
        fig_roi = px.line(roi_data, x='Year', y='Cumulative_Savings')
        fig_roi.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_roi, use_container_width=True)
    
    with col3:
        st.markdown('<div class="section-header">Cost Trends</div>', unsafe_allow_html=True)
        
        # Monthly cost trends
        months = pd.date_range('2024-01-01', '2024-12-31', freq='M')
        monthly_costs = np.random.normal(500000, 50000, len(months))
        
        cost_trend = pd.DataFrame({
            'Month': months,
            'Cost': monthly_costs
        })
        
        fig_trend = px.line(cost_trend, x='Month', y='Cost')
        st.plotly_chart(fig_trend, use_container_width=True)

else:
    st.markdown(f'<h1 class="main-header">{page}</h1>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Advanced Module</div>', unsafe_allow_html=True)
    st.info(f"The {page} module provides specialized analytics and controls for industrial operations.")
    
    # Placeholder for additional advanced modules
    if "Quality" in page:
        st.write("• Statistical Quality Control")
        st.write("• Six Sigma Analytics")
        st.write("• Process Capability Studies")
        st.write("• Defect Tracking & Analysis")
    elif "Compliance" in page:
        st.write("• Regulatory Compliance Monitoring")
        st.write("• Audit Trail Management")
        st.write("• Safety Standards Tracking")
        st.write("• Environmental Impact Assessment")
    elif "Risk" in page:
        st.write("• Enterprise Risk Assessment")
        st.write("• Monte Carlo Simulations")
        st.write("• Scenario Planning")
        st.write("• Risk Mitigation Strategies")

# Footer
st.markdown("---")
st.markdown("**AI4I Industrial Analytics Platform** | Advanced Predictive Maintenance & Operations Intelligence")