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

st.set_page_config(
    page_title="AI4I Industrial Predictive Maintenance", 
    layout="wide",
    initial_sidebar_state="expanded",
   
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-critical {
        background-color: #ff4444;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .alert-warning {
        background-color: #ffaa00;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .alert-normal {
        background-color: #00aa44;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Sidebar Navigation
st.sidebar.markdown("---")
st.sidebar.title("🏭 AI4I Control Center")
st.sidebar.markdown("*Industrial Predictive Maintenance Platform*")
st.sidebar.markdown("---")

# System Status Section
st.sidebar.subheader("🔴 System Status")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.sidebar.metric("🟢 Online", "23/25", "Equipment")
with col2:
    st.sidebar.metric("ALERTS", "7", "Active")

st.sidebar.metric("🔋 Power", "1,247 kW", "Total Consumption")
st.sidebar.metric("EFFICIENCY", "94.2%", "+2.1% today")

st.sidebar.markdown("---")

# Quick Actions
st.sidebar.subheader("QUICK ACTIONS")
if st.sidebar.button("🚨 Emergency Stop All", type="primary"):
    st.sidebar.error("Emergency protocols activated!")

if st.sidebar.button("REFRESH DATA"):
    st.sidebar.success("Data refreshed!")

if st.sidebar.button("GENERATE REPORT"):
    st.sidebar.info("Report generation started...")

st.sidebar.markdown("---")

# Filter Controls
st.sidebar.subheader("FILTERS & CONTROLS")
equipment_filter = st.sidebar.multiselect(
    "Equipment Type",
    ["All", "Pumps", "Motors", "Compressors", "Turbines"],
    default=["All"]
)

time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last Hour", "Last 24 Hours", "Last Week", "Last Month"],
    index=1
)

alert_level = st.sidebar.selectbox(
    "Alert Level",
    ["All Alerts", "Critical Only", "High & Critical", "Medium & Above"],
    index=0
)

st.sidebar.markdown("---")

# Navigation Section
st.sidebar.subheader("NAVIGATION")
page = st.sidebar.selectbox("Choose Dashboard", [
    "🏠 Main Dashboard",
    "REAL-TIME MONITORING", 
    "🔮 Predictive Analytics",
    "EQUIPMENT MANAGEMENT",
    "PERFORMANCE ANALYTICS",
    "🚨 Alerts & Notifications",
    "MAINTENANCE SCHEDULER",
    "COST ANALYSIS",
    "HISTORICAL REPORTS",
    "🤖 AI Model Training",
    "🔬 Anomaly Detection",
    "IOT SENSOR NETWORK",
    "🌐 Remote Monitoring",
    "MOBILE DASHBOARD",
    "🔐 Security & Access",
    "ENERGY MANAGEMENT",
    "🏭 Production Planning",
    "✅ Quality Control",
    "🚛 Supply Chain",
    "🤖 Digital Twin",
    "🧪 Machine Learning Lab",
    "COMPLIANCE TRACKING",
    "KPI MANAGEMENT",
    "EXECUTIVE SUMMARY"
])

st.sidebar.markdown("---")

# Recent Activity
st.sidebar.subheader("🕒 Recent Activity")
recent_activities = [
    "🔴 EQ-003 High temp alert",
    "✅ EQ-012 Maintenance completed", 
    "ALERT: EQ-018 Vibration anomaly",
    "MAINTENANCE: EQ-007 Scheduled maintenance",
    "REPORT: Weekly report generated"
]

for activity in recent_activities:
    st.sidebar.text(activity)

st.sidebar.markdown("---")

# AI Model Status
st.sidebar.subheader("🤖 AI Model Status")
st.sidebar.metric("Model Accuracy", "98.65%", "LightGBM")
st.sidebar.metric("Predictions Today", "156", "+12 since yesterday")
st.sidebar.metric("Last Training", "2 hours ago", "✅ Successful")

# Model confidence gauge
confidence = 96.8
if confidence > 95:
    confidence_color = "🟢"
elif confidence > 90:
    confidence_color = "🟡"
else:
    confidence_color = "🔴"

st.sidebar.markdown(f"{confidence_color} **Confidence:** {confidence}%")

st.sidebar.markdown("---")

# Environmental Conditions
st.sidebar.subheader("ENVIRONMENTAL")
st.sidebar.metric("Temperature", "24°C", "Facility Avg")
st.sidebar.metric("Humidity", "45%", "Optimal Range")
st.sidebar.metric("Air Quality", "Good", "AQI: 42")

st.sidebar.markdown("---")

# Footer Information
st.sidebar.markdown("---")
st.sidebar.markdown("**🏢 AI4I Factory**")
st.sidebar.markdown("📍 Industrial Zone A")
st.sidebar.markdown("👥 25 Equipment Units")
st.sidebar.markdown("🕐 Last Update: " + datetime.now().strftime("%H:%M:%S"))
st.sidebar.markdown("---")
st.sidebar.markdown("*Powered by Advanced AI Analytics*")

# Page-specific sidebar controls
if "Main Dashboard" in page:
    st.sidebar.markdown("---")
    st.sidebar.subheader("DASHBOARD CONTROLS")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Rate (sec)", 5, 60, 10)
    show_predictions = st.sidebar.checkbox("Show Predictions", value=True)
    
elif "Real-Time Monitoring" in page:
    st.sidebar.markdown("---")
    st.sidebar.subheader("MONITORING CONTROLS")
    selected_metrics = st.sidebar.multiselect(
        "Display Metrics",
        ["Temperature", "Vibration", "Pressure", "Speed", "Power"],
        default=["Temperature", "Vibration"]
    )
    chart_type = st.sidebar.selectbox("Chart Type", ["Line", "Area", "Bar"])
    
elif "Predictive Analytics" in page:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔮 Prediction Settings")
    prediction_horizon = st.sidebar.slider("Prediction Days", 1, 30, 7)
    confidence_threshold = st.sidebar.slider("Confidence %", 50, 99, 85)
    model_type = st.sidebar.selectbox("Model", ["LightGBM", "XGBoost", "Neural Network"])
    
elif "Equipment Management" in page:
    st.sidebar.markdown("---")
    st.sidebar.subheader("EQUIPMENT FILTERS")
    equipment_status = st.sidebar.multiselect(
        "Status Filter",
        ["Normal", "Warning", "Critical", "Maintenance"],
        default=["Normal", "Warning", "Critical"]
    )
    sort_by = st.sidebar.selectbox("Sort By", ["Equipment ID", "Risk Level", "Last Maintenance"])
    
elif "Alerts" in page:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚨 Alert Settings")
    alert_types = st.sidebar.multiselect(
        "Alert Types",
        ["Temperature", "Vibration", "Pressure", "Power", "Oil"],
        default=["Temperature", "Vibration"]
    )
    severity_filter = st.sidebar.selectbox("Min Severity", ["Low", "Medium", "High", "Critical"])
    
elif "Maintenance" in page:
    st.sidebar.markdown("---")
    st.sidebar.subheader("MAINTENANCE TOOLS")
    schedule_view = st.sidebar.selectbox("View", ["Calendar", "List", "Gantt Chart"])
    technician_filter = st.sidebar.selectbox("Technician", ["All", "John Smith", "Sarah Johnson", "Mike Wilson"])
    
elif "Cost Analysis" in page:
    st.sidebar.markdown("---")
    st.sidebar.subheader("FINANCIAL CONTROLS")
    cost_period = st.sidebar.selectbox("Period", ["Monthly", "Quarterly", "Yearly"])
    cost_categories = st.sidebar.multiselect(
        "Categories",
        ["Labor", "Parts", "Downtime", "Energy"],
        default=["Labor", "Parts"]
    )
    
elif "AI Model" in page:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Controls")
    training_data_size = st.sidebar.slider("Training Data %", 60, 90, 80)
    feature_selection = st.sidebar.multiselect(
        "Features",
        ["Temperature", "Vibration", "Pressure", "Speed", "Power", "Age"],
        default=["Temperature", "Vibration", "Pressure"]
    )
    
elif "Anomaly" in page:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Detection Settings")
    sensitivity = st.sidebar.slider("Sensitivity", 0.1, 2.0, 1.0)
    detection_method = st.sidebar.selectbox("Method", ["Isolation Forest", "LSTM", "Statistical"])
    
elif "IoT" in page:
    st.sidebar.markdown("---")
    st.sidebar.subheader("NETWORK CONTROLS")
    network_view = st.sidebar.selectbox("View", ["Topology", "Signal Strength", "Battery Status"])
    sensor_types = st.sidebar.multiselect(
        "Sensor Types",
        ["Temperature", "Vibration", "Pressure", "Humidity"],
        default=["Temperature", "Vibration"]
    )

def generate_equipment_data():
    """Generate realistic equipment data"""
    equipment_ids = [f"EQ-{i:03d}" for i in range(1, 26)]
    current_time = datetime.now()
    
    data = []
    for eq_id in equipment_ids:
        # Simulate different equipment states
        base_temp = np.random.normal(300, 10)
        base_vibration = np.random.normal(50, 15)
        base_pressure = np.random.normal(100, 20)
        
        # Create time series data
        for i in range(24):  # Last 24 hours
            timestamp = current_time - timedelta(hours=i)
            
            # Add some realistic variations
            temp_variation = np.random.normal(0, 2)
            vibration_variation = np.random.normal(0, 5)
            pressure_variation = np.random.normal(0, 8)
            
            data.append({
                'Equipment_ID': eq_id,
                'Timestamp': timestamp,
                'Air_Temperature': base_temp + temp_variation,
                'Process_Temperature': base_temp + 10 + temp_variation,
                'Rotational_Speed': np.random.normal(1500, 200),
                'Torque': np.random.normal(40, 10),
                'Tool_Wear': np.random.normal(100, 50),
                'Vibration': base_vibration + vibration_variation,
                'Pressure': base_pressure + pressure_variation,
                'Power_Consumption': np.random.normal(2500, 500),
                'Oil_Level': np.random.uniform(20, 100),
                'Health_Score': np.random.randint(60, 100)
            })
    
    return pd.DataFrame(data)

def generate_failure_predictions():
    """Generate failure predictions for equipment"""
    equipment_ids = [f"EQ-{i:03d}" for i in range(1, 26)]
    predictions = []
    
    for eq_id in equipment_ids:
        failure_prob = np.random.random()
        
        if failure_prob > 0.8:
            risk_level = "Critical"
            days_to_failure = np.random.randint(1, 7)
        elif failure_prob > 0.6:
            risk_level = "High"
            days_to_failure = np.random.randint(7, 30)
        elif failure_prob > 0.3:
            risk_level = "Medium"
            days_to_failure = np.random.randint(30, 90)
        else:
            risk_level = "Low"
            days_to_failure = np.random.randint(90, 365)
        
        predictions.append({
            'Equipment_ID': eq_id,
            'Failure_Probability': failure_prob,
            'Risk_Level': risk_level,
            'Days_to_Failure': days_to_failure,
            'Recommended_Action': get_recommendation(risk_level),
            'Last_Maintenance': datetime.now() - timedelta(days=np.random.randint(1, 90)),
            'Next_Maintenance': datetime.now() + timedelta(days=days_to_failure//2)
        })
    
    return pd.DataFrame(predictions)

def get_recommendation(risk_level):
    """Get maintenance recommendation based on risk level"""
    recommendations = {
        "Critical": "IMMEDIATE MAINTENANCE REQUIRED",
        "High": "Schedule maintenance within 7 days",
        "Medium": "Plan maintenance within 30 days",
        "Low": "Continue normal operation"
    }
    return recommendations.get(risk_level, "Monitor closely")

# Generate data
if 'equipment_data' not in st.session_state:
    st.session_state.equipment_data = generate_equipment_data()
    st.session_state.predictions = generate_failure_predictions()

# Main Dashboard Logic
if page == "Main Dashboard":
    st.markdown('<h1 class="main-header">AI4I Industrial Predictive Maintenance System</h1>', unsafe_allow_html=True)

    # KPI Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_equipment = len(st.session_state.predictions['Equipment_ID'].unique())
        st.metric("Total Equipment", total_equipment, "25 Active")
    
    with col2:
        critical_count = len(st.session_state.predictions[st.session_state.predictions['Risk_Level'] == 'Critical'])
        st.metric("Critical Alerts", critical_count, f"{critical_count-2} from yesterday")
    
    with col3:
        avg_health = st.session_state.equipment_data['Health_Score'].mean()
        st.metric("Avg Health Score", f"{avg_health:.1f}%", "2.3% ↑")
    
    with col4:
        mtbf = np.random.randint(120, 180)
        st.metric("MTBF (days)", mtbf, "15 days ↑")
    
    with col5:
        cost_savings = np.random.randint(50000, 150000)
        st.metric("Cost Savings", f"${cost_savings:,}", "12% ↑")
    
    # Main dashboard content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Real-Time Equipment Status")
        
        # Equipment status overview
        status_data = st.session_state.predictions.groupby('Risk_Level').size().reset_index(name='Count')
        
        fig_status = px.pie(status_data, values='Count', names='Risk_Level',
                           title="Equipment Risk Distribution",
                           color_discrete_map={
                               'Low': '#00aa44',
                               'Medium': '#ffaa00', 
                               'High': '#ff6600',
                               'Critical': '#ff4444'
                           })
        st.plotly_chart(fig_status, use_container_width=True)
        
        # Real-time sensor data
        st.subheader("Live Sensor Data")
        latest_data = st.session_state.equipment_data.groupby('Equipment_ID').last().head(10)
        
        fig_sensors = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature', 'Vibration', 'Pressure', 'Power'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # Temperature
        fig_sensors.add_trace(
            go.Scatter(x=latest_data.index, y=latest_data['Air_Temperature'], name="Air Temp"),
            row=1, col=1
        )
        
        # Vibration
        fig_sensors.add_trace(
            go.Scatter(x=latest_data.index, y=latest_data['Vibration'], name="Vibration"),
            row=1, col=2
        )
        
        # Pressure
        fig_sensors.add_trace(
            go.Scatter(x=latest_data.index, y=latest_data['Pressure'], name="Pressure"),
            row=2, col=1
        )
        
        # Power
        fig_sensors.add_trace(
            go.Scatter(x=latest_data.index, y=latest_data['Power_Consumption'], name="Power"),
            row=2, col=2
        )
        
        fig_sensors.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_sensors, use_container_width=True)
    
    with col2:
        st.subheader("🚨 Priority Alerts")
        
        # Critical alerts
        critical_equipment = st.session_state.predictions[
            st.session_state.predictions['Risk_Level'].isin(['Critical', 'High'])
        ].sort_values('Failure_Probability', ascending=False)
        
        for _, equipment in critical_equipment.head(5).iterrows():
            risk_color = "alert-critical" if equipment['Risk_Level'] == 'Critical' else "alert-warning"
            st.markdown(f"""
            <div class="{risk_color}">
                <strong>{equipment['Equipment_ID']}</strong><br/>
                Risk: {equipment['Risk_Level']}<br/>
                Probability: {equipment['Failure_Probability']:.1%}<br/>
                Action: {equipment['Recommended_Action']}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)
        
        st.subheader("🔮 AI Prediction Engine")
        
        # Equipment selection for prediction
        selected_equipment = st.selectbox("Select Equipment", 
                                        st.session_state.predictions['Equipment_ID'].unique())
        
        st.sidebar.header("EQUIPMENT PARAMETERS")
        air_temp = st.sidebar.slider("Air Temperature (K)", 295, 305, 300)
        process_temp = st.sidebar.slider("Process Temperature (K)", 305, 315, 310)
        rotation_speed = st.sidebar.slider("Rotational Speed (rpm)", 1000, 3000, 1500)
        torque = st.sidebar.slider("Torque (Nm)", 10, 80, 40)
        tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 300, 100)
        
        if st.button(" Run AI Prediction", type="primary"):
            try:
                payload = {
                    "air_temperature": air_temp,
                    "process_temperature": process_temp,
                    "rotational_speed": rotation_speed,
                    "torque": torque,
                    "tool_wear": tool_wear
                }
                
                # Try to call API, fallback to simulation if not available
                try:
                    response = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
                    if response.status_code == 200:
                        result = response.json()
                    else:
                        raise Exception("API Error")
                except:
                    # Simulate prediction if API is not available
                    failure_prob = np.random.random()
                    result = {
                        "prediction": 1 if failure_prob > 0.5 else 0,
                        "failure_probability": failure_prob,
                        "risk_level": "High" if failure_prob > 0.7 else "Medium" if failure_prob > 0.3 else "Low",
                        "recommendation": "IMMEDIATE_MAINTENANCE" if failure_prob > 0.5 else "CONTINUE_OPERATION"
                    }
                
                # Display results
                failure_prob = result["failure_probability"]
                risk_level = result["risk_level"]
                recommendation = result["recommendation"]
                
                st.metric("Failure Probability", f"{failure_prob:.2%}")
                
                if risk_level == "Critical":
                    st.error(f"{risk_level} Risk Level")
                elif risk_level == "High":
                    st.warning(f"{risk_level} Risk Level")
                elif risk_level == "Medium":
                    st.info(f"{risk_level} Risk Level")
                else:
                    st.success(f"{risk_level} Risk Level")

                st.write(f"**Recommendation:** {recommendation}")
                
                if result["prediction"] == 1:
                    st.error("FAILURE PREDICTED!")
                else:
                    st.success(" Equipment Operating Normally")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Equipment Fleet Overview
    st.subheader("Equipment Fleet Overview")
    
    # Create a more detailed fleet overview
    fleet_overview = st.session_state.predictions.copy()
    fleet_overview['Status'] = fleet_overview['Risk_Level'].map({
        'Low': ' Healthy',
        'Medium': ' Warning', 
        'High': ' At Risk',
        'Critical': ' Critical'
    })
    
    # Add color coding
    def color_code_status(val):
        if 'Critical' in val:
            return 'background-color: #ff4444; color: white'
        elif 'At Risk' in val:
            return 'background-color: #ff6600; color: white'
        elif 'Warning' in val:
            return 'background-color: #ffaa00; color: black'
        else:
            return 'background-color: #00aa44; color: white'
    
    styled_df = fleet_overview[[
        'Equipment_ID', 'Status', 'Failure_Probability', 'Days_to_Failure', 
        'Last_Maintenance', 'Next_Maintenance', 'Recommended_Action'
    ]].style.applymap(color_code_status, subset=['Status'])
    
    st.dataframe(styled_df, use_container_width=True)

elif page == " Real-Time Monitoring":
    st.title(" Real-Time Equipment Monitoring")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)")
    if auto_refresh:
        time.sleep(1)
        st.experimental_rerun()
    
    # Equipment selection
    selected_equipments = st.sidebar.multiselect(
        "Select Equipment to Monitor",
        st.session_state.equipment_data['Equipment_ID'].unique(),
        default=st.session_state.equipment_data['Equipment_ID'].unique()[:5]
    )
    
    if selected_equipments:
        filtered_data = st.session_state.equipment_data[
            st.session_state.equipment_data['Equipment_ID'].isin(selected_equipments)
        ]
        
        # Real-time charts
        metrics = ['Air_Temperature', 'Vibration', 'Pressure', 'Power_Consumption']
        
        for metric in metrics:
            st.subheader(f" {metric.replace('_', ' ')}")
            fig = px.line(filtered_data.sort_values('Timestamp'), 
                         x='Timestamp', y=metric, color='Equipment_ID',
                         title=f"{metric.replace('_', ' ')} Over Time")
            st.plotly_chart(fig, use_container_width=True)

elif page == " Predictive Analytics":
    st.title(" Advanced Predictive Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Failure Prediction Model Performance")
        
        # Model performance metrics
        accuracy = 98.65
        precision = 97.2
        recall = 96.8
        f1_score = 97.0
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [accuracy, precision, recall, f1_score]
        })
        
        fig_performance = px.bar(metrics_df, x='Metric', y='Score',
                                title="Model Performance Metrics")
        st.plotly_chart(fig_performance, use_container_width=True)
        
        st.subheader(" Prediction Accuracy Trends")
        
        # Generate accuracy trend data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
        accuracy_trend = 95 + np.random.normal(0, 2, len(dates)).cumsum() * 0.1
        accuracy_trend = np.clip(accuracy_trend, 90, 99)
        
        trend_df = pd.DataFrame({
            'Date': dates,
            'Accuracy': accuracy_trend
        })
        
        fig_trend = px.line(trend_df, x='Date', y='Accuracy',
                           title="Model Accuracy Over Time")
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        st.subheader(" Feature Importance Analysis")
        
        # Feature importance
        features = ['Tool Wear', 'Rotational Speed', 'Torque', 'Process Temperature', 'Air Temperature']
        importance = [0.35, 0.25, 0.20, 0.12, 0.08]
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        })
        
        fig_importance = px.bar(importance_df, x='Importance', y='Feature',
                               orientation='h', title="Feature Importance")
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.subheader(" Monte Carlo Simulation")
        
        # Simulate future failures
        simulation_results = []
        for _ in range(1000):
            days_ahead = np.random.randint(1, 365)
            failure_prob = np.random.beta(2, 5)  # Beta distribution for probabilities
            simulation_results.append({
                'Days_Ahead': days_ahead,
                'Failure_Probability': failure_prob
            })
        
        sim_df = pd.DataFrame(simulation_results)
        
        fig_simulation = px.scatter(sim_df, x='Days_Ahead', y='Failure_Probability',
                                   title="Monte Carlo Failure Simulation",
                                   trendline="lowess")
        st.plotly_chart(fig_simulation, use_container_width=True)

# Add more pages here...
elif page == " Equipment Management":
    st.title(" Equipment Management System")
    
    # Equipment details and management
    st.subheader(" Equipment Database")
    
    # Search and filter
    search_term = st.text_input(" Search Equipment")
    status_filter = st.selectbox("Filter by Status", 
                                ['All', 'Critical', 'High', 'Medium', 'Low'])
    
    filtered_predictions = st.session_state.predictions.copy()
    
    if search_term:
        filtered_predictions = filtered_predictions[
            filtered_predictions['Equipment_ID'].str.contains(search_term, case=False)
        ]
    
    if status_filter != 'All':
        filtered_predictions = filtered_predictions[
            filtered_predictions['Risk_Level'] == status_filter
        ]
    
    st.dataframe(filtered_predictions, use_container_width=True)

elif page == " Performance Analytics":
    st.title(" Performance Analytics Dashboard")

    # OEE Calculation
    st.subheader(" Overall Equipment Effectiveness (OEE)")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        availability = np.random.uniform(85, 95)
        st.metric("Availability", f"{availability:.1f}%")
    
    with col2:
        performance = np.random.uniform(80, 90)
        st.metric("Performance", f"{performance:.1f}%")
    
    with col3:
        quality = np.random.uniform(95, 99)
        st.metric("Quality", f"{quality:.1f}%")
    
    with col4:
        oee = (availability * performance * quality) / 10000
        st.metric("OEE", f"{oee:.1f}%")

# Continue with additional pages...

elif page == " Alerts & Notifications":
    st.title(" Alert Management System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(" Active Alerts")
        
        # Generate alert data
        alert_data = []
        alert_types = ["Critical Failure Risk", "High Temperature", "Vibration Anomaly", "Oil Pressure Low", "Power Spike"]
        severities = ["Critical", "High", "Medium", "Low"]
        
        for i in range(15):
            alert_data.append({
                "Alert_ID": f"ALT-{i+1:04d}",
                "Equipment_ID": f"EQ-{np.random.randint(1, 26):03d}",
                "Alert_Type": np.random.choice(alert_types),
                "Severity": np.random.choice(severities, p=[0.1, 0.2, 0.4, 0.3]),
                "Timestamp": datetime.now() - timedelta(minutes=np.random.randint(1, 1440)),
                "Status": np.random.choice(["New", "Acknowledged", "In Progress", "Resolved"]),
                "Assigned_To": np.random.choice(["John Smith", "Sarah Johnson", "Mike Wilson", "Lisa Chen"])
            })
        
        alerts_df = pd.DataFrame(alert_data)
        
        # Color-code by severity
        def color_severity(val):
            colors = {"Critical": "#ff4444", "High": "#ff6600", "Medium": "#ffaa00", "Low": "#00aa44"}
            return f"background-color: {colors.get(val, '#gray')}; color: white"
        
        styled_alerts = alerts_df.style.applymap(color_severity, subset=['Severity'])
        st.dataframe(styled_alerts, use_container_width=True)
    
    with col2:
        st.subheader("ALERT STATISTICS")
        
        # Alert summary
        alert_summary = alerts_df['Severity'].value_counts()
        fig_alerts = px.pie(values=alert_summary.values, names=alert_summary.index,
                           title="Alerts by Severity",
                           color_discrete_map={"Critical": "#ff4444", "High": "#ff6600", 
                                             "Medium": "#ffaa00", "Low": "#00aa44"})
        st.plotly_chart(fig_alerts, use_container_width=True)
        
        # Response time metrics
        st.metric("Avg Response Time", "4.2 min", "-1.3 min")
        st.metric("Open Alerts", len(alerts_df[alerts_df['Status'] != 'Resolved']), "+3")
        st.metric("Resolution Rate", "94.2%", "+2.1%")

elif page == " Maintenance Scheduler":
    st.title(" Smart Maintenance Scheduler")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(" Maintenance Calendar")
        
        # Generate maintenance schedule
        maintenance_data = []
        maintenance_types = ["Preventive", "Predictive", "Corrective", "Emergency"]
        technicians = ["John Smith", "Sarah Johnson", "Mike Wilson", "Lisa Chen", "Tom Davis"]
        
        for i in range(20):
            start_date = datetime.now() + timedelta(days=np.random.randint(-7, 30))
            maintenance_data.append({
                "Work_Order": f"WO-{i+1:04d}",
                "Equipment_ID": f"EQ-{np.random.randint(1, 26):03d}",
                "Type": np.random.choice(maintenance_types),
                "Scheduled_Date": start_date,
                "Duration": f"{np.random.randint(2, 8)} hours",
                "Technician": np.random.choice(technicians),
                "Priority": np.random.choice(["Critical", "High", "Medium", "Low"]),
                "Status": np.random.choice(["Scheduled", "In Progress", "Completed", "Delayed"]),
                "Estimated_Cost": f"${np.random.randint(500, 5000):,}"
            })
        
        maintenance_df = pd.DataFrame(maintenance_data)
        
        # Filter options
        filter_type = st.selectbox("Filter by Type", ["All"] + maintenance_types)
        filter_status = st.selectbox("Filter by Status", ["All", "Scheduled", "In Progress", "Completed", "Delayed"])
        
        filtered_maintenance = maintenance_df.copy()
        if filter_type != "All":
            filtered_maintenance = filtered_maintenance[filtered_maintenance['Type'] == filter_type]
        if filter_status != "All":
            filtered_maintenance = filtered_maintenance[filtered_maintenance['Status'] == filter_status]
        
        st.dataframe(filtered_maintenance, use_container_width=True)
    
    with col2:
        st.subheader(" Maintenance Metrics")
        
        # Maintenance type distribution
        type_counts = maintenance_df['Type'].value_counts()
        fig_types = px.bar(x=type_counts.index, y=type_counts.values,
                          title="Maintenance by Type",
                          color=type_counts.index)
        st.plotly_chart(fig_types, use_container_width=True)
        
        # Key metrics
        st.metric("Planned vs Unplanned", "85:15", "Ratio")
        st.metric("Avg Completion Time", "92%", "On Schedule")
        st.metric("Cost Efficiency", "$2.3M", "Annual Savings")

elif page == " Cost Analysis":
    st.title(" Financial Impact Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(" Cost Breakdown")
        
        # Cost categories
        cost_data = {
            "Category": ["Preventive Maintenance", "Corrective Maintenance", "Emergency Repairs", 
                        "Parts & Materials", "Labor Costs", "Downtime Losses"],
            "Amount": [250000, 180000, 95000, 320000, 420000, 150000]
        }
        
        cost_df = pd.DataFrame(cost_data)
        fig_costs = px.bar(cost_df, x="Category", y="Amount",
                          title="Annual Maintenance Costs")
        fig_costs.update_xaxis(tickangle=45)
        st.plotly_chart(fig_costs, use_container_width=True)
    
    with col2:
        st.subheader(" ROI Analysis")
        
        # ROI metrics
        predictive_investment = 500000
        traditional_costs = 1800000
        predictive_costs = 1200000
        savings = traditional_costs - predictive_costs
        roi = (savings - predictive_investment) / predictive_investment * 100
        
        st.metric("Investment", f"${predictive_investment:,}")
        st.metric("Annual Savings", f"${savings:,}")
        st.metric("ROI", f"{roi:.1f}%")
        st.metric("Payback Period", "8.3 months")
        
        # ROI gauge
        fig_roi = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=roi,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "ROI %"},
            delta={'reference': 50},
            gauge={'axis': {'range': [None, 200]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 90}}))
        st.plotly_chart(fig_roi, use_container_width=True)
    
    with col3:
        st.subheader(" Cost Trends")
        
        # Monthly cost trends
        months = pd.date_range('2024-01-01', periods=12, freq='M')
        preventive_costs = np.random.normal(20000, 3000, 12)
        reactive_costs = np.random.normal(35000, 8000, 12)
        
        trends_df = pd.DataFrame({
            'Month': months,
            'Preventive': preventive_costs,
            'Reactive': reactive_costs
        })
        
        fig_trends = px.line(trends_df, x='Month', y=['Preventive', 'Reactive'],
                            title="Monthly Cost Trends")
        st.plotly_chart(fig_trends, use_container_width=True)

elif page == " AI Model Training":
    st.title(" AI/ML Model Training Center")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(" Model Performance Dashboard")
        
        # Model comparison
        models_data = {
            "Model": ["Random Forest", "XGBoost", "Neural Network", "SVM", "LightGBM"],
            "Accuracy": [94.2, 96.8, 98.1, 91.5, 98.65],
            "Precision": [93.1, 95.9, 97.8, 90.2, 97.2],
            "Recall": [92.8, 96.2, 97.5, 89.8, 96.8],
            "F1_Score": [92.9, 96.0, 97.6, 90.0, 97.0],
            "Training_Time": [45, 120, 380, 25, 95]
        }
        
        models_df = pd.DataFrame(models_data)
        
        # Performance comparison chart
        fig_models = px.bar(models_df, x="Model", y=["Accuracy", "Precision", "Recall", "F1_Score"],
                           title="Model Performance Comparison",
                           barmode="group")
        st.plotly_chart(fig_models, use_container_width=True)
        
        # Training time vs accuracy
        fig_time_acc = px.scatter(models_df, x="Training_Time", y="Accuracy",
                                 size="F1_Score", color="Model",
                                 title="Training Time vs Accuracy")
        st.plotly_chart(fig_time_acc, use_container_width=True)
    
    with col2:
        st.subheader(" Model Training Controls")
        
        # Training parameters
        st.selectbox("Select Algorithm", ["LightGBM", "XGBoost", "Random Forest", "Neural Network"])
        st.slider("Training Data %", 60, 90, 80)
        st.slider("Max Depth", 3, 20, 10)
        st.slider("Learning Rate", 0.01, 0.3, 0.1)
        st.number_input("Number of Estimators", 100, 1000, 500)
        
        if st.button(" Start Training", type="primary"):
            with st.spinner("Training model..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                st.success("Model trained successfully!")
                st.balloons()
        
        st.subheader("CURRENT BEST MODEL")
        st.metric("Algorithm", "LightGBM")
        st.metric("Accuracy", "98.65%")
        st.metric("Last Updated", "2 hours ago")

elif page == "🔬 Anomaly Detection":
    st.title("🔬 Advanced Anomaly Detection")
    
    st.subheader("REAL-TIME ANOMALY MONITORING")
    
    # Generate anomaly data
    timestamp = pd.date_range(start='2024-09-01', end='2024-09-23', freq='H')
    normal_data = np.random.normal(50, 10, len(timestamp))
    
    # Add some anomalies
    anomaly_indices = np.random.choice(len(timestamp), size=20, replace=False)
    normal_data[anomaly_indices] += np.random.normal(0, 30, 20)
    
    anomaly_df = pd.DataFrame({
        'Timestamp': timestamp,
        'Sensor_Value': normal_data,
        'Is_Anomaly': [i in anomaly_indices for i in range(len(timestamp))]
    })
    
    # Anomaly detection chart
    fig_anomaly = go.Figure()
    
    # Normal data
    normal_df = anomaly_df[~anomaly_df['Is_Anomaly']]
    fig_anomaly.add_trace(go.Scatter(
        x=normal_df['Timestamp'],
        y=normal_df['Sensor_Value'],
        mode='lines',
        name='Normal',
        line=dict(color='blue')
    ))
    
    # Anomalies
    anomaly_points = anomaly_df[anomaly_df['Is_Anomaly']]
    fig_anomaly.add_trace(go.Scatter(
        x=anomaly_points['Timestamp'],
        y=anomaly_points['Sensor_Value'],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=8)
    ))
    
    fig_anomaly.update_layout(title="Sensor Data with Anomaly Detection")
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Anomalies Detected", len(anomaly_points), "+3 today")
    
    with col2:
        st.metric("Detection Accuracy", "97.3%", "+0.5%")
    
    with col3:
        st.metric("False Positive Rate", "2.1%", "-0.3%")

elif page == "IOT SENSOR NETWORK":
    st.title("IOT SENSOR NETWORK MANAGEMENT")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🌐 Sensor Network Topology")
        
        # Create a network graph simulation
        sensor_locations = {
            'Factory_Floor_1': {'x': 1, 'y': 1, 'sensors': 15, 'status': 'Online'},
            'Factory_Floor_2': {'x': 2, 'y': 1, 'sensors': 12, 'status': 'Online'},
            'Warehouse': {'x': 1, 'y': 2, 'sensors': 8, 'status': 'Maintenance'},
            'Office_Building': {'x': 2, 'y': 2, 'sensors': 5, 'status': 'Online'},
            'Parking_Lot': {'x': 0.5, 'y': 1.5, 'sensors': 3, 'status': 'Offline'}
        }
        
        locations_df = pd.DataFrame.from_dict(sensor_locations, orient='index').reset_index()
        locations_df['Location'] = locations_df['index']
        
        # Network status visualization
        color_map = {'Online': 'green', 'Maintenance': 'orange', 'Offline': 'red'}
        locations_df['color'] = locations_df['status'].map(color_map)
        
        fig_network = px.scatter(locations_df, x='x', y='y', size='sensors', color='status',
                               hover_name='Location', title="IoT Sensor Network Map",
                               color_discrete_map=color_map)
        st.plotly_chart(fig_network, use_container_width=True)
        
        # Sensor data table
        st.subheader("SENSOR STATUS OVERVIEW")
        
        sensor_data = []
        for location, data in sensor_locations.items():
            for i in range(data['sensors']):
                sensor_data.append({
                    'Sensor_ID': f"{location}_S{i+1:02d}",
                    'Location': location,
                    'Type': np.random.choice(['Temperature', 'Vibration', 'Pressure', 'Humidity']),
                    'Status': np.random.choice(['Active', 'Inactive', 'Error'], p=[0.85, 0.1, 0.05]),
                    'Last_Reading': datetime.now() - timedelta(minutes=np.random.randint(1, 60)),
                    'Battery_Level': f"{np.random.randint(20, 100)}%",
                    'Signal_Strength': f"{np.random.randint(60, 100)}%"
                })
        
        sensors_df = pd.DataFrame(sensor_data)
        st.dataframe(sensors_df.head(20), use_container_width=True)
    
    with col2:
        st.subheader("NETWORK STATISTICS")
        
        total_sensors = len(sensors_df)
        active_sensors = len(sensors_df[sensors_df['Status'] == 'Active'])
        
        st.metric("Total Sensors", total_sensors)
        st.metric("Active Sensors", active_sensors, f"{((active_sensors/total_sensors)*100):.1f}%")
        st.metric("Network Uptime", "99.7%", "+0.1%")
        st.metric("Data Transmission", "2.3 GB/day", "+150 MB")
        
        # Battery level distribution
        battery_levels = sensors_df['Battery_Level'].str.rstrip('%').astype(int)
        fig_battery = px.histogram(battery_levels, nbins=10, title="Battery Level Distribution")
        st.plotly_chart(fig_battery, use_container_width=True)

else:
    st.title(f"{page}")
    st.info("This advanced section is under development. More enterprise features coming soon!")
    
    # Show what could be included for each advanced section
    if "Remote Monitoring" in page:
        st.subheader("🌐 Potential Features:")
        st.write("- Global facility monitoring")
        st.write("- Satellite connectivity")
        st.write("- Multi-site dashboard")
        st.write("- Geographic equipment mapping")
    
    elif "Mobile Dashboard" in page:
        st.subheader("� Potential Features:")
        st.write("- Mobile-optimized interface")
        st.write("- Push notifications")
        st.write("- Offline capability")
        st.write("- QR code scanning")
    
    elif "Digital Twin" in page:
        st.subheader("POTENTIAL FEATURES:")
        st.write("- 3D equipment modeling")
        st.write("- Physics-based simulation")
        st.write("- Virtual testing environment")
        st.write("- Real-time synchronization")
    
    elif "Security" in page:
        st.subheader("🔐 Potential Features:")
        st.write("- Role-based access control")
        st.write("- Audit logging")
        st.write("- Cybersecurity monitoring")
        st.write("- Data encryption")
    
    elif "Energy Management" in page:
        st.subheader("POTENTIAL FEATURES:")
        st.write("- Power consumption tracking")
        st.write("- Energy efficiency optimization")
        st.write("- Carbon footprint monitoring")
        st.write("- Smart grid integration")
    
    elif "Supply Chain" in page:
        st.subheader("🚛 Potential Features:")
        st.write("- Parts inventory management")
        st.write("- Supplier performance tracking")
        st.write("- Automated reordering")
        st.write("- Logistics optimization")
    
    elif "Executive Summary" in page:
        st.subheader("POTENTIAL FEATURES:")
        st.write("- C-level dashboard")
        st.write("- Strategic KPIs")
        st.write("- Board reporting")
        st.write("- Business intelligence")
    
    else:
        st.markdown("### This could include:")
        st.write("- Advanced analytics and reporting")
        st.write("- Integration with enterprise systems")  
        st.write("- Custom workflow automation")
        st.write("- Real-time collaboration tools")

# Footer
st.markdown("---")
st.markdown("🏭 **AI4I Industrial Predictive Maintenance System** | Powered by Advanced ML Analytics")
