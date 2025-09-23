import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Professional page configuration
st.set_page_config(
    page_title="Angatech AI4I Analytics Platform",
    page_icon="angatech-high-resolution-logo.png",
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
    
    /* Logo styling */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .company-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .footer-branding {
        background: var(--bg-secondary);
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 2rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Stable data generation functions
@st.cache_data(ttl=300)
def generate_equipment_data():
    """Generate stable equipment data"""
    np.random.seed(42)
    
    equipment_types = ['Centrifugal Pump', 'Motor Drive', 'Compressor', 'Heat Exchanger', 
                      'Turbine', 'Generator', 'Conveyor', 'Reactor', 'Separator', 'Valve Assembly']
    
    equipment_data = []
    for i in range(25):
        eq_id = f"EQ-{i+1:04d}"
        eq_type = np.random.choice(equipment_types)
        
        # Realistic operating parameters
        base_temp = 320 + np.random.normal(0, 20)
        base_pressure = 15 + np.random.normal(0, 3)
        base_flow = 100 + np.random.normal(0, 15)
        base_power = 50 + np.random.normal(0, 10)
        
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
            'Location': f"Zone {chr(65 + i//5)}",
            'Temperature': base_temp,
            'Pressure': base_pressure,
            'Flow_Rate': base_flow,
            'Power_Consumption': base_power,
            'Vibration': 20 + np.random.normal(0, 5),
            'Condition_Score': condition_score,
            'Status': status,
            'Availability': availability,
            'Age_Years': age_years,
            'Efficiency': min(98, availability * 100 + np.random.normal(0, 2))
        })
    
    return pd.DataFrame(equipment_data)

@st.cache_data(ttl=300)
def generate_time_series_data():
    """Generate time series data for monitoring"""
    timestamps = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                              end=datetime.now(), freq='H')
    
    time_series = []
    for i, ts in enumerate(timestamps):
        # Add daily pattern
        daily_pattern = 10 * np.sin(2 * np.pi * i / 24)
        noise = np.random.normal(0, 2)
        
        time_series.append({
            'Timestamp': ts,
            'Temperature': 320 + daily_pattern + noise,
            'Pressure': 15 + daily_pattern/2 + noise/2,
            'Vibration': 20 + daily_pattern/3 + noise,
            'Power': 50 + daily_pattern + noise,
            'Flow_Rate': 100 + daily_pattern*2 + noise*2
        })
    
    return pd.DataFrame(time_series)

# Load data with error handling
try:
    equipment_df = generate_equipment_data()
    time_series_df = generate_time_series_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Load and display Angatech company logo
try:
    logo_path = "angatech-high-resolution-logo.png"
    
    # Professional header with Angatech branding
    col_logo, col_title = st.columns([1, 4])
    
    with col_logo:
        st.image(logo_path, width=150)
    
    with col_title:
        st.markdown('<h1 class="main-header">AI4I Analytics Platform</h1>', unsafe_allow_html=True)
        st.markdown("**Industrial Predictive Maintenance Solutions by Angatech**")
        st.markdown("*Advanced Analytics • Machine Learning • IoT Integration*")

except FileNotFoundError:
    # Fallback if logo not found
    st.markdown('<h1 class="main-header">Angatech AI4I Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown("**Industrial Predictive Maintenance Solutions**")

# Sidebar with stable widgets (using unique keys)
# Add Angatech logo to sidebar
try:
    st.sidebar.image("angatech-high-resolution-logo.png", width=120)
    st.sidebar.markdown("**AngaTech**")

except FileNotFoundError:
    st.sidebar.markdown("**Angatech Solutions**")
    st.sidebar.markdown("---")

st.sidebar.markdown("### AI4I Control Center")
st.sidebar.markdown("---")

# System status
total_equipment = len(equipment_df)
critical_equipment = len(equipment_df[equipment_df['Status'] == 'Critical'])
avg_availability = equipment_df['Availability'].mean()

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Equipment", total_equipment)
with col2:
    st.metric("Critical", critical_equipment)

st.sidebar.metric("Availability", f"{avg_availability:.1%}")

# Navigation with unique key
page_options = [
    "Executive Dashboard",
    "Real-Time Operations", 
    "Predictive Analytics",
    "Asset Management",
    "Maintenance Operations",
    "Financial Analysis"
]

page = st.sidebar.selectbox("Select Module", page_options, key="main_nav")

# Time range filter with unique key
time_range = st.sidebar.selectbox("Time Range", 
    ["Last Hour", "Last 4 Hours", "Last 24 Hours", "Last 7 Days"], 
    key="time_filter")

# Refresh button with unique key
if st.sidebar.button("Refresh Data", key="refresh_btn"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")

# Main content based on selected page
if page == "Executive Dashboard":
    st.markdown('<h1 class="main-header">AI4I Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown("**Executive Overview - Industrial Predictive Maintenance by Angatech**")
    
    # Executive KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Fleet Size", f"{total_equipment}", "Assets")
    with col2:
        st.metric("Availability", f"{avg_availability:.1%}", "+1.2%")
    with col3:
        avg_efficiency = equipment_df['Efficiency'].mean()
        st.metric("Efficiency", f"{avg_efficiency:.1f}%", "+0.8%")
    with col4:
        st.metric("Critical Assets", f"{critical_equipment}", f"-2 vs last week")
    with col5:
        total_power = equipment_df['Power_Consumption'].sum()
        st.metric("Total Power", f"{total_power:.0f} kW", "-3.2%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Equipment Health Distribution</div>', unsafe_allow_html=True)
        
        status_counts = equipment_df['Status'].value_counts()
        fig_health = px.pie(values=status_counts.values, names=status_counts.index,
                           color_discrete_map={'Critical': '#dc2626', 'Warning': '#d97706', 'Normal': '#059669'})
        fig_health.update_layout(height=400)
        st.plotly_chart(fig_health, width="stretch")
    
    with col2:
        st.markdown('<div class="section-header">Performance vs Age Analysis</div>', unsafe_allow_html=True)
        
        fig_performance = px.scatter(equipment_df, x='Age_Years', y='Efficiency', 
                                   color='Status', size='Power_Consumption',
                                   hover_data=['Equipment_ID', 'Type'],
                                   color_discrete_map={'Normal': '#059669', 'Warning': '#d97706', 'Critical': '#dc2626'})
        fig_performance.update_layout(height=400)
        st.plotly_chart(fig_performance, width="stretch")
    
    # Power consumption by location
    st.markdown('<div class="section-header">Power Consumption by Location</div>', unsafe_allow_html=True)
    power_by_zone = equipment_df.groupby('Location')['Power_Consumption'].sum()
    
    fig_power = px.bar(x=power_by_zone.index, y=power_by_zone.values)
    fig_power.update_layout(height=350, yaxis_title="Power (kW)", title="Power Distribution")
    st.plotly_chart(fig_power, width="stretch")
    
    # Critical equipment alerts
    st.markdown('<div class="section-header">Critical Equipment Alerts</div>', unsafe_allow_html=True)
    
    critical_eq = equipment_df[equipment_df['Status'] == 'Critical']
    if len(critical_eq) > 0:
        for _, eq in critical_eq.head(3).iterrows():
            st.markdown(f"""
            <div class="alert-critical">
                <strong>{eq['Equipment_ID']} - {eq['Type']}</strong><br>
                Condition Score: {eq['Condition_Score']:.1f}% | Availability: {eq['Availability']:.1%}<br>
                Location: {eq['Location']} | Age: {eq['Age_Years']} years
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-normal">
            <strong>All Equipment Operating Normally</strong><br>
            No critical alerts at this time.
        </div>
        """, unsafe_allow_html=True)

elif page == "Real-Time Operations":
    st.markdown('<h1 class="main-header">Real-Time Operations Center</h1>', unsafe_allow_html=True)
    
    # Real-time monitoring
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Equipment selector with unique key
        selected_equipment = st.selectbox("Monitor Equipment", 
                                        equipment_df['Equipment_ID'].tolist(), 
                                        key="equipment_selector")
        
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
        st.markdown('<div class="section-header">Real-Time Parameter Trends</div>', unsafe_allow_html=True)
        
        # Create time series chart
        fig_trends = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature', 'Pressure', 'Vibration', 'Power'),
            vertical_spacing=0.15
        )
        
        # Add traces
        fig_trends.add_trace(
            go.Scatter(x=time_series_df['Timestamp'], y=time_series_df['Temperature'], 
                      name="Temperature", line=dict(color='#dc2626')),
            row=1, col=1
        )
        
        fig_trends.add_trace(
            go.Scatter(x=time_series_df['Timestamp'], y=time_series_df['Pressure'], 
                      name="Pressure", line=dict(color='#2563eb')),
            row=1, col=2
        )
        
        fig_trends.add_trace(
            go.Scatter(x=time_series_df['Timestamp'], y=time_series_df['Vibration'], 
                      name="Vibration", line=dict(color='#059669')),
            row=2, col=1
        )
        
        fig_trends.add_trace(
            go.Scatter(x=time_series_df['Timestamp'], y=time_series_df['Power'], 
                      name="Power", line=dict(color='#d97706')),
            row=2, col=2
        )
        
        fig_trends.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_trends, width="stretch")
    
    # Fleet status overview
    st.markdown('<div class="section-header">Fleet Status Overview</div>', unsafe_allow_html=True)
    
    status_summary = equipment_df.groupby(['Location', 'Status']).size().unstack(fill_value=0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Normal Equipment", len(equipment_df[equipment_df['Status'] == 'Normal']))
    with col2:
        st.metric("Warning Equipment", len(equipment_df[equipment_df['Status'] == 'Warning']))
    with col3:
        st.metric("Critical Equipment", len(equipment_df[equipment_df['Status'] == 'Critical']))

elif page == "Predictive Analytics":
    st.markdown('<h1 class="main-header">Predictive Analytics Engine</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">Equipment Risk Assessment</div>', unsafe_allow_html=True)
        
        # Calculate risk scores
        equipment_df['Risk_Score'] = (
            (100 - equipment_df['Condition_Score']) * 0.4 +
            (equipment_df['Age_Years'] / 15 * 100) * 0.3 +
            ((equipment_df['Temperature'] - 320) / 50 * 100).clip(0, 100) * 0.2 +
            (equipment_df['Vibration'] / 30 * 100) * 0.1
        ).clip(0, 100)
        
        # Risk categorization
        equipment_df['Risk_Category'] = pd.cut(equipment_df['Risk_Score'], 
                                             bins=[0, 25, 50, 75, 100],
                                             labels=['Low', 'Medium', 'High', 'Critical'])
        
        risk_counts = equipment_df['Risk_Category'].value_counts()
        
        fig_risk = px.bar(x=risk_counts.index, y=risk_counts.values,
                         color=risk_counts.index,
                         color_discrete_map={'Low': '#059669', 'Medium': '#d97706', 
                                           'High': '#dc2626', 'Critical': '#7c2d12'})
        fig_risk.update_layout(title="Equipment Risk Distribution")
        st.plotly_chart(fig_risk, width="stretch")
        
        # Risk vs condition scatter
        fig_scatter = px.scatter(equipment_df, x='Condition_Score', y='Risk_Score',
                               color='Status', size='Age_Years',
                               hover_data=['Equipment_ID'])
        fig_scatter.update_layout(title="Risk vs Condition Score")
        st.plotly_chart(fig_scatter, width="stretch")
    
    with col2:
        st.markdown('<div class="section-header">Risk Analysis</div>', unsafe_allow_html=True)
        
        # High risk equipment
        high_risk = equipment_df[equipment_df['Risk_Score'] > 75]
        
        st.metric("High Risk Assets", len(high_risk))
        st.metric("Average Risk Score", f"{equipment_df['Risk_Score'].mean():.1f}")
        
        st.subheader("Top Risk Equipment")
        if len(high_risk) > 0:
            for _, eq in high_risk.head(5).iterrows():
                st.markdown(f"""
                <div class="equipment-card">
                    <strong>{eq['Equipment_ID']}</strong><br>
                    Risk Score: {eq['Risk_Score']:.1f}<br>
                    Age: {eq['Age_Years']} years
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No high-risk equipment detected")

elif page == "Asset Management":
    st.markdown('<h1 class="main-header">Asset Management Dashboard</h1>', unsafe_allow_html=True)
    
    # Asset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_age = equipment_df['Age_Years'].mean()
        st.metric("Average Age", f"{avg_age:.1f} years")
    with col2:
        newest_eq = equipment_df['Age_Years'].min()
        st.metric("Newest Equipment", f"{newest_eq} years")
    with col3:
        oldest_eq = equipment_df['Age_Years'].max()
        st.metric("Oldest Equipment", f"{oldest_eq} years")
    with col4:
        high_efficiency = len(equipment_df[equipment_df['Efficiency'] > 90])
        st.metric("High Efficiency", f"{high_efficiency} units")
    
    # Asset details table
    st.markdown('<div class="section-header">Asset Inventory</div>', unsafe_allow_html=True)
    
    # Display equipment table
    display_df = equipment_df[['Equipment_ID', 'Type', 'Location', 'Age_Years', 
                              'Condition_Score', 'Status', 'Availability', 'Efficiency']]
    st.dataframe(display_df, width="stretch")
    
    # Age distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        fig_age = px.histogram(equipment_df, x='Age_Years', nbins=10)
        st.plotly_chart(fig_age, width="stretch")
    
    with col2:
        st.subheader("Equipment by Type")
        type_counts = equipment_df['Type'].value_counts()
        fig_type = px.pie(values=type_counts.values, names=type_counts.index)
        st.plotly_chart(fig_type, width="stretch")

elif page == "Maintenance Operations":
    st.markdown('<h1 class="main-header">Maintenance Operations</h1>', unsafe_allow_html=True)
    
    # Maintenance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Scheduled Tasks", "12", "+3 vs last week")
    with col2:
        st.metric("Completed Tasks", "45", "This month")
    with col3:
        st.metric("Avg Duration", "4.5 hrs", "-0.5 vs avg")
    with col4:
        st.metric("Cost Savings", "$25K", "This quarter")
    
    # Maintenance recommendations
    st.markdown('<div class="section-header">Maintenance Recommendations</div>', unsafe_allow_html=True)
    
    # Equipment needing attention
    needs_maintenance = equipment_df[
        (equipment_df['Condition_Score'] < 80) | 
        (equipment_df['Age_Years'] > 10)
    ].head(10)
    
    if len(needs_maintenance) > 0:
        st.subheader("Equipment Requiring Attention")
        for _, eq in needs_maintenance.iterrows():
            if eq['Condition_Score'] < 60:
                alert_class = "alert-critical"
                priority = "URGENT"
            elif eq['Condition_Score'] < 75:
                alert_class = "alert-warning"
                priority = "HIGH"
            else:
                alert_class = "alert-normal"
                priority = "MEDIUM"
            
            st.markdown(f"""
            <div class="{alert_class}">
                <strong>{eq['Equipment_ID']} - {eq['Type']}</strong> | Priority: {priority}<br>
                Condition: {eq['Condition_Score']:.1f}% | Age: {eq['Age_Years']} years<br>
                Recommended: {"Immediate inspection" if eq['Condition_Score'] < 60 else "Scheduled maintenance"}
            </div>
            """, unsafe_allow_html=True)

elif page == "Financial Analysis":
    st.markdown('<h1 class="main-header">Financial Impact Analysis</h1>', unsafe_allow_html=True)
    
    # Financial metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        annual_maintenance = 1200000
        st.metric("Annual Maintenance", f"${annual_maintenance:,.0f}")
    with col2:
        energy_cost = 2800000
        st.metric("Energy Costs", f"${energy_cost:,.0f}")
    with col3:
        downtime_cost = 450000
        st.metric("Downtime Cost", f"${downtime_cost:,.0f}")
    with col4:
        total_savings = 500000
        st.metric("Total Savings", f"${total_savings:,.0f}", "+12%")
    
    # Cost breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cost Categories")
        cost_categories = {
            'Maintenance': annual_maintenance,
            'Energy': energy_cost,
            'Downtime': downtime_cost,
            'Labor': 1800000,
            'Parts': 650000
        }
        
        fig_costs = px.pie(values=list(cost_categories.values()), 
                          names=list(cost_categories.keys()))
        st.plotly_chart(fig_costs, width="stretch")
    
    with col2:
        st.subheader("ROI Analysis")
        
        # ROI calculations
        traditional_cost = 7500000
        predictive_cost = 6200000
        investment = 800000
        annual_savings = traditional_cost - predictive_cost
        roi = (annual_savings - investment) / investment * 100
        
        st.metric("Annual Savings", f"${annual_savings:,.0f}")
        st.metric("ROI", f"{roi:.1f}%")
        st.metric("Payback Period", "8.2 months")

# Footer with Angatech branding
st.markdown("---")

# Professional footer with Angatech branding
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])


with footer_col1:
    st.markdown("""
    <div style="text-align: center;">
        <strong style="color: #1e293b;">Angatech Technologies</strong><br>
        <span style="color: #475569; font-size: 0.9em;">
            Modular Civic Tech for Africa & the Diaspora<br>
            Scalable • Secure • AI-Powered • Reliable
        </span>
    </div>
    """, unsafe_allow_html=True)


with footer_col2:
    st.markdown("""
  <div style="text-align: center; color: #64748b;">
    <strong>AI4I Analytics Platform</strong><br>
    Advanced Industrial Predictive Maintenance Solutions<br>
    <em>Developed by Angatech</em><br>
    <span style="font-size: 0.9em;">This analytic was proudly made by Angatech</span>
</div>
    """, unsafe_allow_html=True)

with footer_col3:
    st.markdown(f"""
    <div style='text-align: right; color: #64748b; font-size: 0.8rem;'>
        Platform Version: 1.0<br>
        © 2025 Angatech Solutions<br>
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """, unsafe_allow_html=True)