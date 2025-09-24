# AI4I Analytics Platform

**Advanced Industrial Predictive Maintenance Dashboard by Angatech Technologies**

## About Angatech Technologies

**Angatech Technologies** is a growing provider of industrial IoT and predictive maintenance solutions. We focus on delivering practical, data-driven tools that help businesses monitor operations, reduce downtime, and make smarter decisions.

Our areas of expertise include:

- **Advanced Analytics Platforms**
- **Machine Learning Applications**
- **Industrial IoT Integration**
- **Predictive Maintenance Systems**
- **Business Intelligence Dashboards**
- **Data Engineering & Analysis**
- **Cloud-Based Deployments**
- **Dashboard Development**
- **Database Management**
- **Intelligent Scheduling & Resource Optimization**
- **Financial Analysis & Cost Efficiency Tools**

We’re passionate about building scalable solutions that bridge the gap between data and decision-making—especially for industries looking to modernize their operations.



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![AngaTech](angatech-high-resolution-logo.png)](https://www.anga-tech.com)

## Live Demo

**Experience the platform live:** [**https://ai4i-2020-predictive-maintenance.streamlit.app/**](https://ai4i-2020-predictive-maintenance.streamlit.app/)

Advanced Industrial Predictive Maintenance Dashboard developed by **Angatech Technologies**.

## **Overview**

The Angatech AI4I Analytics Platform is a comprehensive industrial IoT dashboard that provides real-time monitoring, predictive analytics, and maintenance optimization for industrial equipment. Built with cutting-edge machine learning algorithms and modern web technologies, the platform empowers maintenance teams and operations managers to enhance equipment reliability, reduce downtime, and optimize maintenance schedules.

> **✅ Clean & Optimized:** This repository has been streamlined to focus on the core `stable_dashboard.py` application, removing duplicate dashboard files for better maintainability and user experience.

## **Key Features**

### **Core Modules**
- **Executive Dashboard** - High-level KPIs and strategic insights
- **Real-Time Operations** - Live equipment monitoring with 4-parameter trending
- **Predictive Analytics** - ML-powered failure prediction and risk assessment
- **Asset Management**: Comprehensive equipment fleet management (25 assets)
- **Maintenance Operations** - Work order optimization and scheduling
- **Financial Analysis** - ROI calculations and cost-benefit analysis

### **Advanced Capabilities**
- **Real-time Parameter Monitoring** (Temperature, Pressure, Vibration, Power)
- **Machine Learning Anomaly Detection** using Isolation Forest
- **Interactive Dashboards** with Plotly visualizations
- **Professional Themes**: Dark Industrial theme optimized for 24/7 control rooms
- **Statistical Process Control** with control charts
- **Financial Impact Analysis** with ROI tracking
- **Maintenance Recommendations** with priority levels

## **Technology Stack**

- **Frontend:** Streamlit with custom CSS styling
- **Visualization:** Plotly (Interactive charts and 3D analytics)
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Isolation Forest for anomaly detection)
- **Deployment:** Docker, Docker Compose
- **Theme Management:** Custom theme switching system

## **Installation**

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Git

## **Quick Start**

1. **Clone the repository**
```bash
git clone https://github.com/Tandasi/AI4I-2020-Predictive-Maintenance-Dataset.git
cd AI4I-2020-Predictive-Maintenance-Dataset
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard**
```bash
streamlit run stable_dashboard.py --server.port 8510
```

4. **Access the platform**
   - **Live Demo**: https://ai4i-2020-predictive-maintenance.streamlit.app/
   - Local: http://localhost:8510
   - Network: http://your-ip:8510

## **Docker Deployment**

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t angatech-ai4i .
docker run -p 8510:8510 angatech-ai4i
```

## **Theme Management**

The platform includes a professional theme management system:

```bash
# List available themes
python theme_manager.py --list

# Switch to different themes
python theme_manager.py --theme dark-industrial  # Control room optimized
python theme_manager.py --theme corporate        # Executive presentations
python theme_manager.py --theme steel           # Industrial aesthetic
python theme_manager.py --theme analytics       # High-contrast for data work
```

## **Data & Analytics**

### **Equipment Fleet**
- **25 Industrial Assets** with realistic operating parameters
- **Multi-zone Management** (Zones A-E)
- **Real-time Condition Monitoring**
- **Age and Performance Tracking**

### **Predictive Models**
- **Risk Scoring Algorithm** based on:
  - Equipment condition (40% weight)
  - Age factors (30% weight)
  - Operating temperature (20% weight)
  - Vibration levels (10% weight)

### **Key Metrics**
- Equipment availability and efficiency
- Mean Time Between Failures (MTBF)
- Power consumption analysis
- Maintenance cost optimization

## **Business Value**

### **Operational Benefits**
- **+12% Equipment Efficiency** through predictive insights
- **-25% Maintenance Costs** via optimized scheduling
- **+99.7% Uptime** with proactive monitoring
- **$500K+ Annual Savings** through predictive maintenance

### **Professional Presentation**
- **Client-ready Interface** for demonstrations
- **Executive Dashboards** for C-suite presentations
- **Technical Depth** for engineering teams
- **Financial Analysis** for ROI justification

## **Project Structure**

```
AI4I-2020-Predictive-Maintenance-Dataset/
├── stable_dashboard.py              #  Main dashboard application
├── predictive_maintenance_api.py    #  REST API interface  
├── theme_manager.py                 # Theme switching utility
├── angatech-high-resolution-logo.png # Company branding assets
├── requirements.txt                 # Python dependencies
├── docker-compose.yml              #  Container orchestration
├── Dockerfile                      # Container configuration
├── .streamlit/                     # Streamlit configuration
│   ├── config.toml                 # Main config
│   ├── config_dark_industrial.toml # Dark theme
│   ├── config_corporate.toml       # Corporate theme
│   ├── config_steel.toml           # Industrial theme
│   └── config_analytics.toml       # Analytics theme
├── AI4I Dataset.ipynb              # Data science & ML analysis
├── predictive_maintenance_model.pkl # Trained ML model
├── ANGATECH_BRANDING_GUIDE.md      # Branding documentation
├── DEPLOYMENT_GUIDE.md             # Deployment instructions
└── README.md                       # Project documentation
```

## **Configuration**

### **Environment Variables**
```bash
# Optional: Customize port
STREAMLIT_PORT=8510

# Optional: Enable debug mode
DEBUG=True
```

### **Streamlit Configuration**
The platform uses custom themes located in `.streamlit/config.toml`. Modify for your deployment needs.

## **Deployment Options**

### **1. Local Development**
```bash
streamlit run stable_dashboard.py --server.port 8510
```

### **2. Production Server**
```bash
streamlit run stable_dashboard.py --server.port 8510 --server.headless true
```

### **3. Docker Container**
```bash
docker-compose up -d
```

### **4. Cloud Deployment**
- Compatible with AWS, Azure, GCP
- Supports Kubernetes deployment
- Scalable architecture

## **Contributing**

This is a proprietary Angatech Technologies solution. For modifications or feature requests, please contact the development team.

## **Support**

### **Angatech Technologies**
- **Website:** [AngaTech](https://www.anga-tech.com)
- **Email:** support@angatech.com
- **Project Team:** AI4I Development Division

### **Documentation**
- [Branding Guide](ANGATECH_BRANDING_GUIDE.md)
- [Theme Documentation](theme_manager.py)
- [API Reference](stable_dashboard.py)

## **License**

Copyright © 2025 AngaTech. All rights reserved.

This software is proprietary to AngaTech and is provided for evaluation and demonstration purposes.

## **About Angatech**

**Angatech Technologies** is a leading provider of industrial IoT and predictive maintenance solutions. We specialize in:

- **Advanced Analytics Platforms**
- **Machine Learning Solutions**
- **Industrial IoT Integration**
- **Predictive Maintenance Systems**
- **Business Intelligence Dashboards**
- **Data Engineering & Analytics**
- **Cloud Solutions & DevOps**


**Developed by the Angatech Team**

*Transforming Industrial Operations Through Intelligent Analytics*
