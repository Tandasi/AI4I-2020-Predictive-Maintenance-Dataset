
# Production Deployment Guide - AI4I Predictive Maintenance System

##  System Overview

This comprehensive guide covers deploying a production-ready predictive maintenance system with:

- **Main API Service**: FastAPI-based prediction endpoints with monitoring
- **Anomaly Detection Service**: Real-time equipment anomaly detection
- **Interactive Dashboard**: Streamlit-based monitoring interface  
- **Database**: PostgreSQL for data persistence
- **Caching**: Redis for session management and caching
- **Monitoring Stack**: Prometheus, Grafana, and Loki for observability
- **Load Balancer**: Nginx reverse proxy with SSL support

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Main API      │    │   Anomaly       │
│   (Nginx)       │────│   Service       │    │   Service       │
│   Port: 80/443  │    │   Port: 8000    │    │   Port: 8002    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐             │
         └──────────────│   Dashboard     │─────────────┘
                        │   (Streamlit)   │
                        │   Port: 8501    │
                        └─────────────────┘
                                 │
                    ┌─────────────────────────────┐
                    │                             │
            ┌───────▼──────┐            ┌────────▼────────┐
            │  PostgreSQL  │            │     Redis       │
            │  Database    │            │     Cache       │
            │  Port: 5432  │            │   Port: 6379    │
            └──────────────┘            └─────────────────┘
```

## **Prerequisites**

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **RAM**: Minimum 8GB, Recommended 16GB
- **CPU**: Minimum 4 cores, Recommended 8+ cores  
- **Storage**: Minimum 50GB free space
- **Network**: Internet connection for pulling Docker images

### Required Software
- **Docker**: Version 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 2.0+ ([Install Compose](https://docs.docker.com/compose/install/))
- **Git**: For cloning the repository

### Verify Installation
```bash
# Check Docker version
docker --version
# Should show: Docker version 20.10.x or higher

# Check Docker Compose version  
docker compose version
# Should show: Docker Compose version v2.x.x or higher

# Test Docker functionality
docker run hello-world
```

##  Quick Start (5-Minute Deployment)

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd AI4I-2020-Predictive-Maintenance-Dataset

# Create environment file
cp .env.example .env

# Edit environment variables (see Configuration section)
nano .env
```

### 2. Deploy with Docker Compose
```bash
# Build and start all services
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs -f
```

### 3. Access the System
- **Dashboard**: http://localhost/dashboard/
- **API Documentation**: http://localhost/api/docs
- **Monitoring**: http://localhost:3000 (Grafana, admin/admin_password_2024)
- **Metrics**: http://localhost:9090 (Prometheus)

### Python Client
```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8, 
    "tool_wear": 0
})

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Risk Level: {result['risk_level']}")
```

### curl Command
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"air_temperature":298.1,"process_temperature":308.6,"rotational_speed":1551,"torque":42.8,"tool_wear":0}'
```

## Security Considerations

- Add API authentication (JWT tokens)
- Implement rate limiting
- Use HTTPS in production
- Validate input data ranges
- Log all predictions for audit

## Scaling Recommendations

- Use Redis for caching predictions
- Implement database logging for monitoring
- Set up automated model retraining pipeline
- Configure alerts for performance degradation

## Maintenance Tasks

- Monitor model drift weekly
- Retrain model monthly with new data
- Update API dependencies quarterly
- Review security configurations annually
