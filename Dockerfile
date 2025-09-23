FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predictive_maintenance_api.py .
COPY predictive_maintenance_model.pkl .

EXPOSE 8000
CMD ["uvicorn", "predictive_maintenance_api:app", "--host", "0.0.0.0", "--port", "8000"]
