
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY api_deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app
EXPOSE 8080
CMD ["uvicorn", "api_deployment.main:app", "--host", "0.0.0.0", "--port", "8080"]
