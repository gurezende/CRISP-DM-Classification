# mlflowprojects/Dockerfile
FROM python:3.12-bullseye

WORKDIR /app

# Copy requirements first for caching
RUN pip install mlflow

# Copy the rest of your project files
COPY . .  

# MLflow default port
EXPOSE 5000

# Adjust path to mlruns
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]