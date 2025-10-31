FROM python:3.11-slim

# Install MLflow 2.9.1 and required packages
RUN pip install --no-cache-dir mlflow==2.9.1 scikit-learn pandas

# Create directory for MLflow artifacts
RUN mkdir -p /mlflow/mlruns

WORKDIR /mlflow

EXPOSE 5000

ENTRYPOINT ["mlflow", "server", \
            "--backend-store-uri", "sqlite:///mlflow.db", \
            "--default-artifact-root", "/mlflow/mlruns", \
            "--host", "0.0.0.0"]
