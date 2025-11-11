import os
import time
import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mlflow.sklearn

from sklearn.metrics import mean_squared_error
import joblib

# Wait a few seconds for MLflow server to be ready
time.sleep(5)

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("HousePricePrediction")

# Load dataset
df = pd.read_csv("data/housing.csv")
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


with mlflow.start_run() as run:
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)

    os.makedirs("ml", exist_ok=True)
    joblib.dump(model, "ml/model.pkl")

    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:5]

    # Log the model
    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

    # Register it in the Model Registry
    model_uri = f"runs:/{run.info.run_id}/model"
    registered_model = mlflow.register_model(model_uri, "HousePriceModel")

print(f"✅ Model registered as '{registered_model.name}' version {registered_model.version}")
