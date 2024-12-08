import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('D:/mlops final project/MLOPS-project-dev/data/weather_data.csv')  # Correct path to your dataset
X = data[['humidity', 'pressure', 'wind_speed']]  # Features
y = data['temperature']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize MLFlow experiment
experiment_name = "Weather_Model_Experiment"
mlflow.set_experiment(experiment_name)

# Start a new run
with mlflow.start_run():

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    # Log parameters, metrics, and model
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mean_absolute_error", mae)
    mlflow.log_param("dataset_size", len(data))
    mlflow.log_param("num_features", X.shape[1])

    # Log the model itself
    mlflow.sklearn.log_model(model, "random_forest_weather_model")

    # Optionally, save model locally
    model_path = 'D:/mlops final project/MLOPS-project-dev/ml-models/weather_model.joblib'
    joblib.dump(model, model_path)

    # Log the model path as an artifact in MLFlow
    mlflow.log_artifact(model_path)

    # Register the model in MLFlow's Model Registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/random_forest_weather_model"
    mlflow.register_model(model_uri, "weather_model")

    print(f"Model registered in MLFlow's Model Registry with version {mlflow.active_run().info.run_id}")
