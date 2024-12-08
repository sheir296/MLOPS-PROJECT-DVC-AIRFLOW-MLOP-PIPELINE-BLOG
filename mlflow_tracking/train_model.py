import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import mlflow_tracking.mlflow_setup as mlflow_setup  # Import custom MLFlow setup

# Set the MLFlow tracking URI (make sure it's correct for your setup)
mlflow.set_tracking_uri("file:///D:/mlops%20final%20project/MLOPS-project-dev/ml-models/mlruns")  # Local file URI for tracking

# Load the dataset from the absolute path
data = pd.read_csv('D:/mlops final project/MLOPS-project-dev/data/weather_data.csv')  # Adjust path
X = data[['humidity', 'pressure', 'wind_speed']]  # Features
y = data['temperature']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the experiment name (this will create the experiment in MLFlow)
mlflow.set_experiment("weather_model_experiment")  # Custom experiment name

# Start a new MLFlow run
with mlflow.start_run():
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Log model parameters and metrics with MLFlow
    mlflow.log_param("n_estimators", 100)  # Log the number of estimators
    mlflow.log_metric("mae", mae)  # Log the mean absolute error metric

    # Ensure the model directory exists
    model_dir = 'D:/mlops final project/MLOPS-project-dev/ml-models'  # Adjust the path for model saving
    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Define the model save path
    model_path = os.path.join(model_dir, "weather_model.joblib")
    print(f"Saving model to: {model_path}")

    # Save the model
    joblib.dump(model, model_path)

    # Log the model artifact using MLFlow
    mlflow.log_artifact(model_path)  # Log the model as artifact

    # Log the trained model with MLFlow
    mlflow.sklearn.log_model(model, "random_forest_weather_model")

    print(f"Model training complete. Mean Absolute Error: {mae}")

    # Optionally, register the model in MLFlow's Model Registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/random_forest_weather_model"
    mlflow.register_model(model_uri, "weather_model")

    print(f"Model registered in MLFlow's Model Registry with version {mlflow.active_run().info.run_id}")
