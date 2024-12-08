import mlflow
import mlflow.sklearn

# Set up the MLFlow tracking URI
mlflow.set_tracking_uri("file:///D:/mlops final project/MLOPS-project-dev/ml-models/mlruns")
mlflow.set_experiment("weather_model_experiment")  # Experiment Name

def start_mlflow_run():
    """Starts a new MLFlow run for tracking."""
    mlflow.start_run()
    print("MLFlow run started.")
