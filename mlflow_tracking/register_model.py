import mlflow
from mlflow.tracking import MlflowClient

# Function to register the trained model in MLFlow's Model Registry
def register_model(model_name="weather_model"):
    """Register the trained model in MLFlow's Model Registry."""
    client = MlflowClient()

    # Get the current run ID (this should be run after your training process)
    run_id = mlflow.active_run().info.run_id  # Automatically gets the run ID of the active experiment
    model_uri = f"runs:/{run_id}/random_forest_weather_model"  # Path to the logged model

    # Register the model
    model_version = mlflow.register_model(model_uri, model_name)

    # Transition model to 'Staging' stage (or 'Production')
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"  # Change to 'Production' when you're ready
    )
    
    print(f"Model registered in the '{model_name}' registry with version {model_version.version}.")

# Register the model after training
if __name__ == "__main__":
    register_model()
