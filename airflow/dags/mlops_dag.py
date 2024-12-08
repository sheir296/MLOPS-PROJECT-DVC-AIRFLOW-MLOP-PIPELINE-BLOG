from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 12, 8),  # Adjust start date as needed
    'retries': 1,
}

# Define the DAG
with DAG(
    'mlops_pipeline',
    default_args=default_args,
    description='An MLOps pipeline for model training, evaluation, and push to remote storage',
    schedule_interval='@daily',  # Set to your desired schedule
) as dag:

    # Task 1: Load and preprocess the data
    def load_and_preprocess_data():
        data = pd.read_csv('D:/mlops final project/MLOPS-project-dev/data/weather_data.csv')
        X = data[['humidity', 'pressure', 'wind_speed']]
        y = data['temperature']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    # Task 2: Train the RandomForest model
    def train_model():
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        model_path = 'D:/mlops final project/MLOPS-project-dev/ml-models/weather_model.joblib'
        joblib.dump(model, model_path)
        return model_path, mae

    # Task 3: Log model to MLflow
    def log_model_to_mlflow():
        model_path, mae = train_model()
        mlflow.set_experiment('weather_model_experiment')
        with mlflow.start_run():
            mlflow.log_metric('mae', mae)
            mlflow.log_artifact(model_path)
            mlflow.sklearn.log_model(joblib.load(model_path), "random_forest_weather_model")
        return "Model logged in MLflow"

    # Task 4: Push model to Google Drive (via DVC)
    def push_model_to_dvc():
        os.system('dvc push')

    # Define the Airflow tasks
    task_log_model = PythonOperator(
        task_id='log_model_to_mlflow',
        python_callable=log_model_to_mlflow,
        dag=dag,
    )

    task_push_model = PythonOperator(
        task_id='push_model_to_dvc',
        python_callable=push_model_to_dvc,
        dag=dag,
    )

    # Task dependencies
    task_log_model >> task_push_model  # Log model to MLflow and then push to DVC
