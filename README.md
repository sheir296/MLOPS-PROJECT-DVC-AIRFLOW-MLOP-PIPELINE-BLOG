# MLOps Pipeline with DVC and Airflow

This project demonstrates a complete MLOps pipeline using DVC (Data Version Control) for managing data and models, and Airflow for automating the workflow. The goal of this pipeline is to efficiently manage and deploy machine learning models while ensuring reproducibility and scalability in real-world applications.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation Instructions](#installation-instructions)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Workflow Automation with Airflow](#workflow-automation-with-airflow)
- [Model Training and Monitoring](#model-training-and-monitoring)
- [Key Learnings](#key-learnings)
- [License](#license)

## Project Overview

### Objective
This project showcases a complete MLOps pipeline that handles data versioning, model training, and deployment using DVC and Airflow. The project includes tools like **MLflow** for model versioning and experiment tracking, enabling a seamless and scalable ML workflow.

### Technologies Used
- **DVC** (Data Version Control): For versioning datasets and models.
- **Airflow**: For orchestrating and automating the machine learning pipeline.
- **Python**: For programming the ML model and processing data.
- **MLflow**: For model versioning and experiment tracking.
- **Pandas, NumPy**: For data processing and manipulation.
- **PostgreSQL** (optional): For storing metadata related to the experiments.

## Installation Instructions

### Prerequisites

Make sure you have the following installed:
- **Python 3.8+**
- **Git**
- **DVC**
- **Airflow** (for workflow automation)

### Steps to Install

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mlops-pipeline.git
   cd mlops-pipeline
#Install Python Dependencies
pip install -r requirements.txt

#Indvc init
itialize DVC

#Set up Airflow
pip install apache-airflow
airflow db init

#Run the pipeline
airflow webserver -p 8080
airflow scheduler

#Data Collection and Preprocessing
dvc add data/processed_data.csv
dvc push

#Workflow Automation with Airflow
DAG Structure
The Airflow DAG consists of the following tasks:

Data Collection: Fetching data from the API.
Data Preprocessing: Cleaning and transforming the data.
Model Training: Training the ML model on the preprocessed data.
Model Evaluation: Evaluating the trained model.
Model Deployment: Deploying the model to a production environment.
Task Automation
Each task is defined in the mlops_pipeline.py file, which orchestrates the pipeline. Below is a simple DAG setup in Airflow:
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

def collect_data():
    # Collect data logic

def preprocess_data():
    # Data preprocessing logic

dag = DAG('mlops_pipeline', default_args=default_args, schedule_interval='@daily')

start = DummyOperator(task_id='start', dag=dag)
collect = PythonOperator(task_id='collect_data', python_callable=collect_data, dag=dag)
preprocess = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data, dag=dag)

start >> collect >> preprocess

#Model Training and Monitoring
dvc add model.pkl
dvc push

#Key Learnings
Version Control with DVC: DVC makes it easier to track large datasets and models. It ensures that your experiments are reproducible and maintainable over time.
Airflow for Automation: Airflow plays a crucial role in automating the machine learning pipeline. It helps in scheduling tasks, handling dependencies, and making the workflow efficient.
Experiment Tracking with MLflow: MLflow allows you to monitor models, track experiments, and compare results, making it easier to maintain and iterate on your models.
End-to-End MLOps Workflow: From data collection to deployment, integrating tools like DVC, Airflow, and MLflow provides a seamless pipeline for machine learning workflows.

#License
This project is licensed under the MIT License
