import mlflow
import dagshub
import json
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error
from mlflow.models import infer_signature
from mlflow.data import from_pandas


class ModelEvaluator:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.train_data_path = root_path / "data/processed/train.csv"
        self.test_data_path = root_path / "data/processed/test.csv"
        self.model_path = root_path / "models/model.joblib"
        self.encoder_path = root_path / "models/encoder.joblib"
        self.run_info_path = root_path / "run_information.json"
        self.logger = self._init_logger()

        # Set scikit-learn display output
        set_config(transform_output="pandas")

        # DagsHub + MLflow tracking setup
        dagshub.init(repo_owner='utsav-04', repo_name='Uber_demand_forecasting', mlflow=True)
        mlflow.set_tracking_uri("https://dagshub.com/utsav-04/Uber_demand_forecasting.mlflow")
        mlflow.set_experiment("DVC Pipeline")