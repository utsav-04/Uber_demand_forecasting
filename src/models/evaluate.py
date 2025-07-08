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

    def _init_logger(self):
        logger = logging.getLogger("ModelEvaluator")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def read_data(self, path: Path):
        try:
            df = pd.read_csv(path, parse_dates=["tpep_pickup_datetime"])
            df.set_index("tpep_pickup_datetime", inplace=True)
            self.logger.info(f"Data read successfully from {path}")
            return df
        except Exception as e:
            self.logger.error(f"Error reading data from {path}: {str(e)}")
            raise

    def load_pickle_model(self, path: Path):
        try:
            model = joblib.load(path)
            self.logger.info(f"Model loaded successfully from {path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model from {path}: {str(e)}")
            raise

    def evaluate(self):
        try:
            # Load data
            train_df = self.read_data(self.train_data_path)
            test_df = self.read_data(self.test_data_path)

            X_test = test_df.drop(columns=["total_pickups"])
            y_test = test_df["total_pickups"]

            # Load and transform with encoder
            encoder = self.load_pickle_model(self.encoder_path)
            X_test_encoded = encoder.transform(X_test)
            self.logger.info("Test data encoded successfully")

            # Load model and predict
            model = self.load_pickle_model(self.model_path)
            y_pred = model.predict(X_test_encoded)
            loss = mean_absolute_percentage_error(y_test, y_pred)
            self.logger.info(f"Model evaluation complete. MAPE: {loss:.4f}")

            # Start MLflow run
            with mlflow.start_run(run_name="model"):
                mlflow.log_params(model.get_params())
                mlflow.log_metric("MAPE", loss)

                # Log datasets
                mlflow.log_input(from_pandas(train_df, targets="total_pickups"), "training")
                mlflow.log_input(from_pandas(test_df, targets="total_pickups"), "validation")

                # Log model with signature
                signature = infer_signature(X_test_encoded, y_pred)
                logged_model = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="demand_prediction",
                    signature=signature,
                    pip_requirements="requirements.txt"
                )

            # Save run information
            self._save_run_info(
                run_id=logged_model.run_id,
                artifact_path=logged_model.artifact_path,
                model_uri=logged_model.model_uri
            )
            self.logger.info("MLflow logging and run information saved successfully")

        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise

    def _save_run_info(self, run_id, artifact_path, model_uri):
        info = {
            "run_id": run_id,
            "artifact_path": artifact_path,
            "model_uri": model_uri
        }
        try:
            with open(self.run_info_path, "w") as f:
                json.dump(info, f, indent=4)
            self.logger.info("Run info written to JSON successfully")
        except Exception as e:
            self.logger.error(f"Failed to write run info to file: {str(e)}")
            raise


def main():
    try:
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent
        evaluator = ModelEvaluator(root_path)
        evaluator.evaluate()
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")


if __name__ == "__main__":
    main()