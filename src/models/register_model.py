import json
import mlflow
import dagshub
import logging
from pathlib import Path
from mlflow.client import MlflowClient


class ModelRegister:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.run_info_file = self.root_path / "run_information.json"
        self.model_name = "uber_demand_prediction_model"
        self.logger = self._init_logger()
        self._init_tracking()

    def _init_logger(self):
        logger = logging.getLogger("ModelRegister")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _init_tracking(self):
        try:
            dagshub.init(repo_owner="utsav-04", repo_name="Uber_demand_forecasting", mlflow=True)
            mlflow.set_tracking_uri("https://dagshub.com/utsav-04/Uber_demand_forecasting.mlflow")
            self.logger.info("MLflow and DagsHub tracking initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing tracking: {e}")
            raise

    def read_run_info(self):
        try:
            with open(self.run_info_file, "r") as f:
                run_info = json.load(f)
                self.logger.info("Run information loaded successfully.")
                return run_info
        except FileNotFoundError:
            self.logger.error(f"File {self.run_info_file} not found.")
            raise
        except json.JSONDecodeError:
            self.logger.error("Error decoding JSON from run information file.")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while loading run info: {e}")
            raise

    def register_model(self, model_uri: str):
        try:
            model_version = mlflow.register_model(model_uri, self.model_name)
            self.logger.info(f"Model registered successfully with name: {model_version.name}, version: {model_version.version}")
            return model_version
        except Exception as e:
            self.logger.error(f"Model registration failed: {e}")
            raise

    def stage_model(self, name: str, version: str, stage: str = "Staging"):
        try:
            client = MlflowClient()
            model_version_details = client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage,
                archive_existing_versions=False
            )
            self.logger.info(f"Model moved to stage: {stage} (version: {version}, name: {name})")
            return model_version_details
        except Exception as e:
            self.logger.error(f"Model staging failed: {e}")
            raise

    def execute(self):
        try:
            run_info = self.read_run_info()
            model_uri = run_info["model_uri"]
            version_obj = self.register_model(model_uri)
            staged_model = self.stage_model(version_obj.name, version_obj.version)
            self.logger.info("Model registration and staging completed successfully.")
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")


def main():
    try:
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent
        registrar = ModelRegister(root_path)
        registrar.execute()
    except Exception as e:
        logging.error(f"Main execution failed: {e}")


if __name__ == "__main__":
    main()
