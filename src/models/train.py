import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn import set_config
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


class ModelTrainer:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.data_path = self.root_path / "data/processed/train.csv"
        self.encoder_path = self.root_path / "models/encoder.joblib"
        self.model_path = self.root_path / "models/model.joblib"
        self.logger = self._init_logger()
        set_config(transform_output="pandas")

    def _init_logger(self):
        logger = logging.getLogger("ModelTrainer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def read_data(self):
        try:
            df = pd.read_csv(self.data_path, parse_dates=["tpep_pickup_datetime"])
            df.set_index("tpep_pickup_datetime", inplace=True)
            self.logger.info("Training data read and index set successfully.")
            return df
        except Exception as e:
            self.logger.error(f"Error reading training data: {e}")
            raise

    def prepare_transformer(self):
        try:
            encoder = ColumnTransformer(
                transformers=[
                    ("ohe", OneHotEncoder(drop="first", sparse_output=False), ["region", "day_of_week"])
                ],
                remainder="passthrough",
                n_jobs=-1,
                verbose_feature_names_out=False,
                force_int_remainder_cols=False
            )
            self.logger.info("Transformer initialized successfully.")
            return encoder
        except Exception as e:
            self.logger.error(f"Error initializing transformer: {e}")
            raise

    def save_artifact(self, obj, path):
        try:
            joblib.dump(obj, path)
            self.logger.info(f"Object saved successfully to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save object to {path}: {e}")
            raise

    def train_and_save(self):
        try:
            df = self.read_data()

            # split features and target
            X = df.drop(columns=["total_pickups"])
            y = df["total_pickups"]

            # initialize and fit transformer
            encoder = self.prepare_transformer()
            encoder.fit(X)

            # save encoder
            self.save_artifact(encoder, self.encoder_path)

            # transform training data
            X_encoded = encoder.transform(X)
            self.logger.info("Training data encoded successfully.")

            # train model
            model = LinearRegression()
            model.fit(X_encoded, y)
            self.logger.info("Model trained successfully.")

            # save model
            self.save_artifact(model, self.model_path)

        except Exception as e:
            self.logger.error(f"Model training pipeline failed: {e}")
            raise


def main():
    try:
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent
        trainer = ModelTrainer(root_path)
        trainer.train_and_save()
    except Exception as e:
        logging.error(f"Training execution failed: {e}")


if __name__ == "__main__":
    main()
