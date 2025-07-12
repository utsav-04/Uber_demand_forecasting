import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn import set_config

set_config(transform_output="pandas")

class PlotDataPreparer:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.data_path = self.root_path / "data/interim/df_without_outliers.csv"
        self.kmeans_path = self.root_path / "models/mb_kmeans.joblib"
        self.scaler_path = self.root_path / "models/scaler.joblib"
        self.output_path = self.root_path / "data/external/plot_data.csv"
        self.logger = self._init_logger()

    def _init_logger(self):
        logger = logging.getLogger("PlotDataPreparer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_data(self):
        try:
            df = pd.read_csv(self.data_path, usecols=["pickup_longitude", "pickup_latitude"])
            self.logger.info("Data loaded successfully.")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def load_model(self, path):
        try:
            model = joblib.load(path)
            self.logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model from {path}: {e}")
            raise

    def prepare_and_save(self):
        try:
            df = self.load_data()
            scaler = self.load_model(self.scaler_path)
            kmeans = self.load_model(self.kmeans_path)

            df_scaled = scaler.transform(df)
            df["region"] = kmeans.predict(df_scaled)

            # sample 500 rows from each region
            sampled_df = df.groupby("region").sample(500, random_state=42)
            sampled_df.to_csv(self.output_path, index=False)
            self.logger.info(f"Sampled plot data saved to {self.output_path}")
        except Exception as e:
            self.logger.error(f"Failed during plot data preparation: {e}")
            raise


def main():
    try:
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent
        preparer = PlotDataPreparer(root_path)
        preparer.prepare_and_save()
    except Exception as e:
        logging.error(f"Script execution failed: {e}")


if __name__ == "__main__":
    main()
