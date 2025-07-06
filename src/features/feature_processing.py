import logging
from pathlib import Path
import pandas as pd


class FeatureProcessor:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.input_path = self.root_path / "data/processed/resampled_data.csv"
        self.train_path = self.root_path / "data/processed/train.csv"
        self.test_path = self.root_path / "data/processed/test.csv"
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("feature_processing")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def load_data(self):
        try:
            df = pd.read_csv(self.input_path, parse_dates=["tpep_pickup_datetime"])
            self.logger.info("Data read successfully from resampled_data.csv")
            return df
        except Exception as e:
            self.logger.error(f"Error reading input data: {e}")
            raise

    def extract_datetime_features(self, df: pd.DataFrame):
        try:
            df["day_of_week"] = df["tpep_pickup_datetime"].dt.day_of_week
            df["month"] = df["tpep_pickup_datetime"].dt.month
            self.logger.info("Datetime features extracted (day_of_week, month)")
            df.set_index("tpep_pickup_datetime", inplace=True)
            self.logger.info("Datetime column set as index")
            return df
        except Exception as e:
            self.logger.error(f"Error extracting datetime features: {e}")
            raise

    def generate_lag_features(self, df: pd.DataFrame, lags=[1, 2, 3, 4]):
        try:
            region_grp = df.groupby("region")
            lag_features = pd.concat([region_grp["total_pickups"].shift(lag) for lag in lags], axis=1)
            lag_features.columns = [f"lag_{lag}" for lag in lags]
            self.logger.info("Lag features generated successfully")
            return pd.concat([lag_features, df], axis=1)
        except Exception as e:
            self.logger.error(f"Error generating lag features: {e}")
            raise

    def prepare_datasets(self, df: pd.DataFrame):
        try:
            df.dropna(inplace=True)
            self.logger.info("Dropped missing values after lagging")

            train = df.loc[df["month"].isin([1, 2]), "lag_1":"day_of_week"]
            test = df.loc[df["month"] == 3, "lag_1":"day_of_week"]
            self.logger.info("Train/test split completed")

            return train, test
        except Exception as e:
            self.logger.error(f"Error preparing train/test datasets: {e}")
            raise

    def save_datasets(self, train: pd.DataFrame, test: pd.DataFrame):
        try:
            train.to_csv(self.train_path, index=True)
            test.to_csv(self.test_path, index=True)
            self.logger.info(f"Train dataset saved to {self.train_path}")
            self.logger.info(f"Test dataset saved to {self.test_path}")
        except Exception as e:
            self.logger.error(f"Error saving datasets: {e}")
            raise

    def run(self):
        try:
            df = self.load_data()
            df = self.extract_datetime_features(df)
            df = self.generate_lag_features(df)
            train, test = self.prepare_datasets(df)
            self.save_datasets(train, test)
            self.logger.info("Feature processing pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise


if __name__ == "__main__":
    try:
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent
        processor = FeatureProcessor(root_path)
        processor.run()
    except Exception as main_err:
        logging.getLogger("feature_processing").error(f"Fatal error in main: {main_err}")
