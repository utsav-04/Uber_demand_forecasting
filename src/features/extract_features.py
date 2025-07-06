import joblib
import pandas as pd
import logging
from pathlib import Path
from yaml import safe_load
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


class FeatureExtractionPipeline:
    def __init__(self, root_path: Path, config_path: str = "params.yaml"):
        self.root_path = root_path
        self.data_path = self.root_path / "data/interim/df_without_outliers.csv"
        self.scaler_path = self.root_path / "models/scaler.joblib"
        self.kmeans_path = self.root_path / "models/mb_kmeans.joblib"
        self.output_path = self.root_path / "data/processed/resampled_data.csv"
        self.config_path = config_path
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("extract_features")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def read_params(self):
        try:
            with open(self.config_path, "r") as file:
                return safe_load(file)
        except Exception as e:
            self.logger.error(f"Error reading params file: {e}")
            raise

    def read_data_in_chunks(self, chunksize=100000, usecols=["pickup_latitude", "pickup_longitude"]):
        try:
            return pd.read_csv(self.data_path, chunksize=chunksize, usecols=usecols)
        except Exception as e:
            self.logger.error(f"Error reading data in chunks: {e}")
            raise

    def save_model(self, model, save_path: Path):
        try:
            joblib.dump(model, save_path)
            self.logger.info(f"Model saved successfully at {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    def fit_scaler(self):
        try:
            scaler = StandardScaler()
            df_reader = self.read_data_in_chunks()
            for chunk in df_reader:
                scaler.partial_fit(chunk)
            self.save_model(scaler, self.scaler_path)
            self.logger.info("Scaler fitted and saved successfully")
            return scaler
        except Exception as e:
            self.logger.error(f"Error fitting scaler: {e}")
            raise

    def fit_kmeans(self, scaler, params):
        try:
            df_reader = self.read_data_in_chunks()
            model = MiniBatchKMeans(**params)
            for chunk in df_reader:
                scaled_chunk = scaler.transform(chunk)
                model.partial_fit(scaled_chunk)
            self.save_model(model, self.kmeans_path)
            self.logger.info("MiniBatchKMeans model trained and saved successfully")
            return model
        except Exception as e:
            self.logger.error(f"Error training MiniBatchKMeans: {e}")
            raise

    def generate_cluster_predictions(self, scaler, model):
        try:
            df = pd.read_csv(self.data_path, parse_dates=["tpep_pickup_datetime"])
            self.logger.info("Full dataset loaded for cluster prediction")

            location_data = df[["pickup_longitude", "pickup_latitude"]]
            scaled_data = scaler.transform(location_data)
            df["region"] = model.predict(scaled_data)

            df.drop(columns=["pickup_latitude", "pickup_longitude"], inplace=True)
            df.set_index("tpep_pickup_datetime", inplace=True)
            self.logger.info("Cluster predictions added and lat/lon columns dropped")

            return df
        except Exception as e:
            self.logger.error(f"Error during cluster prediction: {e}")
            raise

    def resample_data(self, df, ewma_params):
        try:
            grouped = df.groupby("region")
            resampled = grouped["region"].resample("15min").count()
            resampled.name = "total_pickups"

            resampled_df = resampled.reset_index(level=0)
            resampled_df["total_pickups"].replace(0, 10, inplace=True)  # Avoid zero values

            resampled_df["avg_pickups"] = (
                resampled_df.groupby("region")["total_pickups"]
                .ewm(**ewma_params)
                .mean()
                .round()
                .values
            )
            self.logger.info("Resampling and EWMA smoothing completed successfully")
            return resampled_df
        except Exception as e:
            self.logger.error(f"Error during resampling or EWMA: {e}")
            raise

    def save_final_data(self, df):
        try:
            df.to_csv(self.output_path, index=True)
            self.logger.info(f"Final data saved successfully at {self.output_path}")
        except Exception as e:
            self.logger.error(f"Error saving final data: {e}")
            raise

    def run_pipeline(self):
        try:
            params = self.read_params()
            scaler = self.fit_scaler()
            model = self.fit_kmeans(scaler, params["extract_features"]["mini_batch_kmeans"])
            df_clustered = self.generate_cluster_predictions(scaler, model)
            resampled_df = self.resample_data(df_clustered, params["extract_features"]["ewma"])
            self.save_final_data(resampled_df)
            self.logger.info("Feature extraction pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise


if __name__ == "__main__":
    try:
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent

        pipeline = FeatureExtractionPipeline(root_path)
        pipeline.run_pipeline()
    except Exception as main_e:
        logging.getLogger("extract_features").error(f"Fatal error in main: {main_e}")
