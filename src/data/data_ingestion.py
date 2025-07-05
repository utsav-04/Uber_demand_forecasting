import dask.dataframe as dd
import logging
from pathlib import Path


class DataIngestionPipeline:
    def __init__(self, raw_data_dir: Path, df_names: list):
        self.raw_data_dir = raw_data_dir
        self.df_names = df_names
        self.min_latitude = 40.60
        self.max_latitude = 40.85
        self.min_longitude = -74.05
        self.max_longitude = -73.70
        self.min_fare_amount_val = 0.50
        self.max_fare_amount_val = 81.0
        self.min_trip_distance_val = 0.25
        self.max_trip_distance_val = 24.43
        self.columns = ['trip_distance', 'tpep_pickup_datetime', 'pickup_longitude',
                        'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount']
        self.parse_dates = ["tpep_pickup_datetime"]

        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("DataIngestionPipeline")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        return logger

    def read_dask_df(self, file_path: Path):
        try:
            df = dd.read_csv(file_path, parse_dates=self.parse_dates, usecols=self.columns)
            self.logger.info(f"Read file {file_path.name} successfully.")
            return df
        except Exception as e:
            self.logger.error(f"Error reading {file_path.name}: {str(e)}")
            return None

    def load_dataframes(self):
        dfs = []
        for file_name in self.df_names:
            file_path = self.raw_data_dir / file_name
            df = self.read_dask_df(file_path)
            if df is not None:
                dfs.append(df)
        if not dfs:
            raise ValueError("No valid DataFrames loaded. Check your file paths or contents.")
        return dd.concat(dfs, axis=0)

    def clean_data(self, df):
        try:
            df = df.loc[
                (df["pickup_latitude"].between(self.min_latitude, self.max_latitude, inclusive="both")) &
                (df["pickup_longitude"].between(self.min_longitude, self.max_longitude, inclusive="both")) &
                (df["dropoff_latitude"].between(self.min_latitude, self.max_latitude, inclusive="both")) &
                (df["dropoff_longitude"].between(self.min_longitude, self.max_longitude, inclusive="both")) &
                (df["fare_amount"].between(self.min_fare_amount_val, self.max_fare_amount_val, inclusive="both")) &
                (df["trip_distance"].between(self.min_trip_distance_val, self.max_trip_distance_val, inclusive="both"))
            ]
            self.logger.info("Outliers removed successfully.")
        except Exception as e:
            self.logger.error(f"Error during outlier removal: {str(e)}")
            raise

        try:
            cols_to_drop = ['trip_distance', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount']
            df = df.drop(cols_to_drop, axis=1)
            self.logger.info("Unnecessary columns dropped successfully.")
        except Exception as e:
            self.logger.error(f"Error while dropping columns: {str(e)}")
            raise

        try:
            df = df.compute()
            self.logger.info("Dask DataFrame computed successfully.")
        except Exception as e:
            self.logger.error(f"Error during computation of Dask DataFrame: {str(e)}")
            raise

        return df

    def save_dataframe(self, df, output_path: Path):
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            self.logger.info(f"DataFrame saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving DataFrame: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent
        raw_data_dir = root_path / "data/raw"
        output_path = root_path / "data/interim/df_without_outliers.csv"

        df_names = ["yellow_tripdata_2016-01.csv",
                    "yellow_tripdata_2016-02.csv",
                    "yellow_tripdata_2016-03.csv"]

        pipeline = DataIngestionPipeline(raw_data_dir, df_names)
        df = pipeline.load_dataframes()
        pipeline.logger.info("DataFrames loaded and merged successfully.")

        df_clean = pipeline.clean_data(df)
        pipeline.logger.info("Data cleaning completed successfully.")

        pipeline.save_dataframe(df_clean, output_path)
        pipeline.logger.info("Pipeline executed end-to-end successfully.")

    except Exception as e:
        logging.getLogger("DataIngestionPipeline").error(f"Pipeline execution failed: {str(e)}")
