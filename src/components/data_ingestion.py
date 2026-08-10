import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd

from src.constant import APPLICATION_TRAIN_PATH
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    artifacts_folder: str = "artifacts"
    train_file_name: str = "application_train.csv"

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            os.makedirs(self.config.artifacts_folder, exist_ok=True)
            logging.info(f"Artifacts folder created at: {self.config.artifacts_folder}")

            dst_path = os.path.join(self.config.artifacts_folder, self.config.train_file_name)
            logging.info(f"Copying data from {APPLICATION_TRAIN_PATH} to {dst_path}")

            df = pd.read_csv(APPLICATION_TRAIN_PATH)
            df.to_csv(dst_path, index=False)
            logging.info(f"Data saved to {dst_path}")

            logging.info("Data ingestion completed successfully")

            return dst_path
        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {str(e)}")
            raise CustomException(e, sys)