import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.data_transformation import DataTransformation
from src.logger import logging

if __name__ == "__main__":
    data_transformation = DataTransformation()
    print(f"Starting data transformation process...")
    train_path, test_path, preprocessor_path = data_transformation.initiate_data_transformation()

    print(f"Transformed train data saved at: {train_path}")
    print(f"Transformed test data saved at: {test_path}")
    print(f"Preprocessor saved at: {preprocessor_path}")

    logging.info("Data transformation process finished.")
