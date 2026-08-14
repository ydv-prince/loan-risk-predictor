import os
import sys

import numpy as np
import pandas as pd
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.main_utils import MainUtils
from src.logger import logging
from src.exception import CustomException

def extract_and_save_top_features(
        preprocessor_path = "artifacts/preprocessor.pkl",
        model_path = "artifacts/model.pkl",
        train_csv_path = "artifacts/train_processed.csv",
        output_json_path = "artifacts/top_features.json",
        top_n = 10
):
    try:
        logging.info(f"Loading preprocessor and model objects...")
        preprocessor = MainUtils.load_object(preprocessor_path)
        model = MainUtils().load_object(model_path)
        train_df = pd.read_csv(train_csv_path)

        feature_names = preprocessor['numeric_cols'] + preprocessor['categorical_columns']
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in top_indices]
        logging.info(f"Top {top_n} features: {top_features}")

        # Filter top_features to only include those present in train_df columns
        features_for_mean = [f for f in top_features if f in train_df.columns]
        feature_means = train_df[features_for_mean].mean().to_dict()

        with open(output_json_path, "w") as f:
                    json.dump({
                        "top_features": top_features,
                        "feature_means": feature_means},
                        f,
                        indent=4)
        logging.info(f"Top features and means saved to: {output_json_path}")
        return top_features, feature_means

    except Exception as e:
        logging.error(f"Error occured while extracting and saving top featrures {e}")
        raise CustomException(e, sys)