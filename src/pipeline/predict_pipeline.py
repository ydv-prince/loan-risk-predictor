import os
import sys
import pandas as pd
import numpy as np
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.logger import logging
from src.exception import CustomException
from src.constant import TARGET_COLUMN
from src.utils.main_utils import MainUtils

class PredictPipeline:
    def __init__(self):
        self.utils = MainUtils()
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.top_features_path = os.path.join("artifacts", "top_features.json")

    def load_artifacts(self):
        try:
            model = self.utils.load_object(self.model_path)
            preprocessor = self.utils.load_object(self.preprocessor_path)
            with open(self.top_features_path, "r") as f:
                features_info = json.load(f)

            top_features = features_info["top_features"]
            feature_means = features_info["feature_means"]

            return model, preprocessor, top_features, feature_means
        except Exception as e:
            logging.error("Error occured while loading artifacts.")
            raise CustomException(e, sys)

    def preprocess_input(self, input_df, preprocessor, feature_means):
        try:
            # Drop columns as in training
            drop_cols = ['SK_ID_CURR']
            for col in drop_cols:
                if col in input_df.columns:
                    input_df = input_df.drop(columns=[col])

            # Fill missing features with means using reindex for performance
            all_features = preprocessor['numeric_cols'] + preprocessor['categorical_columns']
            input_df = input_df.reindex(columns=all_features, fill_value=np.nan)
            input_df = input_df.fillna(feature_means)

            # Numeric and categorical
            numeric_cols = preprocessor['numeric_cols']
            categorical_cols = preprocessor['categorical_columns']
            X_num = preprocessor['numeric_pipeline'].transform(input_df[numeric_cols])
            X_cat = pd.get_dummies(input_df[categorical_cols], drop_first=True)
            X_cat = X_cat.reindex(columns=preprocessor['categorical_columns'], fill_value=0)
            X_processed = np.hstack([X_num, X_cat.values])
            return X_processed
        except Exception as e:
            logging.error("Error in preprocessing input")
            raise CustomException(e, sys)

    def predict_from_csv(self, csv_path):
        try:
            model, preprocessor, top_features, feature_means = self.load_artifacts()
            input_df = pd.read_csv(csv_path)
            # If Unnamed: 0 exists, drop it
            if "Unnamed: 0" in input_df.columns:
                input_df = input_df.drop(columns="Unnamed: 0")
            X_processed = self.preprocess_input(input_df, preprocessor, feature_means)
            preds = model.predict(X_processed)
            input_df[TARGET_COLUMN] = preds
            # Map to labels
            target_column_mapping = {0: 'bad', 1: 'good'}
            input_df[TARGET_COLUMN] = input_df[TARGET_COLUMN].map(target_column_mapping)
            
            # Print counts of each prediction
            counts = input_df[TARGET_COLUMN].value_counts()
            print(f"Prediction counts:\n{counts.to_string()}")
            logging.info(f"Prediction counts:\n{counts.to_string()}")

            # Save predictions
            output_dir = os.path.join("artifacts", "predictions")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "prediction_file.csv")
            input_df.to_csv(output_path, index=False)
            logging.info(f"Predictions saved to {output_path}")
            return output_path
        except Exception as e:
            raise CustomException(e, sys)

    def predict_from_dict(self, user_input_dict):
        try:
            model, preprocessor, top_features, feature_means = self.load_artifacts()
            all_features = preprocessor['numeric_cols'] + preprocessor['categorical_columns']
            # Fill missing features with means (defaulting to 0.0 if not present in feature_means)
            full_input = {f: user_input_dict.get(f, feature_means.get(f, 0.0)) for f in all_features}
            input_df = pd.DataFrame([full_input])
            X_processed = self.preprocess_input(input_df, preprocessor, feature_means)
            pred = model.predict(X_processed)[0]
            return pred
        except Exception as e:
            raise CustomException(e, sys)