import sys
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    artifact_dir: str = os.path.join("artifacts")
    ingested_train_path: str = os.path.join(artifact_dir, "application_train.csv")
    transformed_train_file_path = os.path.join(artifact_dir, 'train.npy')
    transformed_test_file_path = os.path.join(artifact_dir, 'test.npy')
    transformed_train_csv_path = os.path.join(artifact_dir, 'train_processed.csv')
    transformed_test_csv_path = os.path.join(artifact_dir, 'test_processed.csv')
    transformed_object_file_path = os.path.join(artifact_dir, 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.utils = MainUtils()

    def initiate_data_transformation(self):
        try:
            df = pd.read_csv(self.config.ingested_train_path)
            # Drop SK_ID_CURR
            if 'SK_ID_CURR' in df.columns:
                df = df.drop(columns=['SK_ID_CURR'])
            X = df.drop(columns=['TARGET'])
            y = df['TARGET']

            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            # Identify columns
            categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

            # Numeric pipeline
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])

            X_train_num = numeric_pipeline.fit_transform(X_train[numeric_cols])
            X_test_num = numeric_pipeline.transform(X_test[numeric_cols])

            # Categorical encoding
            X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
            X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)
            X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)

            # Combine
            all_feature_names = numeric_cols + X_train_cat.columns.tolist()
            X_train_processed = np.hstack([X_train_num, X_train_cat.values])
            X_test_processed = np.hstack([X_test_num, X_test_cat.values])

            # Save processed data
            np.save(self.config.transformed_train_file_path, {'X': X_train_processed, 'y': y_train.values})
            np.save(self.config.transformed_test_file_path, {'X': X_test_processed, 'y': y_test.values})


            # Save as .csv for inspection with column names and TARGET as last column
            train_df_out = pd.DataFrame(X_train_processed, columns=all_feature_names)
            train_df_out['TARGET'] = y_train.values
            train_df_out.to_csv(self.config.transformed_train_csv_path, index=False)

            test_df_out = pd.DataFrame(X_test_processed, columns=all_feature_names)
            test_df_out['TARGET'] = y_test.values
            test_df_out.to_csv(self.config.transformed_test_csv_path, index=False)

            # Extract medians from numeric pipeline for imputation
            numeric_feature_means = {col: numeric_pipeline.named_steps['imputer'].statistics_[i]
                                     for i, col in enumerate(numeric_cols)}

            # Combine all feature names
            all_input_features = numeric_cols + categorical_cols

            # For categorical features, we can use 0 as a default if they are one-hot encoded later
            # (assuming they will be aligned and non-existent dummies become 0)
            categorical_feature_defaults = {col: 0 for col in X_train_cat.columns}

            full_feature_defaults = {**numeric_feature_means, **categorical_feature_defaults}

            # Save preprocessor
            preprocessor = {
                'numeric_cols': numeric_cols,
                'categorical_cols': categorical_cols,
                'numeric_pipeline': numeric_pipeline,
                'categorical_columns': X_train_cat.columns.tolist(),
                'feature_defaults': full_feature_defaults # Store all feature defaults here
            }
            self.utils.save_object(self.config.transformed_object_file_path, preprocessor)

            logging.info("Data transformation completed successfully.")
            return (
                self.config.transformed_train_file_path,
                self.config.transformed_test_file_path,
                self.config.transformed_object_file_path
            )

        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)