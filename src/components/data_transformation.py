import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Creating data transformer object.")

            # Numerical and categorical feature names
            numerical_features = ["reading_score", "writing_score"]
            categorical_features = ["gender",
                                    "race_ethnicity",
                                    "parental_level_of_education",
                                    "lunch",
                                    "test_preparation_course"]
            
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())])
            
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))])
            
            logging.info("Numerical and categorical pipelines created.")
            
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ])
            
            return preprocessor           
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data for transformation.")
            
            # obtain preprocessing object
            logging.info("Obtaining preprocessor object.")
            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = "math_score"
            numerical_columns = ["reading_score", "writing_score"]
            
            # Split features and target variable
            feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_train_df = train_df[target_column_name]
            
            feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_test_df = test_df[target_column_name]
            
            # Apply transformations
            logging.info("Applying preprocessing object on training and testing data.")
            input_feature_train_arr = preprocessor_obj.fit_transform(feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(feature_test_df)
            logging.info("Preprocessing completed.")
            
            # Creating final train and test arrays
            logging.info("Combining processed features with target variable.")
            train_arr = np.c_[input_feature_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test_df)]
            
            # Save the preprocessor object as pkl file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj)
            logging.info("Preprocessor object saved successfully.")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
           
        except Exception as e:
            raise CustomException(e, sys)