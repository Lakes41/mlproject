import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class DataTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.data_trainer_config = DataTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regresssor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            # Define model parameters if needed
            param_grid = {
                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [False, True],
                },

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 11],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],  # 1=Manhattan, 2=Euclidean
                },

                "Decision Tree Regresssor": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },

                "Random Forest Regressor": {
                    "n_estimators": [200, 500],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt", "log2"],
                },

                "Gradient Boosting Regressor": {
                    "n_estimators": [200, 500],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [2, 3, 4],
                    "subsample": [0.8, 1.0],
                },

                "XGB Regressor": {
                    "n_estimators": [300, 600],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                },

                "CatBoost Regressor": {
                    "iterations": [500, 1000],
                    "learning_rate": [0.05, 0.1],
                    "depth": [4, 6, 8],
                },

                "AdaBoost Regressor": {
                    "n_estimators": [200, 500],
                    "learning_rate": [0.05, 0.1, 0.5],
                },
            }

            
            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, models, param_grid=param_grid
            )
            
            # get best model score from report
            best_model_score = max(sorted(model_report.values()))
            # get best model name from report
            best_model_name = max(model_report, key=model_report.get)
            # get best model object
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found: {best_model_name} with accuracy {model_report[best_model_name]}")
            
            save_object(
                file_path=self.data_trainer_config.trained_model_path,
                obj=best_model
            )
            
            return best_model_name, best_model_score
        except Exception as e:
            logging.info("Error occurred during model training")
            raise CustomException(e, sys)
        