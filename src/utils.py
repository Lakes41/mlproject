import sys
import os
import pandas as pd
import numpy as np
import dill
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error saving object at {file_path}")
        raise CustomException(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models, param_grid=None):
    try:
        report = {}
        
        for model_name, model in models.items():
            params = param_grid.get(model_name, {})
            grid = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=3,
                scoring="r2",      # change if you prefer: "neg_root_mean_squared_error"
                n_jobs=-1,
                verbose=0,
                refit=True
            )
            grid.fit(x_train, y_train)

            model.set_params(**grid.best_params_)
            model.fit(x_train, y_train)
            
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_r2
            
            logging.info(f"{model_name} evaluated successfully.")
        
        return report
    except Exception as e:
        logging.error("Error during model evaluation")
        raise CustomException(e, sys)