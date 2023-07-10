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
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            # Split the dataset into train and test
            logging.info("Split trining and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Initialize the models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Evaluate the models
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            # Get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            # Get best model name from dictionary
            index_best_model = list(model_report.values()).index(best_model_score)
            best_model_name = list(model_report.keys())[index_best_model]
            # Get the best model
            best_model = models[best_model_name]
            
            # limit the model score
            threshold = 0.6 # limit
            if best_model_score < threshold:
                raise CustomException("No best model found!")
            logging.info(f"Best found model on both training and testing dataset {best_model_name}")
            
            # save best model
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            # predict with the best model
            predicted = best_model.predict(X_test)
            # evaluate
            r2_square = r2_score(y_test, predicted)
            # return the result
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)












