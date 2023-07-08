import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger  import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
            This function is responsible for data transformation
        '''
        try:
            # define categorical and numerical columns
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            # Define Pipeline
            # for numerical columns
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            # for categorical columns
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # logging info
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            # Combination of pipelines
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read training and testing dataset completed")

            # ----------------------------------------------------------------
            logging.info("Creating preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math score"
            numerical_columns = ["writing score", "reading score"]

            # ----------------------------------------------------------------
            # define variable X and y for training data
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            # ----------------------------------------------------------------
            # define variable X and y for testing data
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            # ----------------------------------------------------------------
            logging.info(f"Applying preprocessing object on training and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # ----------------------------------------------------------------
            '''
            Concatenating the arrays input_feature_train_arr and 
            np.Array(target_feature_train_df), which must have 
            the same number of rows. The result is an array called 
            train_arr that has the columns of both original arrays
            '''
            train_arr = np.c_[
                input_feature_train_arr, 
                np.array(target_feature_train_df)]
            
            test_arr = np.c_[
                input_feature_test_arr, 
                np.array(target_feature_test_df)]

            # ----------------------------------------------------------------
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Saved preprocessing object")

            # ----------------------------------------------------------------
            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path,)

        except Exception as e:
            raise CustomException(e,sys)




















