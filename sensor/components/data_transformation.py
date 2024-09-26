from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact
from sensor.entity.config_entity import DataTransformationConfig
from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.utils.main_utils import save_object,save_numpy_array_data
from sensor.ML.model.estimator import TargetValueMapping
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek

import numpy as np
import pandas as pd
import os,sys

class DataTransformation :

    def __init__(self,data_trans_config:DataTransformationConfig,
                 data_val_artifact:DataValidationArtifact) -> None:
        """
        :param data_trans_config : Configuration for data transformation
        :param data_val_artifact : Output reference of data validation stage artifacts
        """
        try:
            self.data_trans_config = data_trans_config
            self.data_validation_artifact = data_val_artifact

        except Exception as e :
            raise SensorException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame :
        try:
            return pd.read_csv(file_path)
        except Exception as e :
            raise SensorException(e,sys) 
        
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy="constant",fill_value=0)
            preprocessor = Pipeline(
                steps=[
                    ("Imputer",simple_imputer), # replace missing values with 0
                    ("Scaler",robust_scaler) # keep every feature in same range and handle outlier
                ]
            )
            return preprocessor

        except Exception as e:
            raise SensorException(e,sys) from e

    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
                train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
                test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

                preprocessor = self.get_data_transformer_object()

                # Training data
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
                target_feature_train = train_df[TARGET_COLUMN]
                target_feature_train = target_feature_train.replace(TargetValueMapping().to_dct()).infer_objects(copy=False)

                # Testing data
                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
                target_feature_test = test_df[TARGET_COLUMN]
                target_feature_test = target_feature_test.replace(TargetValueMapping().to_dct()).infer_objects(copy=False)

                preprocessor_obj = preprocessor.fit(input_feature_train_df)

                transformed_input_train_feature = preprocessor_obj.transform(input_feature_train_df)
                transformed_input_test_feature = preprocessor_obj.transform(input_feature_test_df)

                smt = SMOTETomek(sampling_strategy="minority")

                input_feature_train_final,target_feature_train_final = smt.fit_resample(
                    transformed_input_train_feature,target_feature_train
                )
                input_feature_test_final,target_feature_test_final = smt.fit_resample(
                    transformed_input_test_feature,target_feature_test
                )

                train_arr = np.c_[input_feature_train_final,np.array
                (target_feature_train_final)]
                test_arr = np.c_[input_feature_test_final,np.array
                (target_feature_test_final)]

                # Save numpy array data
                save_numpy_array_data(self.data_trans_config.transformed_train_file_path,train_arr)
                save_numpy_array_data(self.data_trans_config.transformed_test_file_path,test_arr)

                save_object(self.data_trans_config.transformed_obj_file_path,preprocessor_obj)

                data_trans_artifact = DataTransformationArtifact(
                    transformed_train_file_path = self.data_trans_config.transformed_train_file_path,
                    transformed_test_file_path = self.data_trans_config.transformed_test_file_path,
                    transformed_obj_file_path = self.data_trans_config.transformed_obj_file_path
                )

                logging.info(f"Data Transformation Artifact : {data_trans_artifact}")
                return data_trans_artifact

        except Exception as e :
            raise SensorException(e,sys)