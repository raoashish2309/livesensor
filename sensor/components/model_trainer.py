from sensor.utils.main_utils import load_numpy_array,save_object,load_object
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.config_entity import ModelTrainingConfig
from sensor.entity.artifact_entity import DataTransformationArtifact,ModelTrainingArtifact
import numpy as np
import os,sys
from xgboost import XGBClassifier
from sensor.ML.model.estimator import SensorModel
from sensor.ML.metric.classification_metric import get_classification_score

class ModelTrainer:

    def __init__(self,model_training_config:ModelTrainingConfig,
                 data_transformation_artifact:DataTransformationArtifact) -> None:
        
        try:
            self.model_training_config = model_training_config
            self.data_transformation_artifact = data_transformation_artifact 

        except Exception as e :
            raise SensorException(e,sys) from e
        
    def perform_hyper_parameter_tuning(self):...


    def fit_model(self,x_train,y_train):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train,y_train)
            return xgb_clf
        except Exception as e:
            raise e
        
    def initiate_model_training(self)->ModelTrainingArtifact:

        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            # Load training array and testing array
            train_arr = load_numpy_array(train_file_path)
            test_arr = load_numpy_array(test_file_path)

            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            model = self.fit_model(x_train=x_train,y_train=y_train)
            y_train_pred = model.predict(x_train)
            classification_train_metric = get_classification_score(y_train,y_train_pred)

            if classification_train_metric.f1_score<=self.model_training_config.expected_accuracy:
                raise Exception("Trained model has accuracy less or equal to expected.")
            
            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_test,y_test_pred)

            # Overfitting and Underfitting
            diff = abs(classification_train_metric.f1_score-classification_test_metric.f1_score)

            if diff>self.model_training_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good, try to do more experiments.")
            
            preprocessor = load_object(self.data_transformation_artifact.transformed_obj_file_path)
            model_dir = os.path.dirname(self.model_training_config.trained_model_file_path)
            os.makedirs(model_dir,exist_ok=True)

            sensor_model = SensorModel(preprocessor,model)
            save_object(self.model_training_config.trained_model_file_path,sensor_model)

            # Model Training artifact
            model_training_artifact = ModelTrainingArtifact(
                trained_model_file_path = self.model_training_config.trained_model_file_path,
                train_metric_artifact = classification_train_metric,
                test_metric_artifact = classification_test_metric
            )
            logging.info(f"Model training artifact:{model_training_artifact}")
            return model_training_artifact


        except Exception as e:
            raise SensorException(e,sys)