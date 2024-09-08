from sensor.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainingArtifact
)
from sensor.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    TrainingPipelineConfig,
    DataTransformationConfig,
    ModelTrainingConfig
)
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.exception import SensorException
from sensor.logger import logging
import sys,os

class TrainPipeline:

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config = self.training_pipeline_config
            )
            logging.info("Starting data ingestion.")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed with artifact : {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e :
            raise SensorException(e,sys)
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            self.data_validation_config = DataValidationConfig(
                trainingpipelineconfig = self.training_pipeline_config
            )
            logging.info("Starting data validation.")
            data_validation = DataValidation(
                data_validation_config=self.data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed.")
            return data_validation_artifact

        except Exception as e :
            raise SensorException(e,sys)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:

        try:
            logging.info(f"Starting data Transformation.")
            self.data_trans_config = DataTransformationConfig(self.training_pipeline_config)

            data_transformer = DataTransformation(
                data_trans_config = self.data_trans_config,
                data_val_artifact = data_validation_artifact
            )

            data_trans_artifact = data_transformer.initiate_data_transformation()
            logging.info(f"Data transformation completed.")
            
            return data_trans_artifact

        except Exception as e :
            raise SensorException(e,sys)
        
    def start_model_trainer(self,data_trans_artifact:DataTransformationArtifact)->ModelTrainingArtifact:
        try:
            logging.info(f"Starting model training.")
            self.model_training_config = ModelTrainingConfig(self.training_pipeline_config)

            model_training = ModelTrainer(
                model_training_config = self.model_training_config,
                data_transformation_artifact=data_trans_artifact
            )

            model_training_artifacts = model_training.initiate_model_training()
            logging.info(f"Model training finished.")
            return model_training_artifacts

        except Exception as e :
            raise SensorException(e,sys) from e

        
    def run_pipeline(self)->None :
        try:
            data_ingestion_artifact:DataIngestionArtifact =  self.start_data_ingestion()
            data_validation_artifact:DataValidationArtifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_trans_artifact:DataTransformationArtifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )
            model_training_artifact:ModelTrainingArtifact = self.start_model_trainer(
                data_trans_artifact=data_trans_artifact
            )

        except Exception as e :
            raise SensorException(e,sys)