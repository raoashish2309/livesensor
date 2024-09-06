from sensor.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from sensor.entity.config_entity import DataIngestionConfig,DataValidationConfig,TrainingPipelineConfig
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
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
            logging.info(f"Data validation completed with artifact : {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e :
            raise SensorException(e,sys)
        
    def run_pipeline(self)->None :
        try:
            data_ingestion_artifact:DataIngestionArtifact =  self.start_data_ingestion()
            data_validation_artifact:DataValidationArtifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

        except Exception as e :
            raise SensorException(e,sys)