from sensor.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainingArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)
from sensor.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    TrainingPipelineConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher
from sensor.exception import SensorException
from sensor.logger import logging
import sys,os

from sensor.cloud_storage.s3_syncer import S3Sync
from sensor.constant.s3_bucket import TRAINING_BUCKET_NAME
from sensor.constant.training_pipeline import SAVED_MODEL_DIR

class TrainPipeline:

    is_pipepline_running = False
    self.s_sync = S3Sync()

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
        
    def start_model_evaluation(self,
        data_validation_artifact:DataValidationArtifact,
        model_training_artifact:ModelTrainingArtifact)->ModelEvaluationArtifact:
        try:
            logging.info(f"Starting model evaluation.")
            self.model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)

            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_eval_config,
                data_validation_artifact=data_validation_artifact,
                model_training_artifact=model_training_artifact
            )

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info(f"Model evaluation finished.")
            return model_evaluation_artifact

        except Exception as e :
            raise SensorException(e,sys) from e
        
    def start_model_pusher(self,
        model_evaluation_artifact:ModelEvaluationArtifact)->ModelEvaluationArtifact:
        try:
            logging.info(f"Starting model pusher.")
            self.model_pusher_config = ModelPusherConfig(self.training_pipeline_config)

            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_evaluation_artifact=model_evaluation_artifact
            )

            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info(f"Model pusher finished.")
            return model_pusher_artifact

        except Exception as e :
            raise SensorException(e,sys) from e
        
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                folder = self.training_pipeline_config.artifact_dir_name,
                aws_bucket_url = aws_bucket_url
            ) 

        except Exception as e :
            raise SensorException(e,sys)
        
    def sync_saved_model_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
            self.s3_sync.sync_folder_to_s3(
                folder = SAVED_MODEL_DIR,
                aws_bucket_url = aws_bucket_url
            ) 

        except Exception as e :
            raise SensorException(e,sys)

        
    def run_pipeline(self)->None :
        try:
            TrainPipeline.is_pipepline_running = True
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
            model_evaluation_artifact:ModelEvaluationArtifact = self.start_model_evaluation(
                data_validation_artifact=data_validation_artifact,
                model_training_artifact=model_training_artifact
            )

            if not model_evaluation_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the best model.")
            
            model_pusher_artifact:ModelPusherArtifact = self.start_model_pusher(
                model_evaluation_artifact=model_evaluation_artifact
            )
            TrainPipeline.is_pipepline_running = False

            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_to_s3()


        except Exception as e :
            self.sync_artifact_dir_to_s3()
            TrainPipeline.is_pipepline_running = False
            raise SensorException(e,sys)