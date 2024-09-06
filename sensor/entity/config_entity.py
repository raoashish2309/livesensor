from datetime import datetime
from sensor.constant import training_pipeline
import os

class TrainingPipelineConfig :

    def __init__(self):

        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name:str = training_pipeline.PIPELINE_NAME
        self.artifact_dir_name:str = os.path.join(training_pipeline.ARTIFACT_DIR,timestamp)
        self.timestamp:str = timestamp
        

class DataIngestionConfig :

    def __init__(self,trainingpipelineconfig:TrainingPipelineConfig) :
        
        self.data_ingestion_dir = os.path.join(
            trainingpipelineconfig.artifact_dir_name,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )
        
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME
        )

        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )

        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )

        self.train_test_split_ratio = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

        self.collection_name = training_pipeline.DATA_INGESTION_COLLECTION_NAME