import os

TARGET_COLUMN = "class"
PIPELINE_NAME = "sensor"
FILE_NAME = "sensor.csv"
ARTIFACT_DIR = "artifact"

TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join('config',"schema.yaml")
SCHEMA_DROP_COLS = "drop_columns"

"""
Data ingestion related constant values
"""

DATA_INGESTION_COLLECTION_NAME:str = "sensor"
DATA_INGESTION_DIR_NAME:str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str = "feature_store"
DATA_INGESTION_INGESTED_DIR:str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float = 0.2 

"""
Data validation related constant values
"""

DATA_VALIDATION_DIR_NAME:str = "data_validation"
DATA_VALIDATION_VALID_DIR:str = "validated"
DATA_VALIDATION_INVALID_DIR:str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "report.yaml"

"""
Data transformation related constant values
"""

DATA_TRANSFORMATION_DIR_NAME:str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str = "transformed_data"
DATA_TRANSFORMATION_TRANSFORMED_OBJ_DIR:str = "transformed_obj"

"""
Model Training related constant values
"""

MODEL_TRAINING_DIR_NAME:str = "model_trainer"
MODEL_TRAINING_TRAINED_MODEL_DIR:str = "trained_model"
MODEL_TRAINING_TRAINED_MODEL_NAME:str = "model.pkl"
MODEL_TRAINING_EXPECTED_SCORE:float = 0.65
MODEL_TRAINING_OVER_FITTING_UNDER_FITTING_THRES:float = 0.05

