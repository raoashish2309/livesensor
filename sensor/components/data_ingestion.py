import os
import sys
from pandas import DataFrame
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.data_access.sensor_data import SensorData 
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.entity.config_entity import DataIngestionConfig
from sklearn.model_selection import train_test_split
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig) -> None:
        try:
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e :
            raise SensorException(e,sys)
        
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Export mongo db collection record as Dataframe into feature store
        """

        try:
            logging.info("Exporting data from mongodb to feature store")
            sensor_data = SensorData()
            dataframe = sensor_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            # Making Directories
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)

            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe

        except Exception as e :
            raise SensorException(e,sys)
        
    def split_data_as_train_test_predict(self,dataframe: DataFrame) -> None :

        try:
            train_set, test_set = train_test_split(
                dataframe,test_size=self.data_ingestion_config.train_test_split_ratio
            )
            split_index = len(test_set)
            predict_set = test_set[split_index:]
            test_set = test_set[:split_index]

            logging.info("Performed train test split on the dataframe")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Exporting train and test file path.")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,header=True
            )

            predict_set.to_csv(
                self.data_ingestion_config.pred_file_path,
                index=False,header=True
            )
            logging.info(f"Exported train and test file path.")
            logging.info("Exited split_data_as_train_test_predict method of DataIngestion class")

        except Exception as e :
            raise SensorException(e,sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_data_into_feature_store()
            dataframe.drop(
                columns=self._schema_config["drop_columns"],
                axis=1,
                inplace=True
            )            
            self.split_data_as_train_test_predict(dataframe=dataframe)
            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return dataingestionartifact


        except Exception as e :
            raise SensorException(e,sys)