from sensor.pipeline.training_pipeline import TrainPipeline
#from sensor.utils import dump_csv_to_mongodb_collection


if __name__ == "__main__" :

    # Send csv file to mongodb
    #filepath = "/home/sage/Projects/livesensor/aps_failure_training_set1.csv"
    #database_name = "AirPressureSystem"
    #collection_name = "sensor"
    #dump_csv_to_mongodb_collection(filepath,database_name,collection_name)

    train_pipeline_obj = TrainPipeline()
    train_pipeline_obj.run_pipeline()