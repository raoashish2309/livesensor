from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.constant.application import APP_HOST,APP_PORT 
from sensor.constant.training_pipeline import SAVED_MODEL_DIR,PRED_FILE_NAME
from sensor.logger import logging 
from fastapi import FastAPI,File,UploadFile,Response
from sensor.exception import SensorException
from starlette.responses import RedirectResponse
from uvicorn import run as app_run

from fastapi.responses import Response
from sensor.ML.model.estimator import ModelResolver,TargetValueMapping
from sensor.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware
import os,sys
import pandas as pd

app = FastAPI()

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train():
    try:
        training_pipeline = TrainPipeline()
        if training_pipeline.is_pipepline_running:
            return Response("Training Pipeline is already running.")
        training_pipeline.run_pipeline()
        return Response("Training successfully completed.")
    
    except Exception as e:
        return Response(f"Error occured : {e}")

@app.get("/predict")
async def predict():

    try:
        # Get data from the csv file
        pred_df = pd.read_csv(PRED_FILE_NAME)
        # Convert it to dataframe
        print(len(pred_df.columns))
        df = pred_df.drop(columns=["class"])
        print(len(df.columns))

        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists() :
            return Response("Model does not exist.")
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(best_model_path)
        y_pred = model.predict(df)
        df["class_pred"] = y_pred
        df["class_pred"].replace(TargetValueMapping().reverse_mapping,inplace=True)

        # Get the prediction output as you want
        return Response(df["class_pred"])

    except Exception as e:
            raise SensorException(e,sys)

def main():
    try:
        train_pipeline_obj = TrainPipeline()
        train_pipeline_obj.run_pipeline()

    except Exception as e:
        logging.exception(e)
        print(e)

if __name__ == "__main__" :

    # Send csv file to mongodb
    #filepath = "/home/sage/Projects/livesensor/aps_failure_training_set1.csv"
    #database_name = "AirPressureSystem"
    #collection_name = "sensor"
    #dump_csv_to_mongodb_collection(filepath,database_name,collection_name 
    app_run(app,host=APP_HOST,port=APP_PORT)
    