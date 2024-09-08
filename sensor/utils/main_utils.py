import pandas as pd
import dill
import os
import sys
import numpy as np
import yaml
from sensor.exception import SensorException 
from sensor.logger import logging 

def read_yaml_file(file_path:str) -> dict :

    try:
        with open(file_path,"rb") as yaml_file :
            return yaml.safe_load(yaml_file)
        
    except Exception as e :
        raise SensorException(e,sys)
    
def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:

    try:
        if replace :
            if os.path.exists(file_path) :
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as file :
            yaml.dump(content,file)

    except Exception as e :
        raise SensorException(e,sys)
    
def save_numpy_array_data(file_path:str,array:np.array)->None:
    """
    Save a numpy array to a file
    file_path : string location of file
    array : numpy array to save
    """
    try:
        dir = os.path.dirname(file_path)
        os.makedirs(dir,exist_ok=True)
        with open(file_path,"wb") as file_obj :
            np.save(file_obj,array) 
    except Exception as e :
        raise SensorException(e,sys) from e
    
def load_numpy_array(file_path:str)->np.array:
    """
    Load a numpy array from a file
    file_path : string location of file
    """
    try:
        with open(file_path,"rb") as file_obj :
            return np.load(file_obj)
        
    except Exception as e :
        raise SensorException(e,sys) from e
    
def save_object(file_path:str,obj:object)->None:
    """
    Save an object to a file
    file_path : string location of file
    obj : object to save
    """
    try:
        dir = os.path.dirname(file_path)
        os.makedirs(dir,exist_ok=True)
        with open(file_path,"wb") as file :
            dill.dump(obj,file)
        logging.info("Object saved successfully.") 
    except Exception as e :
        raise SensorException(e,sys) from e
    
def load_object(file_path:str) -> object:
    """
    Load an object from a file
    file_path : string location of file
    """
    try:
        if not os.path.exists(file_path) :
            raise Exception(f"The file :{file_path} does not exist.")
        
        with open(file_path,"rb") as file :
            return dill.load(file)
        
        logging.info("Object load successfull.")

    except Exception as e :
        raise SensorException(e,sys) from e