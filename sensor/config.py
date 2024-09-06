import pymongo
import os
from dataclasses import dataclass

@dataclass

class EnvironmentVariable :
    mongodb_url:str=os.getenv("MONGO_DB_URL")

env_var = EnvironmentVariable()

mongo_client = pymongo.MongoClient(env_var.mongodb_url)