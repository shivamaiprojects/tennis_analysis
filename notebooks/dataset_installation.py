from dotenv import load_dotenv
import os

load_dotenv()

from roboflow import Roboflow

api_key = os.getenv("roboflow_api_key")
rf = Roboflow(api_key= api_key )
project = rf.workspace("viren-dhanwani").project("tennis-ball-detection")
version = project.version(6)
dataset = version.download("yolo26")
                