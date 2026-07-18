import os
import sys
from pathlib import Path
from time import sleep

import redis
import pandas as pd
import mlflow
import mlflow.sklearn
from loguru import logger
from quixstreams import Application

while True:
    print(sys.version, __file__)
    sleep(10)