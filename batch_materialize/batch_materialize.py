import sys
import os
from datetime import timedelta
from pathlib import Path
from time import sleep

import redis
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, LongType, TimestampType
    )
from loguru import logger

while True:
    print(sys.version, __file__, os.getcwd())
    sleep(10)