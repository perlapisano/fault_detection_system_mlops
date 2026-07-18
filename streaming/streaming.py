import os
import sys
from time import sleep

import redis
from loguru import logger
from quixstreams import Application

while True:
    print(sys.version, __file__)
    sleep(10)