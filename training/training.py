import os
import sys
import json
import getpass
from pathlib import Path
from time import sleep

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, accuracy_score, ConfusionMatrixDisplay, RocCurveDisplay,
)
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")  # disegna su file senza aprire finestre
import matplotlib.pyplot as plt

while True:
    print(sys.version, __file__)
    sleep(10)