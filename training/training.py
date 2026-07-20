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

# -- Configurazione

NUMERIC_FEATURES = [
    "temperature_C",
    "avg_temperature_C_1h",
    "avg_temperature_C_5min",
    "max_temperature_C_1min",
    "vibration_mm_per_s",
    "avg_vibration_mm_per_s_1h",
    "avg_vibration_mm_per_s_5min",
    "max_vibration_mm_per_s_1min",
    "pressure_kPa",
    "avg_pressure_kPa_1h",
    "avg_pressure_kPa_5min",
    "max_pressure_kPa_1min",
    "rotor_speed_rpm",
    "avg_rotor_speed_rpm_1h",
    "avg_rotor_speed_rpm_5min",
    "max_rotor_speed_rpm_1min",
]
CATEGORICAL_FEATURES = ["machine_type"]
FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

OFFLINE_STORE = Path(__file__).parent / "data" / "features" / "features.parquet"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT   = "fault_training_base"
MLFLOW_MODEL_NAME   = "fault_model_base"

# Iperparametri Random Forest
N_ESTIMATORS = 80
MAX_DEPTH    = 8

# Split e riproducibilità
TEST_SIZE    = 0.2
RANDOM_STATE = 42

def run_training():
    logger.info("Training BASE — un solo Random Forest, iperparametri fissi")

    # ── 1. Carica i dati ──────────────────────────────────
    if not OFFLINE_STORE.exists():
        logger.error(f"Tabella di training assente: {OFFLINE_STORE}")
        sys.exit(1)


    df = pd.read_parquet(OFFLINE_STORE)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)

    # Esempio undersampling
    # frazione = 10_000 / len(df)
    # df = df.groupby("faulty", group_keys=False).sample(frac=frazione, random_state=RANDOM_STATE)

    logger.info(f"Dataset: {len(df)} rilevazioni, avarie al {df['faulty'].mean():.1%}")


    # ── 2. Split TEMPORALE: train = passato, test = futuro ─
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    cut = int(len(df) * (1.0 - TEST_SIZE))
    train_df = df.iloc[:cut]
    test_df = df.iloc[cut:]


    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["faulty"].astype(int)
    X_test = test_df[FEATURE_NAMES]
    y_test = test_df["faulty"].astype(int)
    logger.info(f"Split temporale: train={len(train_df)}, test={len(test_df)}")

    # ── 3. Il modello (preprocessing + Random Forest in Pipeline) ──
    # OneHot sulle categoriche, numeriche grezze (agli alberi la scala non serve).
    # Preprocessing DENTRO la Pipeline: così al serving si riapplica identico.
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",  # avarie ~8%: ribilancia le classi
        random_state=RANDOM_STATE,
    )
    model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    # ── 4. MLflow: apri un run e traccia tutto ────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"random_forest_base_{pd.Timestamp.now():%H%M%S}"):
        # (a) addestra
        model.fit(X_train, y_train)

        # (b) valuta sul test futuro
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # (c) logga gli IPERPARAMETRI usati
        mlflow.log_params({
            "model": "RandomForest",
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "test_size": TEST_SIZE,
        })

        # (d) logga le METRICHE (per le avarie, la classe 1)
        mlflow.log_metric("precision_fault", precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        mlflow.log_metric("recall_fault", recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        mlflow.log_metric("f1_fault", f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

        roc_auc = roc_auc_score(y_test, y_proba)
        recall_fault = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        # (e) logga il MODELLO come artifact (con signature = schema input/output)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
        )

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)

        client = MlflowClient()
        client.update_model_version(
            name=MLFLOW_MODEL_NAME,
            version=mv.version,
            description=(
                f"Random Forest base (iperparametri fissi). "
                f"ROC-AUC={roc_auc:.3f}, recall_fault={recall_fault:.3f}."
            ),
        )
        client.set_model_version_tag(MLFLOW_MODEL_NAME, mv.version, "model_type", "RandomForest")
        client.set_model_version_tag(MLFLOW_MODEL_NAME, mv.version, "roc_auc", f"{roc_auc:.3f}")

        client.transition_model_version_stage(name=MLFLOW_MODEL_NAME, version=mv.version, stage="Staging")

    logger.info("FINE TRAINING")

if __name__ == "__main__":
    run_training()