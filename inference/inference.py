import os
import sys
from pathlib import Path
from time import sleep

import redis
import pandas as pd
import mlflow
import mlflow.sklearn
from alembic.operations import batch
from loguru import logger
from quixstreams import Application

# ── Feature del modello ───────
MACHINE_FEATURE_NAMES = [
    "temperature_C",
    "vibration_mm_per_s",
    "pressure_kPa",
    "rotor_speed_rpm",
]
BATCH_FEATURE_NAMES= [
    "avg_temperature_C_1h",
    "avg_vibration_mm_per_s_1h",
    "avg_pressure_kPa_1h",
    "avg_rotor_speed_rpm_1h",
]

STREAMING_FEATURE_NAMES = [
    "avg_temperature_C_5min",
    "max_temperature_C_1min",
    "avg_vibration_mm_per_s_5min",
    "max_vibration_mm_per_s_1min",
    "avg_pressure_kPa_5min",
    "max_pressure_kPa_1min",
    "avg_rotor_speed_rpm_5min",
    "max_rotor_speed_rpm_1min",
]
NUMERIC_FEATURES = MACHINE_FEATURE_NAMES + STREAMING_FEATURE_NAMES

CATEGORICAL_FEATURES = ["machine_type"]
MODEL_FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Features in ordine
ORDERED_NUMERIC_FEATURES = [
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
ORDERED_FEATURE_NAMES = ORDERED_NUMERIC_FEATURES + CATEGORICAL_FEATURES

KEY_PATTERN = "feature:machine_id={machine_id}:{feature_name}"

def feature_key(machine_id, feature_name):
    """Costruisce la chiave Redis di una feature (uguale a chi la scrive)."""
    return KEY_PATTERN.format(machine_id=machine_id, feature_name=feature_name)


def get_redis_client():
    """Apre la connessione a Redis"""
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def _to_number(valore):
    """Prova a convertire una stringa Redis in numero"""
    if valore is None:
        return None
    try:
        return float(valore)
    except ValueError:
        return valore


def read_features(redis_client, machine_id, names):
    """
    Legge più feature di un utente da Redis e le restituisce in un dizionario.

    I numeri salvati come stringa vengono riconvertiti in float.
    """
    risultato = {}
    for name in names:
        valore = redis_client.get(feature_key(machine_id, name))
        risultato[name] = _to_number(valore)
    return risultato


def assemble_features(base):
    """
    Costruisce il vettore completo del modello per UNA predizione.

    `base` contiene i campi della predizione e gli
    aggregati  (1h + streaming). Aggiunge le feature on-demand e
    restituisce solo le MODEL_FEATURE_NAMES, nell'ordine giusto.
    """
    #amount_vs_avg_ratio = base["amount"] / (base["avg_amount_24h"] + 1e-9)

    # Flag paese insolito (diverso dall'Italia). bool → 0/1 per il modello.
    #is_unusual_country = int(base["country"] != "IT")

    full = {
        **base,
    }
    return {name: full[name] for name in ORDERED_FEATURE_NAMES}


# ── Configurazione ────────────────────────────────────────
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "feature-simulator")
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC", "fault-predictions")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "fault_model_base"  # nome registrato dal training
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")  # quale versione servire

# Soglia di cutoff
FAULT_THRESHOLD = 0.3


def load_model():
    """Carica il modello dal Model Registry di MLflow"""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info(f"Carico il modello dal registry: {model_uri}")
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.error(f"Impossibile caricare {model_uri} dal registry: {e}")
        sys.exit(1)


def predict_fault(model, r, record):
    """
    Classifica una singola transazione
    Ritorna None se l'utente non ha ancora aggregati su Redis (cold start).
    """
    machine_id = record["machine_id"]

    # Aggregati BATCH dell'utente:
    #    Se mancano è cold start: l'utente non è mai stato materializzato su Redis.
    batch_features = read_features(r, machine_id, BATCH_FEATURE_NAMES)
    if any(batch_features[name] is None for name in BATCH_FEATURE_NAMES):
        return None  # cold start: niente contesto, non classifichiamo

    # Aggregati STREAMING dell'utente:
    #    Se mancano non è cold start: significa "nessuna attività recente"
    #    Trattiamo l'assenza come 0, non come dato mancante.
    streaming_features = read_features(r, machine_id, STREAMING_FEATURE_NAMES)
    for name in STREAMING_FEATURE_NAMES:
        if streaming_features[name] is None:
            streaming_features[name] = 0

    # Assembla il vettore (transazione + aggregati + on-demand)

    base = {
        "temperature_C": record["temperature_C"],
        "vibration_mm_per_s": record["vibration_mm_per_s"],
        "pressure_kPa": record["pressure_kPa"],
        "rotor_speed_rpm": record["rotor_speed_rpm"],
        "machine_type": record["machine_type"],
        **batch_features,
        **streaming_features,
    }

    feature = assemble_features(base)

    # Predizione. La Pipeline applica lo stesso OneHotEncoder del training.
    X = pd.DataFrame([feature])[ORDERED_FEATURE_NAMES]
    faulty_proba = float(model.predict_proba(X)[0][1])

    return {
        "machine_id": machine_id,
        "faulty_proba": round(faulty_proba, 3),
        "faulty": faulty_proba >= FAULT_THRESHOLD,
    }


def run_inference():
    """Inference real-time: consuma il flusso live e classifica ogni transazione."""
    logger.info("Avvio Inference Pipeline (real-time)")

    model = load_model()
    r = get_redis_client()

    app = Application(
        broker_address=KAFKA_BROKER,
        consumer_group="inference-pipeline-group",
        auto_offset_reset="latest",
    )
    input_topic = app.topic(INPUT_TOPIC, value_deserializer="json")
    output_topic = app.topic(OUTPUT_TOPIC, value_serializer="json")
    sdf = app.dataframe(input_topic)

    def infer(record: dict):
        """Classifica la rilevazione e ritorna la predizione (None se cold start)."""
        risultato = predict_fault(model, r, record)
        if risultato is None:
            logger.debug(f"machine={record['machine_id']} senza aggregati su Redis")
        elif risultato["faulty"]:
            logger.warning(f"AVARIA sospetta: {risultato}")
        else:
            logger.info(f"ok: {risultato}")
        return risultato

    # Pubblicazione predizioni sul topic di output
    sdf = sdf.apply(infer)
    sdf = sdf.filter(lambda risultato: risultato is not None)  # filtriamo le prediction None
    sdf.to_topic(output_topic)

    logger.info(f"In ascolto su {INPUT_TOPIC} — predizioni su {OUTPUT_TOPIC}")
    app.run()


if __name__ == "__main__":
    run_inference()
