import os
import sys
from time import sleep

import redis
from loguru import logger
from quixstreams import Application

import os
import redis
from loguru import logger
from quixstreams import Application

# ── Configurazione ────────────────────────────────────────
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "feature-simulator")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Durata delle finestre, in millisecondi
WINDOW_1MIN = 60 * 1000
WINDOW_5MIN = 5 * 60 * 1000

# TTL delle feature su Redis in secondi
FEATURE_TTL_SECONDS = 60

KEY_PATTERN = "feature:machine_id={machine_id}:{feature_name}"

NUMERICAL_FEATURES = ["temperature_C","vibration_mm_per_s","pressure_kPa","rotor_speed_rpm", ]


# ── Scrittura su Redis  ─────────────────────
def get_redis_client() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def write_feature_to_redis(r, machine_id, feature_name, value):
    """Scrive una feature su Redis con scadenza"""
    key = KEY_PATTERN.format(machine_id=machine_id, feature_name=feature_name)
    r.setex(key, FEATURE_TTL_SECONDS, str(value))


def estrai_machine_id(key):
    """Converte in str la chiave del messaggio Redpanda"""
    if isinstance(key, bytes):
        return key.decode()
    return str(key)


# ── La pipeline ───────────────────────────────────────────
def run_streaming_pipeline():
    logger.info("Avvio Streaming Feature Pipeline")

    r = get_redis_client()

    app = Application(
        broker_address=KAFKA_BROKER,
        consumer_group="streaming-pipeline-group",
        auto_offset_reset="latest"
    )

    input_topic = app.topic(INPUT_TOPIC, value_deserializer="json")

    # Lo stream di partenza, condiviso da tutti e tre i branch.
    sdf = app.dataframe(input_topic)

    for feature in NUMERICAL_FEATURES:
        # MAX 1 minuto
        sdf_sliding = (
            sdf
            .apply(lambda row: row[feature])
            .sliding_window(duration_ms=WINDOW_1MIN, name="max_"+feature+"_1min")
            .max()
            .current()
        )
        sdf_sliding.update(
            lambda result, key, timestamp, headers: write_feature_to_redis(
                r, estrai_machine_id(key), "max_"+feature+"_1min", round(result["value"], 2)
            ),
            metadata=True
        )
        # Media 5 minuti
        sdf_sliding = (
            sdf
            .apply(lambda row: row[feature])
            .sliding_window(duration_ms=WINDOW_5MIN, name="avg_" + feature + "_5min")
            .mean()
            .current()
        )
        sdf_sliding.update(
            lambda result, key, timestamp, headers: write_feature_to_redis(
                r, estrai_machine_id(key), "avg_"+feature+"_5min", round(result["value"], 2)
            ),
            metadata=True
        )



    logger.info(f"Pipeline avviata: {INPUT_TOPIC} → Redis")
    app.run()


if __name__ == "__main__":
    run_streaming_pipeline()
