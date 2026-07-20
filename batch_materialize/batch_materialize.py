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

import sys
import os
from datetime import timedelta
from pathlib import Path

import redis
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, LongType, TimestampType
    )
from loguru import logger

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
KEY_PATTERN = "feature:machine_id={machine_id}:{feature_name}"
FEATURE_TTL_SECONDS = 60 * 60 

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))

FEATURE_SCHEMA = StructType([
    StructField("machine_id",     IntegerType(), False),
    StructField("timestamp_ms", LongType(), False),
    StructField("temperature_C",         DoubleType(),  False),
    StructField("vibration_mm_per_s", DoubleType(),  False),
    StructField("pressure_kPa", DoubleType(),  False),
    StructField("rotor_speed_rpm", DoubleType(),  False),
    StructField("machine_type", StringType(), True),
    StructField("faulty",       IntegerType(), False)
])

# Le tre feature batch 24h che il modello usa davvero (stesse lette dal serving, 06).
BATCH_FEATURE_NAMES = [
    "avg_temperature_C_1h",
    "avg_vibration_mm_per_s_1h",
    "avg_pressure_kPa_1h",
    "avg_rotor_speed_rpm_1h",
]

def get_redis_client() -> redis.Redis:
    """Apre la connessione a Redis"""
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def write_feature_to_redis(r, machine_id, feature_name, value):
    """Scrive una feature su Redis con scadenza"""
    key = KEY_PATTERN.format(machine_id=machine_id, feature_name=feature_name)
    r.setex(key, FEATURE_TTL_SECONDS, str(value))


def create_spark_session():
    return (
        SparkSession.builder
        .appName("FaultDetection-BatchFeaturePipeline")
        .master("local[4]")  
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def load_features(spark):
    input_path = DATA_DIR / "raw_features.parquet"

    if not input_path.exists():
        logger.info("File non trovato")
        sys.exit(1)

    logger.info(f"Caricamento transazioni da {input_path}")
    return spark.read.schema(FEATURE_SCHEMA).parquet(str(input_path)).repartition(4)


def materialize_online_store(features_df) -> int:
    """
    Scrive su Redis le feature Batch
    """
    df = features_df.withColumn(
        "event_timestamp",
        (F.col("timestamp_ms") / 1000).cast(TimestampType())
    )

    # Prendo il lasso di tempo di dati più recenti
    now_ts = df.agg(F.max("event_timestamp")).first()[0]
    cutoff = now_ts - timedelta(hours=1)
    logger.info(f"Materializzazione: finestra 1h da {cutoff} a {now_ts}")

    # Un'unica groupBy per macchina sull'ultima ora
    aggregati = (
        df.filter(F.col("event_timestamp") > F.lit(cutoff))
        .groupBy("machine_id")
        .agg(
            F.avg("temperature_C").alias("avg_temperature_C_1h"),
            F.avg("vibration_mm_per_s").alias("avg_vibration_mm_per_s_1h"),
            F.avg("pressure_kPa").alias("avg_pressure_kPa_1h"),
            F.avg("rotor_speed_rpm").alias("avg_rotor_speed_rpm_1h"),
        )
    )

    # Scrittura su Redis 
    r = get_redis_client()
    righe = aggregati.collect()

    # Una pipeline Redis accumula i comandi e li spedisce tutti in un solo round-trip di rete
    pipe = r.pipeline()
    for riga in righe:
        for nome_feature in BATCH_FEATURE_NAMES:
            write_feature_to_redis(pipe, riga["machine_id"], nome_feature, riga[nome_feature])
    pipe.execute()

    logger.info(f"{len(righe)} dati accumulati sulle macchine scritti su Redis")
    return len(righe)


def batch_pipeline():

    spark = create_spark_session()

    # Carica i dati
    features_df = load_features(spark)
    logger.info(f"Features caricate: {features_df.count()}")

    # Materializzazione su Redis
    materialize_online_store(features_df)

    spark.stop()
    logger.info("Batch Pipeline completata")


if __name__ == "__main__":
    batch_pipeline()
