import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid

from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, LongType, TimestampType
)
from loguru import logger

from feature_generator import MachineDataGenerator, create_factory_fleet


# ── Configurazione ────────────────────────────────────────
# I path arrivano dalle variabili d'ambiente impostate nel docker-compose; il
# volume mappa la cartella host TRUE_PIPELINE/data su /app/data nel container.
DATA_DIR   = Path(os.getenv("DATA_DIR", "/app/data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/data/features"))


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("FaultDetection-BatchFeaturePipeline")
        .master("local[4]")  
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )



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


def load_features(spark):

    input_path = DATA_DIR / "raw_features.parquet"

    if not input_path.exists():
        logger.info("File non trovato — genero dati sintetici")
        df = _generate_synthetic_data(spark, n_machines=12, n_seconds=3600)
        df.write.mode("overwrite").parquet(str(input_path))
        logger.info(f"Dati sintetici salvati in {input_path}")
        return df

    logger.info(f"Caricamento rilevazioni da {input_path}")
    return spark.read.schema(FEATURE_SCHEMA).parquet(str(input_path)).repartition(4)


def _generate_synthetic_data(spark: SparkSession, n_machines: int, n_seconds: int):

    # Punto di partenza: n_seconds nel passato
    start: datetime = datetime.now(timezone.utc) - timedelta(seconds=n_seconds)
    # Conversione in timestamp (secondi da EPOCH)
    start: float = start.timestamp()
    # Conversione in millisecondi
    start: int = int(start * 1000)

    records = []

    factory_fleet = create_factory_fleet(n_machines)

    for s in range(n_seconds):

        for machine in factory_fleet:
            data = machine.step()

            records.append((
                data["machine_id"],
                start + (s * 1000), # timestamp in millisecondi, intero
                data["temperature_C"],
                data["vibration_mm_per_s"],
                data["pressure_kPa"],
                data["rotor_speed_rpm"],
                data["machine_type"],
                int(data["faulty"])
            ))

    return spark.createDataFrame(records, schema=FEATURE_SCHEMA)


def compute_features(df):
    """
    Costruisce la tabella di training
    """
    logger.info("Creazione features per il training...")

    # Conversione timestamp_ms in colonna timestamp
    df = df.withColumn(
        "event_timestamp",
        (f.col("timestamp_ms") / 1000).cast(TimestampType())
    )

    # Window temporali per macchina, escludendo la rilevazione corrente
    # (rangeBetween fino a -1 ms)

    ONE_HOUR_MS = 1 * 60 * 60 * 1000
    # ONE_DAY_MS = 24 * 60 * 60 * 1000
    FIVE_MIN_MS = 5 * 60 * 1000
    ONE_MIN_MS = 60 * 1000

    def finestra_macchina(durata_ms: int):
        return (
            Window.partitionBy("machine_id")
            .orderBy("timestamp_ms")
            .rangeBetween(-durata_ms, -1)
        )

    window_1h = finestra_macchina(ONE_HOUR_MS)
    window_5min = finestra_macchina(FIVE_MIN_MS)
    window_1min = finestra_macchina(ONE_MIN_MS)

    df_features = df.select(
        # ID macchina e timestamp
        "machine_id",
        "event_timestamp",

        # Campi della riga corrente
        "temperature_C",
        "vibration_mm_per_s",
        "pressure_kPa",
        "rotor_speed_rpm",
        "machine_type",

        # Aggregati storici della macchina (equivalente batch)
        f.coalesce(f.avg("temperature_C").over(window_1h), f.lit(0.0))
            .alias("avg_temperature_C_1h"),
        f.coalesce(f.avg("vibration_mm_per_s").over(window_1h), f.lit(0.0))
            .alias("avg_vibration_mm_per_s_1h"),
        f.coalesce(f.avg("pressure_kPa").over(window_1h), f.lit(0.0))
            .alias("avg_pressure_kPa_1h"),
        f.coalesce(f.avg("rotor_speed_rpm").over(window_1h), f.lit(0.0))
            .alias("avg_rotor_speed_rpm_1h"),

        # Aggregati "veloci" della macchina, equivalente streaming.
        f.coalesce(f.avg("temperature_C").over(window_5min), f.lit(0.0))
            .alias("avg_temperature_C_5min"),
        f.coalesce(f.max("temperature_C").over(window_1min), f.lit(0.0))
            .alias("max_temperature_C_1min"),

        f.coalesce(f.avg("vibration_mm_per_s").over(window_5min), f.lit(0.0))
        .alias("avg_vibration_mm_per_s_5min"),
        f.coalesce(f.max("vibration_mm_per_s").over(window_1min), f.lit(0.0))
        .alias("max_vibration_mm_per_s_1min"),

        f.coalesce(f.avg("pressure_kPa").over(window_5min), f.lit(0.0))
        .alias("avg_pressure_kPa_5min"),
        f.coalesce(f.max("pressure_kPa").over(window_1min), f.lit(0.0))
        .alias("max_pressure_kPa_1min"),

        f.coalesce(f.avg("rotor_speed_rpm").over(window_5min), f.lit(0.0))
        .alias("avg_rotor_speed_rpm_5min"),
        f.coalesce(f.max("rotor_speed_rpm").over(window_1min), f.lit(0.0))
        .alias("max_rotor_speed_rpm_1min"),

        # Label della rilevazione corrente
        "faulty",
    )

    # Rimuoviamo rilevazioni senza storico per max.
    for feature in [
        "max_temperature_C_1min",
        "max_vibration_mm_per_s_1min",
        "max_pressure_kPa_1min",
        "max_rotor_speed_rpm_1min",
    ]:
        df_features = df_features[df_features[feature] != 0 ]

    logger.info(f"Tabella di training: {df_features.count()} rilevazioni")
    return df_features


def save_features(df, output_dir):
    """Salva la tabella di training in Parquet"""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "features.parquet")

    logger.info(f"Salvataggio tabella di training in {output_path}")
    (
        df.coalesce(1)   # un solo file Parquet per semplicità
        .write
        .mode("overwrite")
        .parquet(output_path)
    )
    logger.info("Tabella salvata")


def run_batch_pipeline():
    """Entry point della Batch Feature Pipeline."""

    spark = create_spark_session()

    # 1. Carica dati grezzi
    features_df = load_features(spark)
    logger.info(f"Rilevazioni caricate: {features_df.count()}")

    # 2. Calcola feature derivate
    features_df = compute_features(features_df)

    # 3. Salva in Parquet (tabella di training letta dalla Training Pipeline)
    save_features(features_df, OUTPUT_DIR)

    # 4. Mostra anteprima
    logger.info("Anteprima tabella di training:")
    features_df.select(
        "machine_id", "temperature_C",
        "avg_temperature_C_1h",
        "avg_temperature_C_5min",
        "max_temperature_C_1min",
        "vibration_mm_per_s",
        "pressure_kPa",
        "rotor_speed_rpm",
        "machine_type",
        "faulty"
    ).show(10, truncate=False)

    spark.stop()
    logger.info("Batch Pipeline completata")

if __name__ == "__main__":
    run_batch_pipeline()
