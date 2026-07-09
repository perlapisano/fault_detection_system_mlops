"""
Feature Simulation Source
==================================
Simula un flusso continuo di dati e li pubblica
su un topic Redpanda/Kafka.

Concetti MLOps dimostrati:
  - Source personalizzata con Quixstreams
  - Produzione di messaggi JSON su Kafka
  - Simulazione di pattern
"""

import os
import time
import random
import uuid
from datetime import datetime, timezone

from quixstreams import Application
from quixstreams.sources import Source
from loguru import logger


# ── Configurazione ────────────────────────────────────────
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
TOPIC_NAME = os.getenv("TOPIC_NAME", "feature-simulator")

TPS = float(os.getenv("RILEVAZIONI_AL_S", "5"))

COUNTRIES = ["IT", "US", "CH", "RO", "FR", "GE", "SP"]

COUNTRIES_NUM = len(COUNTRIES)

CARDINAL_DIRECTION = ["N", "S", "E", "W"]
STATION_IDS = list(range(1, 201))

ID_TO_COUNTRY = {i: COUNTRIES[i % COUNTRIES_NUM] for i in STATION_IDS}

# Velocità del vento
NORMAL_AMOUNT_MIN = 0.0
NORMAL_AMOUNT_MAX = 300.0
WIND_SPEED_MEAN = 5.0
WIND_SPEED_SD = 2.5

# TODO: paesi diversi possono essere più o meno ventosi
# NORMAL_COUNTRY_WEIGHTS = [0.60, 0.20, 0.12, 0.08]
# NORMAL_MERCHANT_WEIGHTS = [0.40, 0.30, 0.20, 0.10]


class DataSimulatorSource(Source):
    """
    Source personalizzata: genera rilevazioni della velocità e direzione del vento sintetiche a ritmo costante.

    Il metodo run() è il cuore della Source: gira finché l'app è attiva.
    Per ogni iterazione:
      1. Genera una rilevazione (dizionario Python)
      2. La serializza con self.serialize()
      3. La pubblica su Kafka con self.produce()
    """

    def __init__(self, tps: float = 5.0):
        super().__init__(name="tx-simulator")
        self.tps = tps
        self.sleep_time = 1.0 / tps

    def run(self):
        logger.info(f"Simulatore avviato — {self.tps} rilevazioni/secondo")
        count = 0

        while self.running:
            data = self._generate_data()

            # serialize() prepara key+value per Kafka
            msg = self.serialize(
                key=str(data["station_id"]),
                value=data,
            )
            self.produce(value=msg.value, key=msg.key)

            count += 1
            if count % 50 == 0:
                logger.info(f"Prodotte {count} rilevazioni")

            time.sleep(self.sleep_time)

    def _generate_data(self) -> dict:
        """
        Genera un singolo dato.
        """

        station_id = random.choice(STATION_IDS)

        wind_speed = max(
            NORMAL_AMOUNT_MIN,
            round(random.normalvariate(WIND_SPEED_MEAN, WIND_SPEED_SD), 2),
        )

        cardinal_direction = random.choice(CARDINAL_DIRECTION)

        return {
            "measurement_id": str(uuid.uuid4()),
            "station_id": station_id,
            "wind_speed": wind_speed,
            "cardinal_direction": cardinal_direction,
            "country": ID_TO_COUNTRY[station_id],
            "timestamp_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
        }


# ── Entry point ───────────────────────────────────────────
def main():
    app = Application(
        broker_address=KAFKA_BROKER,
        consumer_group="tx-simulator-group",
    )

    topic = app.topic(TOPIC_NAME, value_serializer="json")

    source = DataSimulatorSource(tps=TPS)

    # topic passato subito qui: la Source scrive direttamente su feature-simulator,
    # invece che sul suo topic interno di default (source__tx-simulator)
    sdf = app.dataframe(topic=topic, source=source)

    logger.info(f"Connesso a {KAFKA_BROKER}, topic → {TOPIC_NAME}")
    app.run()


if __name__ == "__main__":
    main()
