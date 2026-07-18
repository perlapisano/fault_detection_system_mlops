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

from feature_generator import MachineDataGenerator, create_factory_fleet


# ── Configurazione ────────────────────────────────────────
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
TOPIC_NAME = os.getenv("TOPIC_NAME", "feature-simulator")

TPS = float(os.getenv("RILEVAZIONI_AL_S", "1"))


class DataSimulatorSource(Source):
    """
    Source personalizzata: genera rilevazioni dei sensori a ritmo costante.

    Il metodo run() è il cuore della Source: gira finché l'app è attiva.
    Per ogni iterazione:
      1. Genera una rilevazione (dizionario Python)
      2. La serializza con self.serialize()
      3. La pubblica su Kafka con self.produce()
    """

    def __init__(self, tps: float = 1.0):
        super().__init__(name="feature-simulator")
        self.tps = tps
        self.sleep_time = 1.0 / tps

        self.factory_fleet = create_factory_fleet(12)

    def run(self):
        logger.info(f"Simulatore avviato — {self.tps} rilevazioni/secondo")
        count = 0

        while self.running:
            for machine in self.factory_fleet:
                # serialize() prepara key+value per Kafka

                data = machine.step()

                data["timestamp_ms"] = int(datetime.now(timezone.utc).timestamp() * 1000)

                msg = self.serialize(
                    key=str(data["machine_id"]),
                    value=data,
                )
                self.produce(value=msg.value, key=msg.key)

            count += 1
            if count % 10 == 0:
                logger.info(f"Prodotte {count} rilevazioni")

            time.sleep(self.sleep_time)

# --- Entry point
def main():
    app = Application(
        broker_address=KAFKA_BROKER,
        consumer_group="feature-simulator-group",
    )

    topic = app.topic(TOPIC_NAME, value_serializer="json")

    source = DataSimulatorSource(tps=TPS)

    # topic passato subito qui: la Source scrive direttamente su feature-simulator,
    # invece che sul suo topic interno di default
    sdf = app.dataframe(topic=topic, source=source)

    logger.info(f"Connesso a {KAFKA_BROKER}, topic → {TOPIC_NAME}")
    app.run()


if __name__ == "__main__":
    main()
