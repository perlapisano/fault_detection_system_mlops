import os

from quixstreams import Application, State

from quixstreams.dataframe.windows import Mean

from quixstreams.dataframe.windows import aggregations as agg

from loguru import logger
from datetime import timedelta

# ── Configurazione ────────────────────────────────────────
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
TOPIC_NAME = os.getenv("TOPIC_NAME", "feature-simulator")


app = Application(
    broker_address=KAFKA_BROKER,
    consumer_group="wind-alerts-v1",
    auto_offset_reset="earliest",
)

feature_topic = app.topic(TOPIC_NAME, value_deserializer="json")
alerts_topic = app.topic("feature-alerts", value_serializer="json")

sdf = app.dataframe(topic=feature_topic)

# Esempio: conversione da C a F
"""sdf = sdf.apply(
    lambda v: {**v, "temperature_F": v["temperature_C"] * 9 / 5 + 32}
)"""

# log numero rilevazioni
# TODO: completare
def count_events(value: dict, state: State) -> dict:
    # Legge il contatore corrente per la chiave del messaggio (default 0)
    total = state.get("total", default=0)
    total += 1
    # Salva il nuovo valore nello state store
    state.set("total", total)
    value["event_count"] = total
    if total % 10:
        logger.info(f"Ricevute {total} rilevazioni dalla stazione")

    return value
# sdf = sdf.apply(count_events, stateful=True)


# Media velocità vento
# Il messaggio deve avere una chiave (es. sensor_id) per essere raggruppato
avg_sdf = (
    sdf.tumbling_window(
        duration_ms=timedelta(seconds=10),
        grace_ms=timedelta(seconds=1),  # tollera fino a 1s di dati in ritardo
    )
    # calcola la media del campo wind_speed
    .agg(avg_wind_speed=Mean(column="wind_speed"))
    .final()  # emette solo a finestra chiusa
)

avg_sdf = avg_sdf.to_topic(alerts_topic)

# window_def = TumblingWindow(
#     duration=timedelta(seconds=10),
#     grace=timedelta(seconds=1),
#     time_column="event_timestamp",  # default, change if your timestamp column differs
# )
# 
# windowed_sdf = sdf.apply_window(window_def)
# 
# avg_sdf = windowed_sdf.agg(mean_wind_speed_10s=Mean(column="wind_speed")).final()
# 
# avg_sdf.to_topic("feature-alerts")


# Filtro pericolo vento forte
# TODO: diverse categorie di intensità/pericolo

sdf = sdf[sdf["wind_speed"] > 6]  # 2. Filtra velocità sopra 6 per gli alert
sdf = sdf.update(lambda v: print(f"ALERT: {v}"))  # 3. Stampa ogni alert (debug)
sdf = sdf.to_topic(alerts_topic)  # 4. Produce l'alert sul topic di output

if __name__ == "__main__":
    app.run()  # bloccante: CTRL+C per uscire
