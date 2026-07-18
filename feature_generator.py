"""
Generatore di dati sintetici
"""

import random
from datetime import datetime, timedelta

# --- Configurazione

# Numero di step di transizione da healthy a faulty e viceversa (degrading, recovering)
transition_steps = 5

# Medie delle rilevazioni dei sensori per macchine funzionanti correttamente
healthy_means = {
    "temperature_C": 45.0,
    "vibration_mm_per_s": 2.5,
    "pressure_kPa": 101.0,
    "rotor_speed_rpm": 1500.0,
}
# Medie delle rilevazioni dei sensori per macchine in avaria
faulty_means = {
    "temperature_C": 49.0,
    "vibration_mm_per_s": 3.2,
    "pressure_kPa": 102.5,
    "rotor_speed_rpm": 1475.0,
}
# Dev. standard delle rilevazioni dei sensori per macchine funzionanti correttamente
healthy_std_devs = {
    "temperature_C": 3.0,
    "vibration_mm_per_s": 0.8,
    "pressure_kPa": 2.0,
    "rotor_speed_rpm": 15.0,
}
# Dev. standard delle rilevazioni dei sensori per macchine in avaria
faulty_std_devs = {
    "temperature_C": 4.0,
    "vibration_mm_per_s": 0.5,
    "pressure_kPa": 2.5,
    "rotor_speed_rpm": 40.0,
}


# --- Classe generatore dati
class MachineDataGenerator:
    def __init__(self, machine_id: int, machine_type: str):
        self.machine_id = machine_id
        self.machine_type = machine_type

        # Stato di funzionamento della macchina
        self.phase = "healthy"  # "healthy", "degrading", "faulty", "recovering"

        self.timer = 0

        # Valori della rilevazione precedente per questa macchina
        self.value_previous = healthy_means.copy()

    def _state_transition(self):

        if self.phase == "healthy":
            # 2% di probabilità di innescare una avaria a ogni step
            # TODO: prob a 5% quando sotto carico
            if random.random() < 0.02:
                self.phase = "degrading"
                self.timer = transition_steps

        elif self.phase == "degrading":
            self.timer -= 1
            if self.timer == 0:
                self.phase = "faulty"
                # L'avaria persiste per 3-7 turni
                self.timer = random.randint(3, 7)

        elif self.phase == "faulty":
            self.timer -= 1
            if self.timer == 0:
                self.phase = "recovering"
                self.timer = transition_steps

        elif self.phase == "recovering":
            self.timer -= 1
            if self.timer == 0:
                self.phase = "healthy"

    @property
    def alpha(self):

        if self.phase == "healthy":
            return 0

        elif self.phase == "degrading":
            # Cresce linearmente fino a 1.0
            return 1.0 - (self.timer / transition_steps)

        elif self.phase == "faulty":
            return 1

        elif self.phase == "recovering":
            # Diminuisce linearmente fino a 0.0
            return self.timer / transition_steps

        else:
            # Errore: non dovrebbe essere accessibile
            print(f"Errore di generazione, fase '{self.phase}' sconosciuta.")
            return 0

    def step(self) -> dict:
        # 1. Transizione stato di funzionamento
        #    e calcolo del parametro alpha
        #    (0.0 = funzionamento corretto, 1.0 = avaria)
        self._state_transition()

        # 2. Generazione dati

        # Leggero aumento se il macchinario è sotto carico
        # TODO: timer macchina sotto carico
        is_under_load = random.choice([True, False])
        load_multiplier = 1.05 if is_under_load else 1.0

        record = {
            "machine_id": self.machine_id,
            "machine_type": self.machine_type,
            "is_under_load": is_under_load,
            "faulty": self.phase == "faulty",
        }

        # Calcolo fattore di interpolazione tra stati
        alpha = self.alpha

        # Genera una rilevazione in base all'alpha attuale
        # e alla rilevazione precedente
        # per ogni sensore della macchina
        for sensor in [
            "temperature_C",
            "vibration_mm_per_s",
            "pressure_kPa",
            "rotor_speed_rpm",
        ]:
            # Media del nuovo valore basata su stato di salute della macchina
            current_mean = (
                healthy_means[sensor] * (1 - alpha) + faulty_means[sensor] * alpha
            )

            # Leggero aumento se la macchina è sotto sforzo
            current_mean *= load_multiplier

            # Dev. Standard del nuovo valore
            current_std_dev = (
                healthy_std_devs[sensor] * (1 - alpha) + faulty_std_devs[sensor] * alpha
            )

            # Genera un valore casuale distribuito normalmente intorno alla media interpolata
            val = random.normalvariate(current_mean, current_std_dev)

            # Calcola la media aritmetica tra il nuovo valore e il valore precedente
            val = (val + self.value_previous[sensor]) / 2.0

            # Registrazione della rilevazione
            record[sensor] = round(val, 2)

            # Sostituzione valore conservato in memoria
            self.value_previous[sensor] = val

        return record


def create_factory_fleet(n_machines: int) -> list:
    """Crea una lista di oggetti MachineDataGenerator suddivisi in tre tipi: A, B, C)."""
    fleet = []

    if n_machines < 1:
        return fleet

    for i in range(1, n_machines):
        if (i-1) % 3 == 0: # 1, 4, 7, 10...
            machine_type = "A"

        elif (i-2) % 3 == 0: # 2, 5, 8, 11...
            machine_type = "B"

        else: # 3, 6, 9, 12...
            machine_type = "C"

        fleet.append(
            MachineDataGenerator(machine_id=i, machine_type=machine_type)
        )

    return fleet


# --- Test di funzionamento

def __batch_test():
    # Inizializza 12 macchine
    factory_fleet = create_factory_fleet(12)
    batch_data = []

    logical_time = datetime(2026, 1, 1, 0, 0, 0)

    # Generate 10 time-steps (resulting in 120 total rows)
    for _ in range(10):
        for machine in factory_fleet:
            record = machine.step(logical_time)
            batch_data.append(record)

        # Advance logical time by 1 second for the whole factory
        logical_time += timedelta(seconds=1)

    # Convert to a PySpark DataFrame
    # df = spark.createDataFrame(batch_data)
    # df.show(24) # Show the first two seconds of data across the factory

    print(*batch_data, sep="\n")


def __stream_test():
    import time

    # Initialize the 12 machines
    factory_fleet = create_factory_fleet(12)

    while True:
        current_time = datetime.now()

        # Generate data for all 12 machines at this exact second
        for machine in factory_fleet:
            record = machine.step(current_time)

            # producer.send(value=record)
            # print(f"Streaming Sent: {record['machine_id']} | Faulty: {record['faulty']}")
            print(record)

        # Wait 1 second before the next factory-wide reading
        time.sleep(1)


if __name__ == "__main__":
    __batch_test()
    #__stream_test()
