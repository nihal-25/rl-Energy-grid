# preprocess_opsd.py
import pandas as pd

# Load CSV
df = pd.read_csv("time_series_60min_singleindex.csv", index_col=0, parse_dates=True)

# Pick Germany (DE) load + renewables
load = df["DE_load_actual_entsoe_transparency"]
solar = df["DE_solar_generation_actual"]
wind = df["DE_wind_generation_actual"]

# Merge into one DataFrame
data = pd.concat([load, solar, wind], axis=1).dropna()
data.columns = ["load", "solar", "wind"]

# Normalize MW scale for simplicity
data = data / 1000.0  # GW

# Save preprocessed
data.to_csv("germany_energy.csv")
print("Saved germany_energy.csv with columns: load, solar, wind")
