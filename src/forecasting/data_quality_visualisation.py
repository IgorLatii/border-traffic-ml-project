from src.forecasting.io import load_hourly_csv
from src.forecasting.data_quality import (
    check_missing_values,
    check_negative_values,
    check_hour_range,
    check_complete_hourly_panel,
)

df = load_hourly_csv("../../data/traffic_auto_hour.csv")
#df = load_hourly_csv("../../data/traffic_people_hour.csv")

print("Missing values:")
print(check_missing_values(df))

count_cols = [
    "intrare_auto", "iesire_auto",
    "car_in", "car_out",
    "bus_in", "bus_out",
    "light_trucks_in", "light_trucks_out",
    "trucks_in", "trucks_out",
]

print("Negative values:")
print(check_negative_values(df, count_cols))

print("Hour range OK:", check_hour_range(df))

missing_panel = check_complete_hourly_panel(df)
print("Missing (pctf,date,hour) rows:", len(missing_panel))
