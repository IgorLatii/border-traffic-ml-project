# Data

This folder contains the synthetic (non-operational) datasets used in the project.

## Files
- `traffic_people_hour.csv`
  - Hourly aggregated border-crossing counts for persons.
  - Columns include: `pctf`, `date`, `day_of_week`, `hour`, `in_count`, `out_count`, `total`,
    and citizenship groups (e.g., `md`, `ro`, `ua`, `eu`, `other`).

- `traffic_auto_hour.csv`
  - Hourly aggregated border-crossing counts for vehicles.
  - Columns include: `pctf`, `date`, `day_of_week`, `hour`, `in_count`, `out_count`, `total`,
    and vehicle categories split by direction (cars/buses/light trucks/trucks).

## Notes on data protection
The datasets are derived from aggregated statistics and additionally transformed for academic use.
They preserve temporal patterns and internal proportions while preventing direct correspondence
with operational values. Raw/internal datasets are not included in this repository.

## Expected invariants (per row)
Persons:
- `in_count + out_count = total`
- `md + ro + ua + eu + other = total`

Vehicles:
- `in_count + out_count = total`
- `car_in + bus_in + light_trucks_in + trucks_in = in_count`
- `car_out + bus_out + light_trucks_out + trucks_out = out_count`
