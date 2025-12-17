# SQL

This folder contains *conceptual* SQL queries used to illustrate how the hourly datasets can be
built from a transactional border-crossing fact table.

Important:
- Queries are intentionally written as "concepts" and may require adaptation to a specific schema.
- The purpose is to document the aggregation logic and feature construction used in the project.

## Files
- `concept_vehicles_hourly.sql`
  - Builds an hourly panel (PCTF × date × hour) and aggregates vehicle traffic by direction
    and vehicle category.

- `concept_people_hourly.sql`
  - Builds an hourly panel (PCTF × date × hour) and aggregates person traffic by direction
    and citizenship groups, including an EU/non-EU split using a reference table.
