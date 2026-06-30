# Research Data Contracts

This directory contains small, redistributable research data contracts used when
private local data and generated outputs are unavailable.

The project separates:

- stable catalog snapshots: durable place, hotel, source, and curated annotation
  records;
- context snapshots: time-sensitive weather, routing, closure, hotel-rate, or
  disruption observations;
- run artifacts: solver inputs, parent plans, repaired plans, evaluations, and
  dashboards generated for one experiment run.

The first committed fallback snapshot is `snapshots/california_v1`. It is a
small California planning seed, not a comprehensive tourism dataset. It is
intended to keep clean clones auditable and runnable without committing private
Yelp files or generated `results/outputs` artifacts.

