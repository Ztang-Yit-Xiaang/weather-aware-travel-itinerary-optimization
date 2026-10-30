# Railway detects this root Dockerfile automatically. No secrets are build args.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    WEB_CONCURRENCY=1 \
    PRODUCT_COPILOT_ADAPTER=deterministic \
    HOSTED_ENABLE_PAID_OPENAI=false \
    PORT=8080

WORKDIR /app
COPY ["deploy/requirements-hosted.txt", "/tmp/requirements-hosted.txt"]
RUN python -m pip install --no-cache-dir --only-binary=:all: -r /tmp/requirements-hosted.txt \
    && python -m pip check \
    && groupadd --gid 10001 itinerary \
    && useradd --uid 10001 --gid itinerary --create-home itinerary

# Keep the full Python source tree: product_build_id fingerprints every .py file.
COPY ["src/", "./src/"]
COPY ["scripts/run_hosted_app.py", "scripts/run_product_app.py", "./scripts/"]
COPY ["configs/default_trip_config.yaml", "configs/product_app_registry.json", "configs/product_route_delta.json", "./configs/"]
COPY ["configs/interest_regions/", "./configs/interest_regions/"]
COPY ["configs/product_poi_catalogs/", "./configs/product_poi_catalogs/"]

# These are immutable registered packages, including their hash-bound matrices.
COPY ["runs/e3ux-weather-repair-demo-v6/", "./runs/e3ux-weather-repair-demo-v6/"]
COPY ["runs/california-coast-product-demo-v2/", "./runs/california-coast-product-demo-v2/"]
COPY ["runs/california-coast-coherent-demo-v1/", "./runs/california-coast-coherent-demo-v1/"]
COPY ["results/route_deltas/product_pfeiffer_point_lobos_20260908_v1/", "./results/route_deltas/product_pfeiffer_point_lobos_20260908_v1/"]

USER 10001:10001
# Verify the actual slim dependency set without creating an app or contacting DB/providers.
RUN --network=none PYTHONPATH=/app/src python -c "from itinerary_system.product_app.api import create_product_app; from itinerary_system.product_app.hosted import HostedSettings; import openai, psycopg"
EXPOSE 8080
# The launcher reads Railway's PORT and explicitly enforces one worker.
CMD ["python", "scripts/run_hosted_app.py"]
