# VisionQC — Monitoring Guide

## Overview

VisionQC uses **Prometheus** and **Grafana** to monitor the FastAPI inference API in real time.

```
┌──────────────┐     scrape /metrics     ┌─────────────┐     query      ┌──────────────┐
│  FastAPI API  │ ◄───────────────────── │  Prometheus  │ ◄──────────── │   Grafana    │
│  :8000        │    (every 10-15s)      │  :9090       │               │   :3000      │
└──────────────┘                         └─────────────┘               └──────────────┘
```

## What Each Component Does

### Prometheus
- **Time-series database** that scrapes metrics from the API at regular intervals.
- Stores metric data with timestamps.
- Provides a query language (PromQL) for analyzing metrics.
- Access the Prometheus UI at `http://localhost:9090`.

### Grafana
- **Visualization dashboard** that queries Prometheus and displays metrics as graphs, counters, and gauges.
- Pre-configured with the "VisionQC API Monitoring" dashboard.
- Access at `http://localhost:3000` (default credentials: `admin` / `admin`).

### FastAPI `/metrics` Endpoint
- The VisionQC API exposes metrics in Prometheus format at `GET /metrics`.
- Uses the `prometheus_client` Python library.
- The `/metrics` endpoint itself is excluded from request counters to avoid self-referential inflation.

## Metrics Exposed

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests (labels: method, endpoint, status_code) |
| `http_request_duration_seconds` | Histogram | Request latency distribution (labels: method, endpoint) |
| `http_requests_in_progress` | Gauge | Currently active requests (labels: method, endpoint) |
| `visionqc_predictions_total` | Counter | Total prediction requests (labels: status — success/error/no_model) |
| `visionqc_inference_duration_seconds` | Histogram | YOLO model inference time only |

## How Prometheus Scrapes Metrics

1. Prometheus runs as a Docker container on port `9090`.
2. Configuration in `monitoring/prometheus/prometheus.yml` tells Prometheus to scrape `visionqc-api:8000/metrics` every 10 seconds.
3. Prometheus resolves the hostname `visionqc-api` via the shared Docker network `visionqc-net`.
4. Scraped metrics are stored locally with a 7-day retention period.

## How Grafana Visualizes Metrics

1. Grafana runs as a Docker container on port `3000`.
2. The Prometheus data source is provisioned automatically from `monitoring/grafana/provisioning/datasources/prometheus.yml`.
3. The "VisionQC API Monitoring" dashboard is provisioned from `monitoring/grafana/provisioning/dashboards/visionqc-api.json`.
4. Dashboard auto-refreshes every 10 seconds.

## Dashboard Panels

The provisioned "VisionQC API Monitoring" dashboard includes:

1. **Total API Requests** — Running count of all HTTP requests.
2. **Request Rate (per second)** — Current throughput.
3. **Avg Response Latency** — Mean response time.
4. **Total Predictions** — Count of `/predict` calls.
5. **Request Rate Over Time** — Time-series graph by endpoint.
6. **Response Latency (p50/p95/p99)** — Percentile latency distribution.
7. **HTTP Status Codes Over Time** — Success vs error rates.
8. **Model Inference Duration** — YOLO inference time (p50/p95).
9. **Prediction Requests by Status** — Success vs error prediction breakdown.

## Starting the Monitoring Stack

### Prerequisites
- The VisionQC API must be running (via `docker-compose.yml` or locally).
- Docker Engine in WSL (Ubuntu-D).

### Commands (run from WSL, in the project directory)

```bash
# 1. Start the API (creates the visionqc-net network)
docker compose up -d

# 2. Start monitoring (joins the same network)
docker compose -f docker-compose.monitoring.yml up -d
```

### Access Points
| Service | URL |
|---------|-----|
| VisionQC API | http://localhost:8000 |
| API Swagger Docs | http://localhost:8000/docs |
| Prometheus Metrics | http://localhost:8000/metrics |
| Prometheus UI | http://localhost:9090 |
| Prometheus Targets | http://localhost:9090/targets |
| Grafana | http://localhost:3000 |

## Generating Test Traffic

To populate the dashboard with data, make several API calls:

```bash
# Health check
curl http://localhost:8000/health

# Prediction (replace with an actual image path)
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/pcb_image.jpg"

# Loop for traffic generation
for i in $(seq 1 20); do
  curl -s http://localhost:8000/health > /dev/null
  echo "Request $i"
done
```

## Verifying Metrics Are Working

1. **Check Prometheus targets**: Visit `http://localhost:9090/targets`. The `visionqc-api` target should show state `UP`.

2. **Query a metric**: In the Prometheus UI, enter `http_requests_total` and click Execute. You should see counter values.

3. **Check Grafana**: Open `http://localhost:3000`, navigate to Dashboards → VisionQC API Monitoring. Panels should show data within 15-30 seconds of making API calls.

## Troubleshooting

### Prometheus Target is DOWN
- Verify the API is running: `docker ps | grep visionqc-api`
- Check network connectivity: `docker exec visionqc-prometheus wget -q -O- http://visionqc-api:8000/metrics`
- Ensure both containers are on the `visionqc-net` network: `docker network inspect visionqc-net`

### Grafana Shows "No Data"
- Wait at least 30 seconds after starting for data to accumulate.
- Make some API requests to generate metrics.
- Check the Prometheus data source in Grafana: Settings → Data Sources → Prometheus → Test.

### Missing Metrics
- Verify `prometheus_client` is installed: `pip list | grep prometheus`
- Check the `/metrics` endpoint directly: `curl http://localhost:8000/metrics`

## Stopping the Monitoring Stack

```bash
# Stop monitoring only
docker compose -f docker-compose.monitoring.yml down

# Stop API
docker compose down

# Stop everything and remove volumes (data loss!)
docker compose -f docker-compose.monitoring.yml down -v
docker compose down -v
```

## Resource Considerations

This monitoring stack is designed for a resource-constrained CPU-only laptop:
- Prometheus limited to 512 MB RAM, 0.5 CPU.
- Grafana limited to 512 MB RAM, 0.5 CPU.
- Prometheus data retention set to 7 days.
- Scrape interval is 10-15 seconds (not aggressive).

