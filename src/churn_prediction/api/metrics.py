from prometheus_client import (
    Counter,
    Histogram,
)

PREDICTION_REQUESTS = Counter(
    "churn_prediction_requests_total",
    "Total prediction requests reaching the endpoint",
)

PREDICTION_ERRORS = Counter(
    "churn_prediction_errors_total",
    "Total prediction requests that failed during inference",
)

PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Time spent performing model inference",
)

PREDICTION_OUTCOMES = Counter(
    "churn_prediction_outcomes_total",
    "Total predictions grouped by predicted class",
    ["prediction"],
)