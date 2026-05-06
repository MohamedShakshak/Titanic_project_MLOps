#!/bin/bash
set -e

echo "Starting MLflow Inference Server"
echo "Tracking URI : $MLFLOW_TRACKING_URI"
echo "Model name   : $MODEL_REGISTRY_NAME"
echo "Model alias  : $MODEL_ALIAS"
echo "Port         : $PORT"

# Use alias URI — @Production instead of /Production
MODEL_URI="models:/${MODEL_REGISTRY_NAME}@${MODEL_ALIAS}"
echo "Loading model from: $MODEL_URI"

exec /app/.venv/bin/mlflow models serve \
  --model-uri "$MODEL_URI" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --no-conda \
  --timeout 60