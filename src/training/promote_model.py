import os
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()

# Get experiment ID by name first
experiment = client.get_experiment_by_name("titanic-training")
if experiment is None:
    raise ValueError("Experiment 'titanic-training' not found. Run training first.")

# Find the best run by roc_auc
best_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="status = 'FINISHED'",
    order_by=["metrics.roc_auc DESC"],
    max_results=1,
)[0]

best_run_id = best_run.info.run_id
best_roc_auc = best_run.data.metrics["roc_auc"]
best_model = best_run.data.params["model_name"]

print(f"Best run: {best_run_id}")
print(f"Model: {best_model}")
print(f"ROC-AUC: {best_roc_auc:.4f}")

# Get the version registered from this run
versions = client.search_model_versions(f"name='titanic-classifier'")
best_version = next(v for v in versions if v.run_id == best_run_id)

if best_version is None:
    raise ValueError(
        f"No registered version found for run {best_run_id}. "
        "Make sure registered_model_name='titanic-classifier' is set in trainer.py."
    )
    
print(f"Found version: {best_version.version}")


client.set_registered_model_alias(
    name="titanic-classifier",
    alias="Production",
    version=best_version.version,
)

print(f"Set alias 'Production' → version {best_version.version}")
