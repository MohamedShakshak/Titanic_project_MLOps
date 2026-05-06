# trainer.py
from dotenv import load_dotenv
import mlflow.sklearn
load_dotenv()

import os
import hydra
import yaml
import joblib
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from src.training.data.download_data import download_data
from src.training.data.load import load_raw_data, save_processed_data
from src.training.evaluate import evaluate_model, save_metrics
from src.training.features.preprocess import build_preprocessor
from src.training.pipeline import build_pipeline
from src.training.train import get_model, get_model_params

logger = logging.getLogger(__name__)

def setup_mlflow(cfg: DictConfig) -> None:
    """Configure MLflow tracking URI and experiment."""
    
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI",  "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    logger.info("MLflow tracking URI: %s", tracking_uri)
    logger.info("MLflow experiment: %s", cfg.mlflow.experiment_name)

def run_download(cfg: DictConfig) -> None:
    download_data(cfg)

def run_train(cfg: DictConfig) -> None:
    train_df= load_raw_data(cfg)

    X = train_df.drop(columns=["Survived", "PassengerId"])
    y = train_df["Survived"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
    )
    logger.info("Train size: %d, Val size: %d", len(X_train), len(X_val))

    processed_dir = Path(to_absolute_path(cfg.data.processed_path))
    save_processed_data(X_train, X_val, y_train, y_val, processed_dir)

    pipeline = build_pipeline(build_preprocessor(), get_model(cfg))
    selected_model_params = get_model_params(cfg)
    
    with mlflow.start_run(run_name= cfg.model.name):
        mlflow.log_params({
            "model_name": cfg.model.name,
            "test_size": cfg.training.test_size,
            "random_state": cfg.training.random_state,
            **selected_model_params,
        })
        pipeline.fit(X_train, y_train)
        logger.info("Training complete.")

        metrics = evaluate_model(pipeline, X_val, y_val)
        for k, v in metrics.items():
            logger.info("%s: %.4f", k, v)
        
        mlflow.log_metrics(metrics)
        
        signature = infer_signature(X_train, pipeline.predict(X_val))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="models",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name="titanic-classifier",
        )
        mlflow.log_artifact("params.yaml")
        run_id = mlflow.active_run().info.run_id
        logger.info("MLflow run ID: %s", run_id)
        
        

    metrics_dir = Path(to_absolute_path(cfg.training.metrics_output_dir))
    save_metrics(metrics, cfg.model.name, metrics_dir)

    models_dir = Path(to_absolute_path(cfg.training.model_output_dir))
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models_dir / f"{cfg.model.name}_pipeline.pkl")
    logger.info("Saved pipeline to %s", models_dir)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Load params.yaml and merge it ON TOP of Hydra config
    # This means DVC's changes to params.yaml take effect
    OmegaConf.set_struct(cfg, False)
    params = OmegaConf.load("./params.yaml")
    cfg = OmegaConf.merge(cfg, params)
    
    logger.info("Pipeline started — stage: %s", cfg.stage)
    logger.info("Parameters:\n%s", OmegaConf.to_yaml(cfg))

    setup_mlflow(cfg)
    if cfg.stage == "download":
        run_download(cfg)
    elif cfg.stage == "train":
        run_train(cfg)
    elif cfg.stage == "all":
        run_download(cfg)
        run_train(cfg)
    else:
        raise ValueError(f"Unknown stage: {cfg.stage}")

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
