import json
import math
import torch
import mlflow
import dagshub
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader

from src.seq2seq.logger import logger
from src.seq2seq.component.model import Model
from src.seq2seq.utils.utils import get_package_info
from src.seq2seq.entity.entity import ModelEvaluationConfig
from src.seq2seq.config.configuration import ConfigurationManager


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) -> None:
        """Initialize the model evaluation pipeline."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.project_name, self.version = get_package_info()
        self._initialize()

    def _initialize(self) -> None:
        """Initialize MLflow tracking and load test data."""
        self._init_mlflow()
        self._load_data()

    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking with DagsHub."""
        try:
            dagshub.init(
                repo_owner=self.config.repo_owner,
                repo_name=self.config.repo_name,
                mlflow=True,
            )
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_experiment(f"v{self.version}")
            self.run_name = (
                f"v{self.version}--{datetime.now().strftime('%Y/%m/%d-%H:%M:%S')}"
            )
            logger.info("MLflow tracking initialized.")
        except Exception as e:
            logger.exception("Error initializing MLflow: %s", str(e))
            raise

    def _load_model(self) -> Model:
        """Load the trained model from the specified path."""
        try:
            logger.info(f"Loading model from: {self.config.model_path}")
            model = Model(config=ConfigurationManager().get_model_config()).to(
                self.device
            )
            model.load_state_dict(
                torch.load(self.config.model_path, map_location=self.device)
            )
            model.eval()
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.exception("Failed to load model: %s", str(e))
            raise

    def _load_data(self) -> None:
        """Load test input and target data from CSV files."""
        try:
            logger.info(
                "Loading test data from: %s, %s",
                self.config.test_input_path,
                self.config.test_target_path,
            )
            self.input_data = pd.read_csv(self.config.test_input_path)
            self.target_data = pd.read_csv(self.config.test_target_path)
        except Exception as e:
            logger.exception("Error loading test data: %s", str(e))
            raise

    def _get_git_commit_hash(self) -> str:
        """Retrieve the current Git commit hash for version tracking."""
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
            )
        except subprocess.CalledProcessError:
            logger.warning("Could not retrieve Git commit hash.")
            return "unknown"

    def _compute_bleu(
        self, reference: list[int], candidate: list[int], max_n: int = 4
    ) -> float:
        """Compute BLEU score for model evaluation."""
        try:
            weights = [1 / max_n] * max_n
            precisions = []
            for n in range(1, max_n + 1):
                ref_ngrams = Counter(
                    tuple(reference[i : i + n]) for i in range(len(reference) - n + 1)
                )
                cand_ngrams = Counter(
                    tuple(candidate[i : i + n]) for i in range(len(candidate) - n + 1)
                )
                match_count = sum(
                    min(ref_ngrams[ng], cand_ngrams[ng]) for ng in cand_ngrams
                )
                total_count = max(len(candidate) - n + 1, 1)
                precisions.append(match_count / total_count if total_count > 0 else 0)
            brevity_penalty = (
                math.exp(1 - len(reference) / len(candidate))
                if len(candidate) < len(reference)
                else 1
            )
            return brevity_penalty * math.exp(
                sum(w * math.log(p) for w, p in zip(weights, precisions) if p > 0)
            )
        except Exception as e:
            logger.exception("Error computing BLEU score: %s", str(e))
            return 0.0

    def _evaluate_model(self) -> float:
        """Evaluate model performance using BLEU score."""
        logger.info("Evaluating model...")
        input_tensor = torch.tensor(self.input_data.values, dtype=torch.long).to(
            self.device
        )
        target_tensor = torch.tensor(self.target_data.values, dtype=torch.long).to(
            self.device
        )
        dataloader = DataLoader(
            TensorDataset(input_tensor, target_tensor),
            batch_size=self.config.params.batch_size,
            shuffle=True,
        )

        total_bleu_score, total_samples = 0, 0
        with torch.no_grad():
            for input_batch, target_batch in dataloader:
                output_batch = self.model(
                    input_batch.to(self.device), target_batch.to(self.device)
                )
                output_batch = output_batch[:, 1:].argmax(
                    dim=-1
                )  # Remove <start> token
                target_batch = target_batch[:, 1:]  # Remove <start> token
                batch_bleu_score = sum(
                    self._compute_bleu(ref, pred)
                    for ref, pred in zip(
                        target_batch.cpu().tolist(), output_batch.cpu().tolist()
                    )
                )
                total_bleu_score += batch_bleu_score
                total_samples += len(target_batch)
        return total_bleu_score / total_samples if total_samples > 0 else 0.0

    def _log_results(self, avg_bleu_score: float) -> None:
        """Log model evaluation results to MLflow and save metrics."""
        commit_hash = self._get_git_commit_hash()
        logger.info("Commit Hash: %s", commit_hash)
        try:
            with mlflow.start_run(run_name=self.run_name):
                mlflow.set_tag("mlflow.source.git.commit", commit_hash)
                mlflow.log_params(self.config.params)
                mlflow.log_metrics({"avg_bleu_score": avg_bleu_score})
                mlflow.pytorch.log_model(self.model, "model")
                logger.info("Model logged as artifact")
            mlflow.end_run()
        except Exception as e:
            logger.error("Error during MLflow run: %s", str(e))
            raise
        Path(self.config.metrics_path).mkdir(parents=True, exist_ok=True)
        metrics_file = Path(self.config.metrics_path) / "metrics.json"
        metrics_file.write_text(
            json.dumps({"avg_bleu_score": avg_bleu_score}, indent=4)
        )
        logger.info("Evaluation complete! Avg BLEU Score: %.4f", avg_bleu_score)

    def run(self) -> None:
        """Execute the full model evaluation pipeline."""
        avg_bleu_score = self._evaluate_model()
        self._log_results(avg_bleu_score)


if __name__ == "__main__":
    try:
        model_evaluation_config = ConfigurationManager().get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.run()
    except Exception as e:
        raise RuntimeError("Model evaluation pipeline failed.") from e
