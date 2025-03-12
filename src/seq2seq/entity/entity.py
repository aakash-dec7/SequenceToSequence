from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    download_path: Path


@dataclass
class DataValidationConfig:
    data_path: Path
    schema: dict


@dataclass
class DataPreprocessingConfig:
    root_dir: Path
    data_path: Path
    tokenizer_path: Path
    params: dict
    schema: dict


@dataclass
class DataTransformationConfig:
    root_dir: Path
    input_data_path: Path
    target_data_path: Path
    params: dict


@dataclass
class ModelConfig:
    model_params: dict


@dataclass
class ModelTrainingConfig:
    root_dir: Path
    train_input_path: Path
    train_target_path: Path
    params: dict


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_input_path: Path
    test_target_path: Path
    model_path: Path
    metrics_path: Path
    params: dict
    repo_name: str
    repo_owner: str
    mlflow_uri: str


@dataclass
class PredictionConfig:
    model_path: Path
    tokenizer_path: Path
    params: dict
