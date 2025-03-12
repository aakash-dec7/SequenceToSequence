from src.seq2seq.constant import *
from src.seq2seq.utils.utils import read_yaml, create_directories
from src.seq2seq.entity.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataPreprocessingConfig,
    DataTransformationConfig,
    ModelConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
    PredictionConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_path=CONFIG_FILE_PATH,
        params_path=PARAMS_FILE_PATH,
        schema_path=SCHEMA_FILE_PATH,
    ):
        """
        Initializes the ConfigManager by reading configuration, parameters, and schema files.
        Creates necessary directories for storing artifacts.
        """

        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.schema = read_yaml(schema_path)

        create_directories(self.config.artifacts_root)

    ### Data Ingestion
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        ingestion_config = self.config.data_ingestion

        create_directories(ingestion_config.root_dir)

        return DataIngestionConfig(
            root_dir=ingestion_config.root_dir,
            source_url=ingestion_config.source.url,
            download_path=ingestion_config.download_path,
        )

    ### Data Validation
    def get_data_validation_config(self) -> DataValidationConfig:
        validation_config = self.config.data_validation
        validation_schema = self.schema.columns

        return DataValidationConfig(
            data_path=validation_config.data_path,
            schema=validation_schema,
        )

    ### Data Preprocessing
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        preprocessing_config = self.config.data_preprocessing
        preprocessing_params = self.params.hyperparameters
        preprocessing_schema = self.schema.columns

        create_directories(preprocessing_config.root_dir)

        return DataPreprocessingConfig(
            root_dir=preprocessing_config.root_dir,
            data_path=preprocessing_config.data_path,
            tokenizer_path=preprocessing_config.tokenizer_path,
            params=preprocessing_params,
            schema=preprocessing_schema,
        )

    ### Data Transformation
    def get_data_transformation_config(self) -> DataTransformationConfig:
        transformation_config = self.config.data_transformation
        transformation_params = self.params.train_test_split

        create_directories(transformation_config.root_dir)

        return DataTransformationConfig(
            root_dir=transformation_config.root_dir,
            input_data_path=transformation_config.input_data_path,
            target_data_path=transformation_config.target_data_path,
            params=transformation_params,
        )

    ### Model
    def get_model_config(self) -> ModelConfig:
        model_params = self.params.hyperparameters

        return ModelConfig(model_params=model_params)

    ### Model Training
    def get_model_training_config(self) -> ModelTrainingConfig:
        training_config = self.config.model_training
        training_params = self.params.hyperparameters

        create_directories(training_config.root_dir)

        return ModelTrainingConfig(
            root_dir=training_config.root_dir,
            train_input_path=training_config.train_input_path,
            train_target_path=training_config.train_target_path,
            params=training_params,
        )

    ### Model Evaluation
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        evaluation_config = self.config.model_evaluation
        evaluation_params = self.params.hyperparameters
        exp_tracking_config = self.config.experiment_tracking

        create_directories(evaluation_config.root_dir)

        return ModelEvaluationConfig(
            root_dir=evaluation_config.root_dir,
            test_input_path=evaluation_config.test_input_path,
            test_target_path=evaluation_config.test_target_path,
            model_path=evaluation_config.model_path,
            metrics_path=evaluation_config.metrics_path,
            params=evaluation_params,
            repo_name=exp_tracking_config.repo_name,
            repo_owner=exp_tracking_config.repo_owner,
            mlflow_uri=exp_tracking_config.mlflow.uri,
        )

    ### Prediction
    def get_prediction_config(self) -> PredictionConfig:
        prediction_config = self.config.prediction
        prediction_params = self.params.hyperparameters

        return PredictionConfig(
            model_path=prediction_config.model_path,
            tokenizer_path=prediction_config.tokenizer_path,
            params=prediction_params,
        )
