import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.seq2seq.logger import logger
from src.seq2seq.entity.entity import DataTransformationConfig
from src.seq2seq.config.configuration import ConfigurationManager


class DataTransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        """Initialize DataTransformation with configuration settings."""
        self.config = config
        self.train_dir = os.path.join(self.config.root_dir, "train")
        self.test_dir = os.path.join(self.config.root_dir, "test")

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load input and target datasets."""
        try:
            logger.info("Loading datasets...")
            input_data = pd.read_csv(self.config.input_data_path)
            target_data = pd.read_csv(self.config.target_data_path)
            logger.info("Datasets loaded successfully.")
            return input_data, target_data
        except FileNotFoundError as e:
            logger.exception(f"File not found: {e.filename}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error while loading data: {e}")
            raise

    def _split_data(
        self, input_data: pd.DataFrame, target_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets."""
        try:
            logger.info("Splitting data into train and test sets...")
            train_input, test_input, train_target, test_target = train_test_split(
                input_data,
                target_data,
                test_size=self.config.params.test_size,
                random_state=self.config.params.random_state,
            )
            logger.info("Data split successfully.")
            return train_input, test_input, train_target, test_target
        except Exception as e:
            logger.exception(f"Error during train-test split: {e}")
            raise

    def _save_data(
        self,
        train_input: pd.DataFrame,
        test_input: pd.DataFrame,
        train_target: pd.DataFrame,
        test_target: pd.DataFrame,
    ) -> None:
        """Save the training and testing sets as CSV files."""
        try:
            logger.info("Saving data...")
            os.makedirs(self.train_dir, exist_ok=True)
            os.makedirs(self.test_dir, exist_ok=True)

            train_input.to_csv(
                os.path.join(self.train_dir, "input_data.csv"),
                index=False,
                header=False,
            )
            train_target.to_csv(
                os.path.join(self.train_dir, "target_data.csv"),
                index=False,
                header=False,
            )
            test_input.to_csv(
                os.path.join(self.test_dir, "input_data.csv"), index=False, header=False
            )
            test_target.to_csv(
                os.path.join(self.test_dir, "target_data.csv"),
                index=False,
                header=False,
            )

            logger.info("Data saved successfully.")
        except Exception as e:
            logger.exception(f"Error while saving data: {e}")
            raise

    def run(self) -> None:
        """Execute the full data transformation pipeline."""
        try:
            logger.info("Starting data transformation pipeline...")
            input_data, target_data = self._load_data()
            train_input, test_input, train_target, test_target = self._split_data(
                input_data, target_data
            )
            self._save_data(train_input, test_input, train_target, test_target)
            logger.info("Data transformation pipeline completed successfully.")
        except Exception as e:
            logger.exception("Data transformation pipeline failed.")
            raise RuntimeError("Data transformation pipeline failed.") from e


if __name__ == "__main__":
    try:
        data_transformation_config = (
            ConfigurationManager().get_data_transformation_config()
        )
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.run()
    except Exception as e:
        logger.exception("Fatal error in data transformation pipeline.")
        raise RuntimeError("Data transformation pipeline terminated.") from e
