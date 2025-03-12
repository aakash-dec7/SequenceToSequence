import pandas as pd
from src.seq2seq.logger import logger
from src.seq2seq.entity.entity import DataValidationConfig
from src.seq2seq.config.configuration import ConfigurationManager


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        """Initialize DataValidation with the provided configuration."""
        self.config: DataValidationConfig = config
        self.expected_columns: set[str] = set(
            self.config.schema.get("required_columns", [])
        )

    def _load_data(self) -> pd.DataFrame | None:
        """Load the dataset from the given path."""
        try:
            logger.info(f"Loading dataset from: {self.config.data_path}")
            data = pd.read_csv(self.config.data_path)
            logger.info(
                f"Dataset loaded successfully from {self.config.data_path} with shape: {data.shape}"
            )
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {self.config.data_path}")
        except pd.errors.EmptyDataError:
            logger.error(f"The file at {self.config.data_path} is empty.")
        except pd.errors.ParserError:
            logger.error(f"Error parsing {self.config.data_path}. Invalid format.")
        except Exception as e:
            logger.error(
                f"Unexpected error while loading data from {self.config.data_path}: {repr(e)}"
            )
        return None

    def _validate_columns(self, data: pd.DataFrame) -> bool:
        """Check if the dataset contains all required columns."""
        if data is None:
            logger.error("Skipping column validation as data is not loaded.")
            return False

        missing_columns = self.expected_columns - set(data.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        logger.info("Column validation passed.")
        return True

    def run(self) -> None:
        """Run the complete validation pipeline for the dataset."""
        data = self._load_data()
        if not self._validate_columns(data):
            logger.error("Validation failed. Stopping pipeline.")
            raise RuntimeError("Data validation failed.")
        logger.info("Validation successful.")


if __name__ == "__main__":
    try:
        data_validation_config: DataValidationConfig = (
            ConfigurationManager().get_data_validation_config()
        )
        data_validation = DataValidation(config=data_validation_config)
        data_validation.run()
    except Exception as e:
        logger.exception("Data validation pipeline failed.")
        raise RuntimeError("Data validation pipeline failed.") from e
