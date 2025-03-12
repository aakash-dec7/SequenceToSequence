import os
import pandas as pd
from transformers import MarianTokenizer
from src.seq2seq.logger import logger
from src.seq2seq.utils.utils import update_yaml_file
from src.seq2seq.entity.entity import DataPreprocessingConfig
from src.seq2seq.config.configuration import ConfigurationManager


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        """Initialize tokenizer and configuration."""
        self.config: DataPreprocessingConfig = config
        self.tokenizer: MarianTokenizer = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-fr"
        )
        self.tokenizer.add_special_tokens({"bos_token": "[BOS]"})
        self.data: pd.DataFrame | None = None

    def _load_data(self) -> None:
        """Load dataset from the specified path."""
        logger.info(f"Loading dataset from: {self.config.data_path}")
        try:
            self.data = pd.read_csv(self.config.data_path)
        except FileNotFoundError:
            logger.exception(f"File not found: {self.config.data_path}")
            raise
        except Exception as e:
            logger.exception(f"Error loading dataset: {str(e)}")
            raise

    def _remove_missing_values(self) -> None:
        """Remove rows with missing values in required columns."""
        if self.data is None:
            logger.error("Data not loaded. Skipping missing value removal.")
            return

        required_columns = set(self.config.schema.get("required_columns", []))
        initial_shape = self.data.shape
        self.data.dropna(subset=required_columns, inplace=True)
        logger.info(
            f"Removed missing values. Shape changed from {initial_shape} to {self.data.shape}."
        )

    def _tokenize_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Tokenize input and target text data."""
        logger.info("Tokenizing text data.")
        try:
            input_data = self.data[self.config.schema.input_column]
            target_data = self.data[self.config.schema.target_column].apply(
                lambda x: f"{self.tokenizer.bos_token} {x} {self.tokenizer.eos_token}"
            )

            input_sequences = self.tokenizer(
                input_data.tolist(),
                padding=True,
                truncation=True,
                max_length=self.config.params.max_length,
                return_tensors="pt",
            ).input_ids

            target_sequences = self.tokenizer(
                target_data.tolist(),
                padding=True,
                truncation=True,
                max_length=self.config.params.max_length,
                return_tensors="pt",
            ).input_ids

            return input_sequences, target_sequences
        except Exception as e:
            logger.exception(f"Error during tokenization: {str(e)}")
            raise

    def _save_tokenizer(self) -> None:
        """Save tokenizer to disk for later use."""
        logger.info("Saving tokenizer.")
        try:
            os.makedirs(self.config.tokenizer_path, exist_ok=True)
            tokenizer_path = os.path.join(self.config.tokenizer_path, "tokenizer")
            self.tokenizer.save_pretrained(tokenizer_path)
            logger.info(f"Tokenizer saved at: {tokenizer_path}.")
        except Exception as e:
            logger.exception(f"Error saving tokenizer: {str(e)}")
            raise

    def _update_vocab_size(self) -> None:
        """Update vocabulary size in the configuration file."""
        logger.info("Updating vocabulary size.")
        try:
            vocab_size = len(self.tokenizer)
            update_yaml_file(
                "hyperparameters", "vocab_size", vocab_size, "config/params.yaml"
            )
            logger.info("Vocabulary size updated in params.yaml.")
        except Exception as e:
            logger.exception(f"Error updating vocabulary size: {str(e)}")
            raise

    def _save_preprocessed_data(
        self, input_sequences: pd.DataFrame, target_sequences: pd.DataFrame
    ) -> None:
        """Save preprocessed sequences as CSV files."""
        logger.info("Saving preprocessed data.")
        try:
            input_path = os.path.join(self.config.root_dir, "input_data.csv")
            target_path = os.path.join(self.config.root_dir, "target_data.csv")

            pd.DataFrame(input_sequences).to_csv(input_path, index=False, header=False)
            pd.DataFrame(target_sequences).to_csv(
                target_path, index=False, header=False
            )
            logger.info(f"Preprocessed data saved at: {self.config.root_dir}.")
        except Exception as e:
            logger.exception(f"Error saving preprocessed data: {str(e)}")
            raise

    def run(self) -> None:
        """Execute the data preprocessing pipeline."""
        self._load_data()
        self._remove_missing_values()
        input_sequences, target_sequences = self._tokenize_data()
        self._save_tokenizer()
        self._update_vocab_size()
        self._save_preprocessed_data(input_sequences, target_sequences)
        logger.info("Data preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    try:
        data_preprocessing_config = (
            ConfigurationManager().get_data_preprocessing_config()
        )
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.run()
    except Exception as e:
        logger.exception("Data preprocessing pipeline failed.")
        raise RuntimeError("Data preprocessing pipeline failed.") from e
