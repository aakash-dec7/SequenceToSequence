import urllib.error
from pathlib import Path
import urllib.request as request
from src.seq2seq.logger import logger
from src.seq2seq.entity.entity import DataIngestionConfig
from src.seq2seq.config.configuration import ConfigurationManager


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Initialize DataIngestion with the provided configuration.
        """
        self.config: DataIngestionConfig = config
        self.download_path: Path = Path(self.config.download_path)

    def _download_file(self) -> None:
        """
        Download the file if it does not already exist.
        """
        if self.download_path.exists():
            logger.info(f"File already exists: {self.download_path}")
            return

        try:
            logger.info(
                f"Downloading from {self.config.source_url} to {self.download_path}..."
            )
            self.download_path.parent.mkdir(parents=True, exist_ok=True)
            request.urlretrieve(self.config.source_url, str(self.download_path))
            logger.info(f"Download successful: {self.download_path}")
        except urllib.error.HTTPError as http_err:
            logger.error(f"HTTP error {http_err.code} while downloading: {http_err}")
            raise RuntimeError("HTTP error occurred during file download") from http_err
        except urllib.error.URLError as url_err:
            logger.error(
                f"URL error while accessing {self.config.source_url}: {url_err}"
            )
            raise RuntimeError("URL error occurred during file download") from url_err
        except Exception as e:
            logger.exception("Unexpected error during file download.")
            raise RuntimeError("Unexpected error occurred during file download") from e

    def run(self) -> None:
        """
        Run the data ingestion pipeline.
        """
        self._download_file()


if __name__ == "__main__":
    try:
        data_ingestion_config: DataIngestionConfig = (
            ConfigurationManager().get_data_ingestion_config()
        )
        data_ingestion: DataIngestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.run()
    except Exception as e:
        logger.exception("Data ingestion pipeline failed.")
        raise RuntimeError("Data ingestion pipeline failed.") from e
