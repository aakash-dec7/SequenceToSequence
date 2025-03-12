import os
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from src.seq2seq.logger import logger
from src.seq2seq.component.model import Model
from src.seq2seq.entity.entity import ModelTrainingConfig
from src.seq2seq.config.configuration import ConfigurationManager
from src.seq2seq.utils.utils import load_checkpoint, save_checkpoint


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig) -> None:
        """Initialize model training with configuration parameters."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model()
        self.optimizer = self._initialize_optimizer()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.start_epoch = self._load_checkpoint()
        self.dataloader = self._load_data()

    def _initialize_model(self) -> Model:
        """Initialize the model and move it to the appropriate device."""
        model = Model(config=ConfigurationManager().get_model_config()).to(self.device)
        return model

    def _initialize_optimizer(self) -> torch.optim.Adam:
        """Initialize the optimizer."""
        return torch.optim.Adam(
            self.model.parameters(), lr=self.config.params.learning_rate
        )

    def _load_checkpoint(self) -> int:
        """Load training checkpoint if available."""
        try:
            start_epoch = load_checkpoint(
                self.model, self.optimizer, self.config.params.num_epochs
            )
            logger.info(f"Resuming training from epoch {start_epoch}")
            return start_epoch
        except Exception as e:
            logger.exception(f"Error loading checkpoint: {e}")
            return 1

    def _load_data(self) -> DataLoader:
        """Load training data and return a DataLoader instance."""
        try:
            input_data = torch.tensor(
                pd.read_csv(self.config.train_input_path).values, dtype=torch.long
            ).to(self.device)
            target_data = torch.tensor(
                pd.read_csv(self.config.train_target_path).values, dtype=torch.long
            ).to(self.device)
            dataloader = DataLoader(
                TensorDataset(input_data, target_data),
                batch_size=self.config.params.batch_size,
                shuffle=True,
            )
            logger.info("Training data loaded successfully.")
            return dataloader
        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            raise

    def _train(self) -> None:
        """Train the model for the specified number of epochs."""
        self.model.train()
        for epoch in range(self.start_epoch, self.config.params.num_epochs + 1):
            epoch_loss = 0
            progress_bar = tqdm(
                self.dataloader, desc=f"Epoch {epoch}/{self.config.params.num_epochs}"
            )

            for input_batch, target_batch in progress_bar:
                input_batch, target_batch = input_batch.to(
                    self.device
                ), target_batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input_batch, target_batch)
                output = output[:, 1:].reshape(-1, output.shape[2])
                target_batch = target_batch[:, 1:].reshape(-1)
                loss = self.criterion(output, target_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.params.clip
                )
                self.optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            logger.info(f"Epoch {epoch} completed with loss: {epoch_loss:.4f}")
            save_checkpoint(epoch, self.model, self.optimizer)

    def _save_model(self) -> None:
        """Save the trained model to disk."""
        model_path = os.path.join(self.config.root_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved successfully at: {model_path}")

    def run(self) -> None:
        """Execute the training pipeline."""
        self._train()
        self._save_model()


if __name__ == "__main__":
    try:
        config = ConfigurationManager().get_model_training_config()
        trainer = ModelTraining(config=config)
        trainer.run()
    except Exception as e:
        logger.exception("Model training pipeline failed")
        raise
