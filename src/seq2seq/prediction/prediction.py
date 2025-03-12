import torch
from main import model, tokenizer
from src.seq2seq.logger import logger
from src.seq2seq.entity.entity import PredictionConfig


class Prediction:
    def __init__(self, config: PredictionConfig):
        """Initializes the Prediction class with configuration settings."""
        self.config: PredictionConfig = config
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model
        self.tokenizer = tokenizer

    def _preprocess_input(self, input_text: str) -> torch.Tensor:
        """Tokenizes and pads the input text for model inference."""
        try:
            input_text = input_text.strip()
            if not input_text:
                logger.warning("Received empty input text.")
                return torch.zeros((1, self.config.params.max_length), dtype=torch.long)

            tokenized_input = self.tokenizer(
                [input_text],
                padding=True,
                truncation=True,
                max_length=self.config.params.max_length,
                return_tensors="pt",
            ).input_ids

            return tokenized_input
        except Exception as e:
            logger.exception(f"Error during input preprocessing: {e}")
            return torch.zeros((1, self.config.params.max_length), dtype=torch.long)

    def _decode_sequence(
        self, encoder_outputs: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> str:
        """Decodes the output sequence from the model using greedy decoding."""
        try:
            bos_token: int = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.bos_token
            )
            eos_token: int = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.eos_token
            )
            pad_token: int = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.pad_token
            )

            if bos_token is None or eos_token is None:
                raise ValueError("Tokenizer is missing required BOS or EOS tokens.")

            x_input = torch.tensor([bos_token], dtype=torch.long)
            translated_tokens = []

            for _ in range(self.config.params.max_length):
                with torch.no_grad():
                    prediction, hidden, cell = self.model.decoder(
                        x_input, hidden, cell, encoder_outputs
                    )
                predicted_token = prediction.argmax(-1).item()

                if predicted_token in {eos_token, pad_token}:
                    break

                translated_tokens.append(predicted_token)
                x_input = torch.tensor([predicted_token], dtype=torch.long)

            return self.tokenizer.decode(translated_tokens)
        except Exception as e:
            logger.exception(f"Error during sequence decoding: {e}")
            return ""

    def predict(self, input_text: str) -> str:
        """Generates a translated output for the given input text."""
        try:
            logger.info(f"Processing input text: {input_text}")
            input_tensor = self._preprocess_input(input_text)
            with torch.no_grad():
                encoder_outputs, hidden, cell = self.model.encoder(input_tensor)
            translated_text = self._decode_sequence(encoder_outputs, hidden, cell)
            logger.info(f"Prediction completed successfully: {translated_text}")
            return translated_text
        except Exception as e:
            logger.exception(f"Prediction error: {e}")
            return ""
