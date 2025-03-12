import torch
import torch.nn as nn
from src.seq2seq.config.configuration import ConfigurationManager


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, (hidden, cell) = self.lstm(embedded)
        return lstm_output, hidden, cell


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w_Q = nn.Linear(hidden_dim, hidden_dim)
        self.w_K = nn.Linear(hidden_dim, hidden_dim)
        self.w_V = nn.Linear(hidden_dim, 1)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden[-1].unsqueeze(1)
        attention_scores = self.w_V(
            torch.tanh(self.w_Q(hidden) + self.w_K(encoder_outputs))
        )
        attention_weights = torch.softmax(attention_scores, dim=1).transpose(1, 2)
        return torch.bmm(attention_weights, encoder_outputs)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.attention = BahdanauAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = self.embedding(x.unsqueeze(1))
        lstm_output, (hidden, cell) = self.lstm(x, (hidden, cell))
        context_vector = self.attention(hidden, encoder_outputs)
        prediction = self.fc(torch.cat((lstm_output, context_vector), dim=2))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)

    def forward(self, input_seq, target_seq):
        batch_size, max_length = target_seq.size()
        outputs = torch.zeros(batch_size, max_length, self.decoder.fc.out_features)

        encoder_outputs, hidden, cell = self.encoder(input_seq)
        target_input_token = target_seq[:, 0]

        for t in range(1, max_length):
            decoder_output, hidden, cell = self.decoder(
                target_input_token, hidden, cell, encoder_outputs
            )
            outputs[:, t, :] = decoder_output.squeeze(1)
            target_input_token = target_seq[:, t]

        return outputs


class Model(Seq2Seq):
    def __init__(self, config):
        super().__init__(
            vocab_size=config.model_params.vocab_size,
            embed_dim=config.model_params.embedding_dim,
            hidden_dim=config.model_params.hidden_dim,
            num_layers=config.model_params.num_layers,
            dropout=config.model_params.dropout,
        )


if __name__ == "__main__":
    try:
        config = ConfigurationManager().get_model_config()
        model = Model(config)
        print("Model initialized successfully.")
    except Exception as e:
        raise RuntimeError("Model initialization failed.") from e
