import torch
import torch.nn as nn

class MandiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 3)  # Predicts 30d, 60d, 90d

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        out, (hn, cn) = self.lstm(x)
        # Take the output of the last time step
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        preds = self.fc(last_out)
        return preds

class MandiBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 3)

    def forward(self, x):
        out, (hn, cn) = self.bilstm(x)
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        preds = self.fc(last_out)
        return preds
