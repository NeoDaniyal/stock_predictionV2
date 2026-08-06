import tourch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_dim: int,
     hidden_dim1: int =128, 
     hidden_dim_2: int =64,
     dense_dim: int =64,
     num_classes: int = 3,
     dropout: float = 0.3):
        super(StockLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_dim,
                             hidden_size=hidden_dim1,
                             batch_first=True,
                             bidirectional=False)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim_2,
            batch_first=True,
            bidirectional=False
        )
        self.dropout2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_dim_2, dense_dim)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.out = nn.Linear(dense_dim, num_classes)

    def forward(self, x):

        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm(out)

        last_step_out = out[:, -1, :]
        last_step_out = self.dropout2(last_step_out)

        dense_out = self.fc1(last_step_out)
        dense_out = self.dropout3(dense_out)

        logits = self.out(dense_out)
        return logits