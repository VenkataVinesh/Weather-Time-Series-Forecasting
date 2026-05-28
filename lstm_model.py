import torch
import torch.nn as nn

class WeatherForecastingLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, output_dim=1, dropout=0.2):
        super(WeatherForecastingLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stacked LSTM layers with batch_first=True
        # input shape: (batch_size, sequence_length, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Linear projection layer mapping final step state to predictions
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass through recurrent cells
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Retrieve the output of the final time step
        # out shape: (batch_size, sequence_length, hidden_dim)
        last_step_out = out[:, -1, :]
        
        # Project output to prediction target dimensions
        predictions = self.fc(last_step_out)
        return predictions
