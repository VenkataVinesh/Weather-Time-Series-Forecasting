import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lstm_model import WeatherForecastingLSTM
from data_loader import prepare_loaders

def generate_synthetic_weather_data(num_hours=720):
    np.random.seed(42)
    time = np.arange(num_hours)
    
    # Temperature cycle: diurnal (24h period) + seasonal trend + noise
    temp = 20.0 + 8.0 * np.sin(2 * np.pi * time / 24.0) + np.random.normal(0.0, 1.0, num_hours)
    # Humidity: correlated negatively with temperature
    humidity = 60.0 - 15.0 * np.sin(2 * np.pi * time / 24.0) + np.random.normal(0.0, 3.0, num_hours)
    humidity = np.clip(humidity, 10.0, 100.0)
    # Pressure: random walk with noise
    pressure = 1013.25 + np.cumsum(np.random.normal(0.0, 0.1, num_hours))
    
    features = np.column_stack((temp, humidity, pressure))
    
    # Target is predicting future temperature (next hour)
    target = temp.reshape(-1, 1)
    
    return features, target

def train_model():
    # Setup hyperparameters
    seq_length = 24
    input_dim = 3
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    batch_size = 32
    epochs = 10
    learning_rate = 0.005
    
    # Load synthetic weather features
    print("Generating synthetic meteorological sensor dataset...")
    features, target = generate_synthetic_weather_data()
    
    # Prepare loaders
    train_loader, val_loader = prepare_loaders(features, target, seq_length, batch_size)
    
    # Instantiate LSTM model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeatherForecastingLSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)
    
    criterion = nn.L1Loss() # Mean Absolute Error
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Beginning training on device: {device}...")
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train MAE: {epoch_loss:.4f} | Val MAE: {val_loss:.4f}")
        
    # Plotting training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss (MAE)')
    plt.plot(val_losses, label='Val Loss (MAE)')
    plt.title('Training & Validation MAE curves - LSTM Weather Forecast')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("Exported training loss curves to loss_curve.png")
    
if __name__ == "__main__":
    train_model()
