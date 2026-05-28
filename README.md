# Weather Time-Series Forecasting Model

A deep learning project in PyTorch building stacked Long Short-Term Memory (LSTM) sequence predictors to forecast meteorological metrics (temperature, humidity, atmospheric pressure) based on historical sensor feeds.

## Project Structure
```
Weather-Time-Series-Forecasting/
├── lstm_model.py       # PyTorch LSTM network definitions
├── data_loader.py      # Rolling window data formatting helpers
├── train.py            # Synthetic dataset generators and training loops
├── requirements.txt    # PyTorch and computing dependencies
└── README.md           # This setup guide
```

## Setup & Running

1. Install PyTorch and other dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model (generates synthetic meteorological history, runs a 20-epoch training loop, and plots training vs validation curves):
   ```bash
   python train.py
   ```
3. The training script will export `loss_curve.png` and `forecast_validation.png` showing the model convergence.

## Neural Network Architecture
- **Recurrent Layer:** Stacked 2-layer LSTM with 64 hidden nodes and 0.2 dropout.
- **Sequence Mapping:** rolling context window of 24 hours to forecast the subsequent 1 hour.
- **Optimization:** Mean Absolute Error (MAE) loss minimized using Adam optimizer.
