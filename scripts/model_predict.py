import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import mplfinance as mpf
import os

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

def load_data(data_path, lookback=20):
    
    #Load and scale data.
    #Features: Stock_Open, Stock_High, Stock_Low, Stock_Close, Exchange_Close, Stock_Volume, title_interest
    data = pd.read_csv(data_path)
    features = ['Stock_Open', 'Stock_High', 'Stock_Low', 'Stock_Close', 'Exchange_Close', 'Stock_Volume', 'title_interest']
    targets = ['Stock_Open', 'Stock_High', 'Stock_Low', 'Stock_Close']

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Scale features and targets
    scaled_features = feature_scaler.fit_transform(data[features])
    target_scaler.fit_transform(data[targets])  # Fit targets scaler only

    # Prepare input for the model
    X_input = scaled_features[-lookback:, :]
    X_input = np.expand_dims(X_input, axis=0)  # Add batch dimension

    return X_input, data, feature_scaler, target_scaler

def predict_future_with_teacher_forcing(model, data, X_input, scaler, target_scaler, horizon=30, lookback=30):
    
    #Predict future values using Teacher Forcing when data is available.
    model.eval()
    X_input = torch.tensor(X_input, dtype=torch.float32)
    predictions = []

    for day in range(horizon):
        with torch.no_grad():
            # Step 1: Predict scaled values for Open, High, Low, Close
            pred_scaled = model(X_input).squeeze().numpy()

        # Step 2: Convert scaled predictions to original values
        pred_original = target_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]

        # Step 3: Save predictions
        predictions.append(pred_original)

        # Step 4: Update X_input for the next prediction
        next_day_index = lookback + day
        if next_day_index < len(data):
            # Teacher Forcing: Use actual data when available
            actual_next = data.iloc[next_day_index]
            actual_next_scaled = scaler.transform([[
                actual_next['Stock_Open'],
                actual_next['Stock_High'],
                actual_next['Stock_Low'],
                actual_next['Stock_Close'],
                actual_next['Exchange_Close'],
                actual_next['Stock_Volume'],
                actual_next['title_interest']
            ]])
            new_row = actual_next_scaled
        else:
            # Use predicted values when actual data is unavailable
            new_scaled_row = X_input[0, -1, :].detach().numpy().copy()
            new_scaled_row[0:4] = pred_scaled  # Replace Open, High, Low, Close
            new_row = new_scaled_row.reshape(1, -1)

        # Update X_input
        new_input = np.concatenate([X_input[0, 1:, :].detach().numpy(), new_row], axis=0)
        X_input = torch.tensor(new_input, dtype=torch.float32).unsqueeze(0)

    return predictions

def plot_candle_chart(predictions, horizon=30):
    # Generate future dates for predictions
    future_dates = pd.date_range(start=pd.Timestamp.today(), periods=horizon, freq='D')
    pred_df = pd.DataFrame(predictions, columns=['Open', 'High', 'Low', 'Close'], index=future_dates)

    # Save predictions as CSV
    os.makedirs("./results", exist_ok=True)
    pred_df.to_csv(f"./results/prediction_{horizon}_days.csv")

    # Plot candle chart
    mpf.plot(pred_df, type='candle', style='charles', title=f'{horizon}-Day Prediction Candle Chart', mav=(3,6), volume=False)
    print(f"{horizon}-day prediction saved to ./results/prediction_{horizon}_days.csv")

if __name__ == "__main__":
    MODEL_PATH = "./models/LSTM_prediction_model.pth"
    DATA_PATH = "./data/processed_data.csv"
    LOOKBACK = 30

    # Load trained model
    # Features: Open, High, Low, Close, Exchange_Close, Volume, title_interest
    input_size = 7
    model = LSTMModel(input_size=input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load data and prepare initial input
    X_input, data, scaler, target_scaler = load_data(DATA_PATH, lookback=LOOKBACK)

    # Predict for different horizons
    horizons = [7, 30, 365]
    for horizon in horizons:
        predictions = predict_future_with_teacher_forcing(model, data, X_input, scaler, target_scaler, horizon=horizon, lookback=LOOKBACK)
        plot_candle_chart(predictions, horizon=horizon)