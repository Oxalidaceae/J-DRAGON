import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# Load and merge data
def load_and_merge_data():
    #Load processed data and encode title interest.
    processed_data = pd.read_csv('./data/processed_data.csv')
    processed_data = encode_title(processed_data)
    processed_data.to_csv('./data/processed_data.csv', index=False)
    return processed_data

def encode_title(data):
    #Encode title interest as binary feature.
    data['title_interest'] = (data['title'] >= 3).astype(int)
    return data

# Preprocess data
def preprocess_data(data, lookback):
    
    # Prepare data for LSTM training.
    # Apply feature scaling and set weights for 'title_interest'.
    features = ['Stock_Open', 'Stock_High', 'Stock_Low', 'Stock_Close', 
                'Exchange_Close', 'Stock_Volume', 'title_interest']
    targets = ['Stock_Open', 'Stock_High', 'Stock_Low', 'Stock_Close']

    # Scale features and targets
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(data[features])
    scaled_targets = target_scaler.fit_transform(data[targets])

    # Apply weights to features
    feature_weights = np.ones(len(features))
    feature_weights[-1] = 0.2  # Lower weight for 'title_interest'
    scaled_features *= feature_weights

    # Create LSTM sequences
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(scaled_features[i-lookback:i, :])
        y.append(scaled_targets[i, :])  # Targets: [Open, High, Low, Close]

    return np.array(X), np.array(y), feature_scaler, target_scaler

# 3. Define PyTorch Dataset
class StockDataset(torch.utils.data.Dataset):
    #Custom dataset for stock data.
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 4. Build the model
class LSTMModel(nn.Module):
    #LSTM model for stock prediction.
    def __init__(self, input_size, hidden_size=64, output_size=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# 5. Train the model
def train_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    #Train LSTM model with MSE loss and Adam optimizer.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                predictions = model(X_val)
                loss = criterion(predictions, y_val)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Save the model
    torch.save(model.state_dict(), './models/LSTM_prediction_model.pth')
    print("Model saved successfully!")

def main():
    #Main function to load data, preprocess, and train the model.
    lookback = 30
    combined_data = load_and_merge_data()
    X, y, feature_scaler, target_scaler = preprocess_data(combined_data, lookback=lookback)

    # Prepare dataset and split into training and validation sets
    dataset = StockDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize and train the model
    model = LSTMModel(input_size=X.shape[2])
    train_model(model, train_loader, val_loader, epochs=30)

if __name__ == "__main__":
    main()