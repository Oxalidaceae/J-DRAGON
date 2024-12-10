import os
from datetime import datetime
import pandas as pd
import FinanceDataReader as fdr
import numpy as np

def load_and_save_raw_data(stock_code: str, currency_code: str, start_date: str, end_date: str):
    os.makedirs('./data/raw', exist_ok=True)

    print("Loading Samsung stock data...")
    stock_data = fdr.DataReader(stock_code, start_date, end_date)
    stock_path = './data/raw/samsung_stock.csv'
    stock_data.to_csv(stock_path, encoding='utf-8', index=True)
    print(f"Samsung stock data saved to {stock_path}")

    print("Loading USD/KRW exchange rate data...")
    exchange_data = fdr.DataReader(currency_code, start_date, end_date)
    exchange_path = './data/raw/usd_krw_exchange.csv'
    exchange_data.to_csv(exchange_path, encoding='utf-8', index=True)
    print(f"USD/KRW exchange rate data saved to {exchange_path}")

    return stock_data, exchange_data

def fill_missing_knn(data: pd.DataFrame, column: str):
    filled_data = data.copy()
    values = filled_data[column].values

    for i in range(len(values)):
        if np.isnan(values[i]):  # Check for missing value
            start = i
            while i < len(values) and np.isnan(values[i]):
                i += 1
            end = i

            if end - start == 1:  # single missing value
                left = values[start - 1] if start > 0 else np.nan
                right = values[end] if end < len(values) else np.nan
                values[start] = np.nanmean([left, right])
            else:  # multiple consecutive missing values
                left = values[start - 1] if start > 0 else np.nan
                right = values[end] if end < len(values) else np.nan
                fill_value = np.nanmean([left, right])
                values[start:end] = fill_value

    filled_data[column] = values
    return filled_data

def preprocess_and_save_data_with_volume(stock_data: pd.DataFrame, exchange_data: pd.DataFrame):
    stock_data.reset_index(inplace=True)
    exchange_data.reset_index(inplace=True)

    # Rename stock columns
    # Raw stock_data column: ['Date','Open','High','Low','Close','Volume']  (Date is created after reset_index)
    stock_data.rename(columns={
        'Open': 'Stock_Open',
        'High': 'Stock_High',
        'Low': 'Stock_Low',
        'Close': 'Stock_Close',
        'Volume': 'Stock_Volume'
    }, inplace=True)

    # Rename exchange column
    exchange_data.rename(columns={'Close': 'Exchange_Close'}, inplace=True)

    # Rename the first column to 'Date' if needed
    stock_data.rename(columns={stock_data.columns[0]: 'Date'}, inplace=True)
    exchange_data.rename(columns={exchange_data.columns[0]: 'Date'}, inplace=True)

    # Fill missing values for Volume
    stock_data = fill_missing_knn(stock_data, 'Stock_Volume')

    # Merge datasets
    print("Merging stock and exchange rate data...")
    processed_data = pd.merge(
        stock_data[['Date', 'Stock_Open', 'Stock_High', 'Stock_Low', 'Stock_Close', 'Stock_Volume']],
        exchange_data[['Date', 'Exchange_Close']],
        on='Date',
        how='inner'
    )

    processed_path = './data/processed_data.csv'
    os.makedirs('./data', exist_ok=True)
    processed_data.to_csv(processed_path, index=False, encoding='utf-8')
    print(f"Processed data (with OHLC, Volume) saved to {processed_path}")

    return processed_data

def main_with_volume():
    STOCK_CODE = '005930'
    CURRENCY_CODE = 'USD/KRW'
    START_DATE = '2020-01-01'

    today = datetime.today()
    today_str = today.strftime('%Y-%m-%d')
    END_DATE = today_str

    stock_data, exchange_data = load_and_save_raw_data(STOCK_CODE, CURRENCY_CODE, START_DATE, END_DATE)
    preprocess_and_save_data_with_volume(stock_data, exchange_data)

if __name__ == "__main__":
    main_with_volume()