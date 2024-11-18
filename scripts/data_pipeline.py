import os
from datetime import datetime
import pandas as pd
import FinanceDataReader as fdr

def load_and_save_raw_data(stock_code: str, currency_code: str, start_date: str, end_date: str):
    
    """
    Load raw data for stock and exchange rates and save them in the raw/ directory.

    Args:
        stock_code (str): Stock code for the company (e.g., '005930' for Samsung Electronics).
        currency_code (str): Currency code for exchange rates (e.g., 'USD/KRW').
        start_date (str): Start date for the data (YYYY-MM-DD).
        end_date (str): End date for the data (YYYY-MM-DD).
    """
    # Some missing values will not be business days(korea or US)
    
    os.makedirs('./data/raw', exist_ok=True)

    # Load Samsung stock data
    print("Loading Samsung stock data...")
    stock_data = fdr.DataReader(stock_code, start_date, end_date)
    stock_path = './data/raw/samsung_stock.csv'
    stock_data.to_csv(stock_path, encoding='utf-8', index=True)
    print(f"Samsung stock data saved to {stock_path}")

    # Load USD/KRW exchange rate data
    print("Loading USD/KRW exchange rate data...")
    exchange_data = fdr.DataReader(currency_code, start_date, end_date)
    exchange_path = './data/raw/usd_krw_exchange.csv'
    exchange_data.to_csv(exchange_path, encoding='utf-8', index=True)
    print(f"USD/KRW exchange rate data saved to {exchange_path}")

    return stock_data, exchange_data

def preprocess_and_save_data(stock_data: pd.DataFrame, exchange_data: pd.DataFrame):
    """
    Preprocess and merge stock and exchange rate data, then save the processed data.

    Args:
        stock_data (pd.DataFrame): Raw stock data.
        exchange_data (pd.DataFrame): Raw exchange rate data.

    Returns:
        pd.DataFrame: Merged and processed data.
    """
    # Reset index for merging
    stock_data.reset_index(inplace=True)
    exchange_data.reset_index(inplace=True)

    # Rename columns for clarity
    stock_data.rename(columns={'Close': 'Stock_Close'}, inplace=True)
    exchange_data.rename(columns={'Close': 'Exchange_Close'}, inplace=True)
    
    # Rename the first column to 'Date' due to some failures during loading process
    stock_data.rename(columns={stock_data.columns[0]: 'Date'}, inplace=True)
    exchange_data.rename(columns={exchange_data.columns[0]: 'Date'}, inplace=True)

    # Merge datasets on the Date column / Consider close price only - inner join
    print("Merging stock and exchange rate data...")
    processed_data = pd.merge(
        stock_data[['Date', 'Stock_Close']],
        exchange_data[['Date', 'Exchange_Close']],
        on='Date',
        how='inner'
    )

    # Save processed data
    processed_path = './data/processed_data.csv'
    os.makedirs('./data', exist_ok=True)
    processed_data.to_csv(processed_path, index=False, encoding='utf-8')
    print(f"Processed data saved to {processed_path}")

    return processed_data

def main():
    # Configuration
    STOCK_CODE = '005930'  # Samsung Electronics stock code
    CURRENCY_CODE = 'USD/KRW'  # USD to KRW exchange rate
    START_DATE = '2020-01-01'
    
    today = datetime.today()
    today_str = today.strftime('%Y-%m-%d')
    END_DATE = today_str

    # Load and save raw data
    stock_data, exchange_data = load_and_save_raw_data(STOCK_CODE, CURRENCY_CODE, START_DATE, END_DATE)

    # Preprocess and save processed data
    preprocess_and_save_data(stock_data, exchange_data)

if __name__ == "__main__":
    main()