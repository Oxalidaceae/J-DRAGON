import pandas as pd
import os

# Set file paths
news_data_summary_path = "data/raw/news_data_summary.csv"
processed_data_path = "data/processed_data.csv"

# Load files
if not os.path.exists(news_data_summary_path):
    raise FileNotFoundError(f"{news_data_summary_path} not found.")
if not os.path.exists(processed_data_path):
    raise FileNotFoundError(f"{processed_data_path} not found.")

news_data_summary = pd.read_csv(news_data_summary_path, dtype={'Date': str})
processed_data = pd.read_csv(processed_data_path, dtype={'Date': str})

# Synchronize date formats
news_data_summary['Date'] = news_data_summary['Date'].str.replace('-', '')
processed_data['Date'] = processed_data['Date'].str.replace('-', '')

# Filter data based on common dates
filtered_data = news_data_summary[news_data_summary['Date'].isin(processed_data['Date'])]

# Skip processing if there are no matching dates
if filtered_data.empty:
    print("No matching dates found. No data merged.")
else:
    # Merge filtered data into processed_data and handle duplicate column names
    merged_data = processed_data.merge(filtered_data, on='Date', how='left', suffixes=('', '_news'))

    # Fill missing values using the rolling mean of the surrounding 5 days
    merged_data.set_index('Date', inplace=True)
    for column in merged_data.columns:
        merged_data[column] = merged_data[column].fillna(
            merged_data[column].rolling(window=11, min_periods=1, center=True).mean()
        )
        # Replace NaN values with 0 and convert to integers
        merged_data[column] = merged_data[column].fillna(0).round().astype('Int64')
    merged_data.reset_index(inplace=True)  # Reset index to 'Date'

    # Maintain the original column order and move newly added columns to the right
    for col in filtered_data.columns:
        if col != 'Date' and col in merged_data.columns:
            merged_data = merged_data[[c for c in merged_data.columns if c != col] + [col]]

    # Save the merged data to a CSV file
    merged_data.to_csv(processed_data_path, index=False, encoding='utf-8-sig')

    print(f"Processed data saved to {processed_data_path}.")