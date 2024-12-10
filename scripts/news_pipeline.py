import os
import urllib.request
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def getresult(client_id, client_secret, query, display=10, start=1, sort='sim'):
    encText = urllib.parse.quote(query)
    url = "https://openapi.naver.com/v1/search/news?query=" + encText + \
          "&display=" + str(display) + "&start=" + str(start) + "&sort=" + sort

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if rescode == 200:
        response_body = response.read()
        response_json = json.loads(response_body)
    else:
        print(f"Error Code: {rescode}")
        return pd.DataFrame()

    return pd.DataFrame(response_json['items'])

# Naver API certificate information
client_id = "example"  # Your Client ID
client_secret = "example"  # Your Client Secret

# Query for news articles
query = '삼성전자 주가 주식'
display = 100  # max articles per query
sort = 'date'

start_date = datetime(2020, 1, 1)
end_date = datetime.now()
date_range = pd.date_range(start=start_date, end=end_date)

result_all = pd.DataFrame()

for target_date in date_range:
    target_date_str = target_date.strftime('%Y-%m-%d')
    query_with_date = f"{query} {target_date_str}"
    
    try:
        result = getresult(client_id, client_secret, query_with_date, display=display, sort=sort)
        if not result.empty:
            result['pubDate'] = pd.to_datetime(result['pubDate'], format='%a, %d %b %Y %H:%M:%S %z', errors='coerce')
            result = result.dropna(subset=['pubDate'])
            result = result[result['pubDate'].dt.date == target_date.date()]
            result_all = pd.concat([result_all, result], ignore_index=True)
    except Exception as e:
        print(f"Failed to fetch data for {target_date_str}: {e}")

# Aggregate the number of articles by date
result_all['Date'] = result_all['pubDate'].dt.strftime('%Y%m%d')
result_gr = result_all[['Date', 'title']].groupby('Date').count()

# Print the results
print(result_gr)

# Create directory for saving files
output_dir = "./data/raw"  # Navigate to the upper directory and refer to data/raw folder
os.makedirs(output_dir, exist_ok=True)

# Save news data in JSON format
news_data_path = os.path.join(output_dir, 'news_data.json')
result_all.to_json(news_data_path, orient='records', force_ascii=False, indent=4)

# Save the aggregated article count as a CSV file
news_summary_path = os.path.join(output_dir, 'news_data_summary.csv')
result_gr.to_csv(news_summary_path, encoding='utf-8-sig')

# Visualize the number of articles per date
plt.figure(figsize=(12, 6))
plt.bar(result_gr.index, result_gr['title'])
plt.xticks(rotation=90)
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.title('Number of Articles per Date (2020-01-01 to Today)')
plt.tight_layout()
plt.show()

print(f"News data saved as JSON: {news_data_path}")
print(f"Summary saved as CSV: {news_summary_path}")
