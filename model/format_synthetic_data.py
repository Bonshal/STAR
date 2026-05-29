import os
import pandas as pd
import numpy as np

# Paths
input_csv = r"c:\Users\bonsh\Desktop\Projects\FCCI\backend\synthetic_bank_data.csv"
output_dir = r"c:\Users\bonsh\Desktop\Projects\FCCI\Multi-GNN\data\synthetic"
output_csv = os.path.join(output_dir, "formatted_transactions.csv")

os.makedirs(output_dir, exist_ok=True)

print(f"Loading raw synthetic data from: {input_csv}")
df = pd.read_csv(input_csv)

# Print columns and size
print(f"Columns: {df.columns.tolist()}")
print(f"Number of rows: {len(df)}")

# Generate account mapping
all_accounts = pd.concat([df['sender'], df['receiver']]).unique()
account_to_id = {acc: idx for idx, acc in enumerate(all_accounts)}

print(f"Mapped {len(account_to_id)} unique entity accounts.")

# Convert ts to seconds from minimum timestamp
df['ts_datetime'] = pd.to_datetime(df['ts'])
min_ts = df['ts_datetime'].min()
df['Timestamp'] = (df['ts_datetime'] - min_ts).dt.total_seconds().astype(int)

# Map edge fields
formatted_df = pd.DataFrame()
formatted_df['EdgeID'] = df.index
formatted_df['from_id'] = df['sender'].map(account_to_id)
formatted_df['to_id'] = df['receiver'].map(account_to_id)
formatted_df['Timestamp'] = df['Timestamp']
formatted_df['Amount Sent'] = df['amount']
formatted_df['Sent Currency'] = 0  # 0 representing US Dollar
formatted_df['Amount Received'] = df['amount']
formatted_df['Received Currency'] = 0  # 0 representing US Dollar
formatted_df['Payment Format'] = 0  # 0 representing ACH
formatted_df['Is Laundering'] = df['label']

# Sort by Timestamp just like in format_kaggle_files.py
formatted_df = formatted_df.sort_values('Timestamp')

# Save
formatted_df.to_csv(output_csv, index=False)
print(f"Saved formatted transactions dataset to: {output_csv}")
