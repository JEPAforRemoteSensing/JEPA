import pandas as pd

# Read the metadata parquet file
# Adjust the path if needed
metadata_path = 'data/BEN_14k/metadata.parquet'

print(f"Reading metadata from: {metadata_path}")
print("=" * 80)

try:
    df = pd.read_parquet(metadata_path)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n" + "=" * 80)
    print("Column Names:")
    print("=" * 80)
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    
    print("\n" + "=" * 80)
    print("Data Types:")
    print("=" * 80)
    print(df.dtypes)
    
    print("\n" + "=" * 80)
    print("First 10 Rows:")
    print("=" * 80)
    print(df.head(10))
    
    print("\n" + "=" * 80)
    print("Basic Statistics:")
    print("=" * 80)
    print(df.describe())
    
    print("\n" + "=" * 80)
    print("Missing Values:")
    print("=" * 80)
    print(df.isnull().sum())
    
    print("\n" + "=" * 80)
    print("Info:")
    print("=" * 80)
    df.info()
    
except FileNotFoundError:
    print(f"\nError: File not found at '{metadata_path}'")
    print("\nPlease update the 'metadata_path' variable with the correct path.")
    print("The file might be in one of these locations:")
    print("  - data/BEN_14k/metadata.parquet")
    print("  - BEN_full/metadata.parquet")
except Exception as e:
    print(f"\nError reading file: {e}")
