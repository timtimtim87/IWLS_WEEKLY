import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def extract_asset_from_filename(filename):
    """
    Extract the main asset symbol from filename like 'BATS_AXP, 1W.csv'
    """
    # Remove path and extension
    base_name = os.path.basename(filename)
    # Remove .csv extension
    name_without_ext = base_name.replace('.csv', '')
    # Split by 'BATS_' and take the part before ', 1W'
    if 'BATS_' in name_without_ext:
        asset_part = name_without_ext.split('BATS_')[1]
        # Remove the ', 1W' part
        main_asset = asset_part.split(',')[0].strip()
        return main_asset
    else:
        # Fallback - just take first part before comma
        return name_without_ext.split(',')[0].strip()

def parse_column_headers(df):
    """
    Parse the column headers to extract clean asset symbols
    The format is like 'MU · NASDAQ: close' or '**AMGN · NASDAQ: close'
    """
    clean_columns = ['time']  # Keep time as-is
    
    for col in df.columns[1:]:  # Skip 'time' column
        if col == 'close':
            # This is the main asset from the filename
            clean_columns.append('MAIN_ASSET')
        else:
            # Extract symbol from format like 'MU · NASDAQ: close' or '**AMGN · NASDAQ: close'
            if ' · ' in col and ': close' in col:
                # Remove ** prefix if present
                symbol_part = col.split(' · ')[0].replace('**', '').strip()
                clean_columns.append(symbol_part)
            else:
                # Fallback - use column as-is
                clean_columns.append(col)
    
    return clean_columns

def load_and_process_file(filepath):
    """
    Load a single CSV file and process it
    """
    print(f"Loading {filepath}...")
    
    try:
        # Load the CSV
        df = pd.read_csv(filepath)
        
        # Extract main asset from filename
        main_asset = extract_asset_from_filename(filepath)
        
        # Clean column headers
        clean_columns = parse_column_headers(df)
        df.columns = clean_columns
        
        # Replace 'MAIN_ASSET' with actual asset symbol
        df.columns = [col.replace('MAIN_ASSET', main_asset) for col in df.columns]
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Convert time column to date (since we're dealing with weekly data)
        df['date'] = df['time'].dt.date
        df = df.drop('time', axis=1)
        
        # Set date as index for easier merging
        df = df.set_index('date')
        
        # Replace NaN with None for cleaner output
        df = df.replace({np.nan: None})
        
        print(f"  Main asset: {main_asset}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")
        
        return df, main_asset
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

def merge_weekly_data():
    """
    Main function to merge all weekly stock data files
    """
    print("WEEKLY STOCK DATA MERGER")
    print("=" * 50)
    
    # Data directory
    data_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/DATA"
    
    # Find all CSV files that match the pattern
    csv_files = []
    for filename in os.listdir(data_dir):
        if filename.startswith('BATS_') and filename.endswith('1W.csv'):
            csv_files.append(os.path.join(data_dir, filename))
    
    if not csv_files:
        print("No CSV files found matching pattern 'BATS_*1W.csv'")
        return
    
    print(f"Found {len(csv_files)} files to merge:")
    for file in csv_files:
        print(f"  {os.path.basename(file)}")
    
    print()
    
    # Load and process each file
    all_dataframes = []
    main_assets = []
    
    for filepath in csv_files:
        df, main_asset = load_and_process_file(filepath)
        if df is not None:
            all_dataframes.append(df)
            main_assets.append(main_asset)
        print()
    
    if not all_dataframes:
        print("No data loaded successfully.")
        return
    
    # Merge all dataframes
    print("Merging all datasets...")
    
    # Start with the first dataframe
    merged_df = all_dataframes[0].copy()
    
    # Merge each subsequent dataframe
    for i, df in enumerate(all_dataframes[1:], 1):
        print(f"  Merging dataset {i+1}...")
        merged_df = merged_df.join(df, how='outer', rsuffix=f'_dup_{i}')
    
    # Handle duplicate columns (if any assets appear in multiple files)
    # Keep the first occurrence of each column
    columns_to_keep = []
    seen_columns = set()
    
    for col in merged_df.columns:
        base_col = col.split('_dup_')[0]  # Remove duplicate suffix
        if base_col not in seen_columns:
            columns_to_keep.append(col)
            seen_columns.add(base_col)
    
    # Select only the columns we want to keep and rename them properly
    final_df = merged_df[columns_to_keep].copy()
    final_df.columns = [col.split('_dup_')[0] for col in columns_to_keep]
    
    # Reset index to make date a regular column
    final_df = final_df.reset_index()
    
    # Convert date back to datetime for consistency
    final_df['date'] = pd.to_datetime(final_df['date'])
    
    # Sort by date
    final_df = final_df.sort_values('date').reset_index(drop=True)
    
    # Summary statistics
    print(f"\nMERGED DATASET SUMMARY:")
    print(f"  Total rows (weeks): {len(final_df)}")
    print(f"  Date range: {final_df['date'].min().strftime('%Y-%m-%d')} to {final_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Total assets: {len(final_df.columns) - 1}")  # -1 for date column
    print(f"  Assets: {', '.join(sorted([col for col in final_df.columns if col != 'date']))}")
    
    # Check data completeness
    print(f"\nDATA COMPLETENESS:")
    for col in sorted([col for col in final_df.columns if col != 'date']):
        non_null_count = final_df[col].count()
        completeness = (non_null_count / len(final_df)) * 100
        print(f"  {col}: {non_null_count}/{len(final_df)} ({completeness:.1f}%)")
    
    # Save merged dataset
    output_file = os.path.join(data_dir, "MERGED_WEEKLY_STOCK_DATA.csv")
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Merged dataset saved to: {output_file}")
    print(f"\nFirst few rows:")
    print(final_df.head())
    
    # Show data types
    print(f"\nData types:")
    print(final_df.dtypes)
    
    return final_df

if __name__ == "__main__":
    merged_data = merge_weekly_data()
    
    if merged_data is not None:
        print(f"\n🎯 SUCCESS: Weekly stock data merged successfully!")
        print(f"   Use 'MERGED_WEEKLY_STOCK_DATA.csv' for your IWLS analysis")
        print(f"   Dataset spans {len(merged_data)} weeks with {len(merged_data.columns)-1} assets")