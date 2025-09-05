import pandas as pd
import numpy as np
from datetime import datetime
import os

def interpolate_weekly_to_daily(input_file, output_file=None):
    """
    Convert weekly SPY stock data to daily by interpolating closing prices.
    Step 1: Create daily rows and place known weekly values
    Step 2: Interpolate between known points
    
    Args:
        input_file (str): Path to input CSV file with weekly data
        output_file (str): Path to output CSV file (optional)
    
    Returns:
        pd.DataFrame: DataFrame with daily interpolated data
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Reading data from: {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Keep only time and close columns, drop rows with NaN close prices
    df = df[['time', 'close']].copy()
    df = df.dropna(subset=['close'])  # Remove rows where close is NaN
    
    print(f"Valid data points after removing NaN: {len(df)}")
    
    # Convert Unix timestamps to datetime
    df['date'] = pd.to_datetime(df['time'], unit='s').dt.date
    
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Sample of known values:")
    print(df[['date', 'close']].head())
    
    # Step 1: Create daily date range and DataFrame
    start_date = df['date'].min()
    end_date = df['date'].max()
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"Creating {len(daily_dates)} daily data points")
    
    # Create daily DataFrame with all dates
    daily_df = pd.DataFrame({
        'date': daily_dates.date,
        'time': (daily_dates.astype('int64') // 10**9).astype(int),
        'close': np.nan
    })
    
    # Step 2: Map known weekly values to their corresponding dates
    for _, row in df.iterrows():
        mask = daily_df['date'] == row['date']
        if mask.any():
            daily_df.loc[mask, 'close'] = row['close']
            print(f"Mapped {row['date']}: ${row['close']}")
    
    print(f"\nKnown values mapped: {daily_df['close'].notna().sum()}")
    print(f"Values to interpolate: {daily_df['close'].isna().sum()}")
    
    # Step 3: Interpolate between known points
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    daily_df['close'] = daily_df['close'].interpolate(method='linear')
    
    # Check if there are still NaN values at the beginning or end
    if daily_df['close'].isna().any():
        print("Warning: Some NaN values remain (likely at start/end)")
        # Forward fill and backward fill to handle edge cases
        daily_df['close'] = daily_df['close'].fillna(method='ffill').fillna(method='bfill')
    
    # Keep only time and close columns for output
    result_df = daily_df[['time', 'close']].copy()
    
    print(f"Final interpolated data shape: {result_df.shape}")
    print(f"Close price range: ${result_df['close'].min():.4f} - ${result_df['close'].max():.4f}")
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_daily.csv"
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Daily data saved to: {output_file}")
    
    return result_df

# Main execution
if __name__ == "__main__":
    # Your specific file path
    input_file = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/DATA/BATS_SPY, 1W-3.csv"
    
    try:
        # Run the interpolation
        daily_data = interpolate_weekly_to_daily(input_file)
        
        # Display first few rows of result
        print("\nFirst 10 rows of daily data:")
        print(daily_data.head(10))
        
        print("\nLast 10 rows of daily data:")
        print(daily_data.tail(10))
        
        # Show some sample interpolated values
        print("\nSample of data around known points:")
        sample_indices = [100, 500, 1000, 1500, 2000]
        for idx in sample_indices:
            if idx < len(daily_data):
                timestamp = daily_data.iloc[idx]['time']
                date = pd.to_datetime(timestamp, unit='s').date()
                close = daily_data.iloc[idx]['close']
                print(f"Index {idx}: {date} - ${close:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the file path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")