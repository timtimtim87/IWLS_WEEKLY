import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rolling_ranges_and_positions():
    """
    Calculate rolling ranges and position percentages for wave physics analysis
    """
    print("ROLLING RANGE WAVE ANALYSIS CALCULATOR")
    print("=" * 50)
    
    # Load the interpolated daily stock data
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    data_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/DATA"
    input_file = os.path.join(data_dir, "DAILY_INTERPOLATED_STOCK_DATA.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found at {input_file}")
        print("   Please run the interpolation script first.")
        return None
    
    print(f"📁 Loading interpolated daily data: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Successfully loaded {len(df):,} daily records")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Get asset columns (exclude 'date')
    asset_columns = [col for col in df.columns if col != 'date']
    total_assets = len(asset_columns)
    
    print(f"📊 Found {total_assets} assets to analyze")
    print(f"📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    # Define rolling windows (trading days)
    windows = {
        '3_month': 63,   # ~3 months of trading days
        '6_month': 126   # ~6 months of trading days
    }
    
    print(f"\n🔄 Calculating rolling ranges and positions...")
    print(f"   3-month window: {windows['3_month']} trading days")
    print(f"   6-month window: {windows['6_month']} trading days")
    
    # Process each asset
    all_results = []
    
    for i, asset in enumerate(asset_columns, 1):
        print(f"\n📈 Processing asset {i}/{total_assets}: {asset}")
        
        # Get asset data and remove NaN values
        asset_data = df[['date', asset]].dropna().copy()
        asset_data = asset_data.sort_values('date').reset_index(drop=True)
        
        if len(asset_data) < windows['6_month']:
            print(f"  ⚠️  Skipping {asset}: insufficient data ({len(asset_data)} days)")
            continue
        
        print(f"  📊 Processing {len(asset_data):,} data points...")
        
        # Calculate rolling metrics for each window
        for window_name, window_days in windows.items():
            print(f"    🔍 Calculating {window_name} metrics...")
            
            # Rolling min and max
            asset_data[f'rolling_low_{window_name}'] = asset_data[asset].rolling(
                window=window_days, min_periods=window_days//2
            ).min()
            
            asset_data[f'rolling_high_{window_name}'] = asset_data[asset].rolling(
                window=window_days, min_periods=window_days//2
            ).max()
            
            # Rolling range as percentage (from low to high)
            asset_data[f'rolling_range_pct_{window_name}'] = (
                (asset_data[f'rolling_high_{window_name}'] - asset_data[f'rolling_low_{window_name}']) / 
                asset_data[f'rolling_low_{window_name}']
            ) * 100
            
            # Current position within the range (0% = at low, 100% = at high)
            asset_data[f'position_pct_{window_name}'] = (
                (asset_data[asset] - asset_data[f'rolling_low_{window_name}']) / 
                (asset_data[f'rolling_high_{window_name}'] - asset_data[f'rolling_low_{window_name}'])
            ) * 100
            
            # Handle division by zero (when high = low)
            asset_data[f'position_pct_{window_name}'] = asset_data[f'position_pct_{window_name}'].fillna(50.0)
        
        # Add asset identifier and reshape for final dataset
        asset_data['asset'] = asset
        asset_data['price'] = asset_data[asset]
        
        # Select final columns
        final_columns = [
            'date', 'asset', 'price',
            'rolling_range_pct_3_month', 'position_pct_3_month',
            'rolling_range_pct_6_month', 'position_pct_6_month',
            'rolling_low_3_month', 'rolling_high_3_month',
            'rolling_low_6_month', 'rolling_high_6_month'
        ]
        
        asset_final = asset_data[final_columns].copy()
        all_results.append(asset_final)
        
        # Show sample statistics for this asset
        valid_3m = asset_final['rolling_range_pct_3_month'].dropna()
        valid_6m = asset_final['rolling_range_pct_6_month'].dropna()
        
        if len(valid_3m) > 0 and len(valid_6m) > 0:
            print(f"  📊 Sample stats:")
            print(f"    3m range: avg={valid_3m.mean():.1f}%, max={valid_3m.max():.1f}%")
            print(f"    6m range: avg={valid_6m.mean():.1f}%, max={valid_6m.max():.1f}%")
    
    if not all_results:
        print("❌ No assets processed successfully.")
        return None
    
    # Combine all results
    print(f"\n🔗 Combining results from {len(all_results)} assets...")
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Sort by date and asset
    combined_df = combined_df.sort_values(['date', 'asset']).reset_index(drop=True)
    
    # Summary statistics
    print(f"\n📊 WAVE ANALYSIS SUMMARY:")
    print(f"   Total records: {len(combined_df):,}")
    print(f"   Assets processed: {combined_df['asset'].nunique()}")
    print(f"   Date range: {combined_df['date'].min().strftime('%Y-%m-%d')} to {combined_df['date'].max().strftime('%Y-%m-%d')}")
    
    # Rolling range statistics
    valid_data = combined_df.dropna(subset=['rolling_range_pct_3_month', 'rolling_range_pct_6_month'])
    
    if len(valid_data) > 0:
        print(f"\n📈 ROLLING RANGE STATISTICS:")
        print(f"   3-month ranges:")
        print(f"     Average: {valid_data['rolling_range_pct_3_month'].mean():.1f}%")
        print(f"     Median:  {valid_data['rolling_range_pct_3_month'].median():.1f}%")
        print(f"     Max:     {valid_data['rolling_range_pct_3_month'].max():.1f}%")
        
        print(f"   6-month ranges:")
        print(f"     Average: {valid_data['rolling_range_pct_6_month'].mean():.1f}%")
        print(f"     Median:  {valid_data['rolling_range_pct_6_month'].median():.1f}%")
        print(f"     Max:     {valid_data['rolling_range_pct_6_month'].max():.1f}%")
        
        # Position distribution
        print(f"\n📍 POSITION DISTRIBUTION:")
        print(f"   3-month positions:")
        print(f"     Average: {valid_data['position_pct_3_month'].mean():.1f}%")
        print(f"     <25% (bottom quartile): {(valid_data['position_pct_3_month'] < 25).sum():,} records")
        print(f"     >75% (top quartile): {(valid_data['position_pct_3_month'] > 75).sum():,} records")
        
        print(f"   6-month positions:")
        print(f"     Average: {valid_data['position_pct_6_month'].mean():.1f}%")
        print(f"     <25% (bottom quartile): {(valid_data['position_pct_6_month'] < 25).sum():,} records")
        print(f"     >75% (top quartile): {(valid_data['position_pct_6_month'] > 75).sum():,} records")
    
    # Identify high volatility assets
    print(f"\n🌊 HIGH VOLATILITY ASSETS:")
    asset_avg_ranges = valid_data.groupby('asset').agg({
        'rolling_range_pct_6_month': 'mean',
        'rolling_range_pct_3_month': 'mean'
    }).round(1)
    
    # Sort by 6-month average range
    top_volatile = asset_avg_ranges.sort_values('rolling_range_pct_6_month', ascending=False)
    
    print(f"   Top 10 most volatile assets (by 6-month average range):")
    for i, (asset, row) in enumerate(top_volatile.head(10).iterrows(), 1):
        print(f"   {i:>2}. {asset}: 6m={row['rolling_range_pct_6_month']:.1f}%, 3m={row['rolling_range_pct_3_month']:.1f}%")
    
    # Save results
    output_file = os.path.join(results_dir, "ROLLING_RANGE_WAVE_ANALYSIS.csv")
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"   File: {output_file}")
    print(f"   Size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # Save asset summary
    summary_file = os.path.join(results_dir, "ASSET_VOLATILITY_SUMMARY.csv")
    asset_avg_ranges.to_csv(summary_file)
    print(f"   Summary: {summary_file}")
    
    print(f"\n✨ WAVE ANALYSIS COMPLETE!")
    print(f"🎯 Key Outputs:")
    print(f"   📊 Rolling range percentages (3m & 6m) for volatility measurement")
    print(f"   📍 Position percentages showing where price sits within range")
    print(f"   🌊 Asset volatility rankings for identifying 'loaded springs'")
    print(f"   🔍 Ready for wave physics backtesting strategies!")
    
    return combined_df

def show_sample_wave_data():
    """
    Display sample of the wave analysis data
    """
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    results_file = os.path.join(results_dir, "ROLLING_RANGE_WAVE_ANALYSIS.csv")
    
    if not os.path.exists(results_file):
        print("No wave analysis results found. Run the calculation first.")
        return
    
    print(f"\n📋 SAMPLE WAVE ANALYSIS DATA:")
    print("-" * 80)
    
    df = pd.read_csv(results_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Show recent data for a few assets
    recent_data = df[df['date'] >= df['date'].max() - pd.Timedelta(days=30)]
    sample_assets = recent_data['asset'].unique()[:3]
    
    for asset in sample_assets:
        asset_data = recent_data[recent_data['asset'] == asset].tail(5)
        print(f"\n{asset} (last 5 trading days):")
        print(f"{'Date':>12} | {'Price':>8} | {'3m Range':>9} | {'3m Pos':>7} | {'6m Range':>9} | {'6m Pos':>7}")
        print("-" * 70)
        
        for _, row in asset_data.iterrows():
            if pd.notna(row['rolling_range_pct_3_month']):
                print(f"{row['date'].strftime('%Y-%m-%d'):>12} | "
                      f"${row['price']:>7.2f} | "
                      f"{row['rolling_range_pct_3_month']:>8.1f}% | "
                      f"{row['position_pct_3_month']:>6.1f}% | "
                      f"{row['rolling_range_pct_6_month']:>8.1f}% | "
                      f"{row['position_pct_6_month']:>6.1f}%")

def analyze_volatility_persistence():
    """
    Quick analysis of volatility persistence patterns
    """
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    results_file = os.path.join(results_dir, "ROLLING_RANGE_WAVE_ANALYSIS.csv")
    
    if not os.path.exists(results_file):
        print("No wave analysis results found. Run the calculation first.")
        return
    
    print(f"\n🔄 VOLATILITY PERSISTENCE ANALYSIS:")
    print("-" * 50)
    
    df = pd.read_csv(results_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Define high volatility threshold (top quartile)
    high_vol_threshold = df['rolling_range_pct_6_month'].quantile(0.75)
    
    print(f"High volatility threshold (6m range): {high_vol_threshold:.1f}%")
    
    # Find assets with most high-volatility periods
    high_vol_data = df[df['rolling_range_pct_6_month'] > high_vol_threshold]
    asset_high_vol_counts = high_vol_data['asset'].value_counts()
    
    print(f"\nAssets with most high-volatility periods:")
    for i, (asset, count) in enumerate(asset_high_vol_counts.head(10).iterrows(), 1):
        total_records = len(df[df['asset'] == asset])
        pct = (count / total_records) * 100
        print(f"   {i:>2}. {asset}: {count:>4,} periods ({pct:>5.1f}% of time)")

if __name__ == "__main__":
    # Calculate rolling ranges and wave analysis
    wave_data = calculate_rolling_ranges_and_positions()
    
    if wave_data is not None:
        # Show sample of the data
        show_sample_wave_data()
        
        # Quick volatility persistence analysis
        analyze_volatility_persistence()
        
        print(f"\n🚀 Ready for Wave Physics Strategy Development!")
        print(f"   🎯 Next step: Build backtests targeting high-volatility assets")
        print(f"   🌊 Test theory: Persistent volatility → eventual profitable alignment")