import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_forward_performance_metrics(prices, current_idx, forward_days):
    """
    Calculate forward performance metrics for a given period
    
    Returns:
    - forward_return: Total return from current price to price at end of period
    - max_gain: Maximum gain achieved during the period
    - max_drawdown: Maximum drawdown from any peak during the period
    """
    current_price = prices.iloc[current_idx]
    
    # Get forward period data
    end_idx = current_idx + forward_days
    if end_idx >= len(prices):
        # Not enough forward data
        return np.nan, np.nan, np.nan
    
    forward_prices = prices.iloc[current_idx:end_idx+1]
    
    if len(forward_prices) < forward_days * 0.8:  # Need at least 80% of expected days
        return np.nan, np.nan, np.nan
    
    # Forward Return: Change from current price to end price
    end_price = forward_prices.iloc[-1]
    forward_return = ((end_price / current_price) - 1) * 100
    
    # Max Gain: Highest point reached during the period
    max_price = forward_prices.max()
    max_gain = ((max_price / current_price) - 1) * 100
    
    # Max Drawdown: Largest drop from any peak during the period
    # Calculate running maximum (peak) and drawdown from that peak
    running_max = forward_prices.expanding().max()
    drawdowns = ((forward_prices - running_max) / running_max) * 100
    max_drawdown = drawdowns.min()  # Most negative value
    
    return forward_return, max_gain, max_drawdown

def add_forward_performance_to_iwls_data():
    """
    Add forward performance metrics to existing IWLS dataset
    """
    print("FORWARD PERFORMANCE CALCULATOR")
    print("=" * 50)
    
    # Load the IWLS results
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    input_file = os.path.join(results_dir, "IWLS_GROWTH_RATES_ALL_ASSETS.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: IWLS results file not found at {input_file}")
        print("   Please run the IWLS calculator script first.")
        return None
    
    print(f"📁 Loading: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Successfully loaded {len(df):,} IWLS records")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Get unique assets
    assets = df['asset'].unique()
    print(f"📊 Found {len(assets)} assets to process")
    
    # Define forward periods (in trading days)
    periods = {
        '6_month': 126,   # ~6 months * 21 trading days
        '12_month': 252,  # ~12 months * 21 trading days  
        '18_month': 378   # ~18 months * 21 trading days
    }
    
    # Find the latest date in dataset
    latest_date = df['date'].max()
    print(f"📅 Latest date in dataset: {latest_date.strftime('%Y-%m-%d')}")
    
    # Calculate cutoff dates for each period (dates after which we can't calculate forward metrics)
    cutoff_dates = {}
    for period_name, days in periods.items():
        cutoff_date = latest_date - pd.Timedelta(days=days)
        cutoff_dates[period_name] = cutoff_date
        print(f"   {period_name}: Can calculate forward metrics until {cutoff_date.strftime('%Y-%m-%d')}")
    
    print()
    
    # Process each asset
    all_results = []
    
    for i, asset in enumerate(assets, 1):
        print(f"Processing asset {i}/{len(assets)}: {asset}")
        
        # Get data for this asset, sorted by date
        asset_data = df[df['asset'] == asset].sort_values('date').reset_index(drop=True)
        
        # Skip if insufficient data
        if len(asset_data) < periods['18_month'] + 100:  # Need extra buffer
            print(f"  Skipping {asset}: insufficient data ({len(asset_data):,} records)")
            continue
        
        # Get clean price series (no NaN values)
        clean_data = asset_data.dropna(subset=['price']).reset_index(drop=True)
        
        if len(clean_data) < periods['18_month'] + 50:
            print(f"  Skipping {asset}: insufficient clean price data ({len(clean_data):,} records)")
            continue
        
        print(f"  Processing {len(clean_data):,} price records...")
        
        # Calculate forward metrics for each record
        asset_results = []
        
        for idx, row in clean_data.iterrows():
            current_date = row['date']
            
            # Initialize result row
            result_row = row.copy()
            
            # Calculate forward metrics for each period
            for period_name, forward_days in periods.items():
                
                # Check if we have enough forward data
                if current_date > cutoff_dates[period_name]:
                    # Not enough forward data for this period
                    result_row[f'forward_return_{period_name}'] = np.nan
                    result_row[f'max_gain_{period_name}'] = np.nan
                    result_row[f'max_drawdown_{period_name}'] = np.nan
                else:
                    # Calculate forward metrics
                    forward_return, max_gain, max_drawdown = calculate_forward_performance_metrics(
                        clean_data['price'], idx, forward_days
                    )
                    
                    result_row[f'forward_return_{period_name}'] = forward_return
                    result_row[f'max_gain_{period_name}'] = max_gain
                    result_row[f'max_drawdown_{period_name}'] = max_drawdown
            
            asset_results.append(result_row)
            
            # Progress indicator
            if (idx + 1) % 500 == 0:
                print(f"    Processed {idx + 1:,}/{len(clean_data):,} records...")
        
        # Convert to DataFrame and add to results
        asset_df = pd.DataFrame(asset_results)
        all_results.append(asset_df)
        
        # Show sample statistics for this asset
        sample_data = asset_df.dropna(subset=['forward_return_6_month'])
        if len(sample_data) > 0:
            print(f"  ✅ {asset}: {len(sample_data):,} records with 6m forward data")
            print(f"      Avg 6m return: {sample_data['forward_return_6_month'].mean():.1f}%")
            print(f"      Avg 6m max gain: {sample_data['max_gain_6_month'].mean():.1f}%")
            print(f"      Avg 6m max drawdown: {sample_data['max_drawdown_6_month'].mean():.1f}%")
        else:
            print(f"  ⚠️  {asset}: No forward performance data available")
        
        print()
    
    if not all_results:
        print("❌ No assets had sufficient data for forward performance calculation.")
        return None
    
    # Combine all results
    print("🔄 Combining all results...")
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Generate summary statistics
    total_records = len(combined_df)
    print(f"📊 FORWARD PERFORMANCE SUMMARY:")
    print(f"   Total records: {total_records:,}")
    
    # Count valid forward data by period
    for period_name in periods.keys():
        valid_count = combined_df[f'forward_return_{period_name}'].count()
        percentage = (valid_count / total_records) * 100
        print(f"   {period_name} forward data: {valid_count:,} records ({percentage:.1f}%)")
        
        if valid_count > 0:
            period_data = combined_df.dropna(subset=[f'forward_return_{period_name}'])
            print(f"      Avg return: {period_data[f'forward_return_{period_name}'].mean():.1f}%")
            print(f"      Avg max gain: {period_data[f'max_gain_{period_name}'].mean():.1f}%")
            print(f"      Avg max drawdown: {period_data[f'max_drawdown_{period_name}'].mean():.1f}%")
    
    # Save enhanced dataset
    output_file = os.path.join(results_dir, "IWLS_WITH_FORWARD_PERFORMANCE.csv")
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"   File: {output_file}")
    print(f"   Size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # Create summary pivot tables
    print(f"\n🔄 Creating summary pivot tables...")
    
    # Performance metrics pivot tables
    performance_metrics = []
    for period_name in periods.keys():
        for metric in ['forward_return', 'max_gain', 'max_drawdown']:
            performance_metrics.append(f'{metric}_{period_name}')
    
    for metric in performance_metrics:
        pivot_data = combined_df.pivot(index='date', columns='asset', values=metric)
        pivot_file = os.path.join(results_dir, f"{metric.upper()}_PIVOT.csv")
        pivot_data.to_csv(pivot_file)
        print(f"   Saved: {metric.upper()}_PIVOT.csv")
    
    print(f"\n🎯 SUCCESS!")
    print(f"   Forward performance metrics calculated for all assets")
    print(f"   Use 'IWLS_WITH_FORWARD_PERFORMANCE.csv' for comprehensive analysis")
    print(f"   New columns added:")
    for period_name in periods.keys():
        print(f"     - forward_return_{period_name}, max_gain_{period_name}, max_drawdown_{period_name}")
    
    return combined_df

def analyze_forward_performance_sample():
    """
    Show sample analysis of the forward performance data
    """
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    results_file = os.path.join(results_dir, "IWLS_WITH_FORWARD_PERFORMANCE.csv")
    
    if not os.path.exists(results_file):
        print("No forward performance results file found. Run the calculation first.")
        return
    
    print("\n📋 SAMPLE FORWARD PERFORMANCE ANALYSIS:")
    print("-" * 80)
    
    df = pd.read_csv(results_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Show sample records with all forward metrics
    sample_data = df.dropna(subset=['forward_return_6_month', 'forward_return_12_month', 'forward_return_18_month'])
    
    if len(sample_data) > 0:
        print(f"Records with complete 18-month forward data: {len(sample_data):,}")
        
        # Show a few examples
        sample_assets = sample_data['asset'].unique()[:2]
        
        for asset in sample_assets:
            asset_sample = sample_data[sample_data['asset'] == asset].tail(3)
            print(f"\n{asset} - Recent examples:")
            
            for _, row in asset_sample.iterrows():
                print(f"  {row['date'].strftime('%Y-%m-%d')} | "
                      f"Price: ${row['price']:.2f} | "
                      f"Deviation: {row['price_deviation_from_trend']:.1f}%")
                print(f"    6m:  Return={row['forward_return_6_month']:.1f}% | "
                      f"Max Gain={row['max_gain_6_month']:.1f}% | "
                      f"Max DD={row['max_drawdown_6_month']:.1f}%")
                print(f"    12m: Return={row['forward_return_12_month']:.1f}% | "
                      f"Max Gain={row['max_gain_12_month']:.1f}% | "
                      f"Max DD={row['max_drawdown_12_month']:.1f}%")
                print(f"    18m: Return={row['forward_return_18_month']:.1f}% | "
                      f"Max Gain={row['max_gain_18_month']:.1f}% | "
                      f"Max DD={row['max_drawdown_18_month']:.1f}%")
                print()
    else:
        print("No records found with complete forward performance data.")
        
        # Show what data is available
        for period in ['6_month', '12_month', '18_month']:
            count = df[f'forward_return_{period}'].count()
            if count > 0:
                print(f"{period} forward data: {count:,} records available")

if __name__ == "__main__":
    # Calculate forward performance metrics
    results = add_forward_performance_to_iwls_data()
    
    if results is not None:
        # Show sample analysis
        analyze_forward_performance_sample()
        
        print(f"\n✨ Analysis Ready!")
        print(f"   You can now correlate current deviation patterns with future performance")
        print(f"   Example: Does -40% deviation + 'consistent_decline' pattern predict")
        print(f"            different outcomes than -40% + 'recent_shock' pattern?")