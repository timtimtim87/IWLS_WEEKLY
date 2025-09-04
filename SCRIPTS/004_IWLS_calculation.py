import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def iwls_regression(x_vals, y_vals, iterations=5):
    """
    Perform Iteratively Weighted Least Squares regression
    """
    n = len(x_vals)
    if n < 2:
        return np.nan, np.nan
    
    # Initialize weights to 1.0
    weights = np.ones(n)
    
    slope, intercept = np.nan, np.nan
    
    for iter in range(iterations):
        # Calculate weighted sums
        sum_w = np.sum(weights)
        sum_wx = np.sum(weights * x_vals)
        sum_wy = np.sum(weights * y_vals)
        sum_wxx = np.sum(weights * x_vals * x_vals)
        sum_wxy = np.sum(weights * x_vals * y_vals)
        
        # Calculate slope and intercept
        denominator = sum_w * sum_wxx - sum_wx * sum_wx
        if abs(denominator) > 1e-10:
            slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denominator
            intercept = (sum_wy - slope * sum_wx) / sum_w
            
            # Calculate residuals and update weights (except on last iteration)
            if iter < iterations - 1:
                predicted = intercept + slope * x_vals
                residuals = np.abs(y_vals - predicted)
                mean_residual = np.mean(residuals)
                
                # Update weights (inverse of residual distance)
                new_weights = 1.0 / (residuals + mean_residual * 0.1)
                weights = new_weights
        else:
            break
    
    return slope, intercept

def calculate_iwls_for_asset(asset_data, asset_name, lookback_period=1500):
    """
    Calculate IWLS growth rate and deviation for a single asset
    """
    print(f"  Processing {asset_name}...")
    
    results = []
    
    for i in range(len(asset_data)):
        current_date = asset_data.iloc[i]['date']
        current_price = asset_data.iloc[i][asset_name]
        
        if pd.isna(current_price):
            # No price data for this date
            results.append({
                'date': current_date,
                'asset': asset_name,
                'price': np.nan,
                'iwls_annual_growth_rate': np.nan,
                'iwls_trend_line_value': np.nan,
                'price_deviation_from_trend': np.nan,
                'slope': np.nan,
                'intercept': np.nan
            })
            continue
        
        if i < lookback_period - 1:
            # Not enough data for IWLS calculation
            results.append({
                'date': current_date,
                'asset': asset_name,
                'price': current_price,
                'iwls_annual_growth_rate': np.nan,
                'iwls_trend_line_value': np.nan,
                'price_deviation_from_trend': np.nan,
                'slope': np.nan,
                'intercept': np.nan
            })
            continue
        
        # Get lookback data (only use rows where this asset has valid data)
        start_idx = i - lookback_period + 1
        end_idx = i + 1
        lookback_data = asset_data.iloc[start_idx:end_idx]
        
        # Filter out rows where this asset has NaN values
        valid_lookback = lookback_data[pd.notna(lookback_data[asset_name])]
        
        if len(valid_lookback) < 50:  # Need minimum valid data points
            results.append({
                'date': current_date,
                'asset': asset_name,
                'price': current_price,
                'iwls_annual_growth_rate': np.nan,
                'iwls_trend_line_value': np.nan,
                'price_deviation_from_trend': np.nan,
                'slope': np.nan,
                'intercept': np.nan
            })
            continue
        
        # Prepare data for regression (x = time index, y = log price)
        x_vals = np.arange(len(valid_lookback), dtype=float)
        y_vals = np.log(valid_lookback[asset_name].values)
        
        # Perform IWLS regression
        slope, intercept = iwls_regression(x_vals, y_vals)
        
        # Calculate results
        if not np.isnan(slope) and not np.isnan(intercept):
            # Convert slope to annual growth rate (252 trading days per year)
            annual_growth_rate = (np.exp(slope * 252) - 1) * 100
            
            # Calculate trend line value at current point
            trend_line_log = intercept + slope * (len(valid_lookback) - 1)
            trend_line_value = np.exp(trend_line_log)
            
            # Calculate deviation from trend
            price_deviation = ((current_price / trend_line_value) - 1) * 100
            
        else:
            annual_growth_rate = np.nan
            trend_line_value = np.nan
            price_deviation = np.nan
        
        results.append({
            'date': current_date,
            'asset': asset_name,
            'price': current_price,
            'iwls_annual_growth_rate': annual_growth_rate,
            'iwls_trend_line_value': trend_line_value,
            'price_deviation_from_trend': price_deviation,
            'slope': slope,
            'intercept': intercept
        })
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1:,}/{len(asset_data):,} days...")
    
    return pd.DataFrame(results)

def calculate_iwls_for_all_assets():
    """
    Main function to calculate IWLS for all assets in the interpolated dataset
    """
    print("IWLS GROWTH RATE CALCULATOR")
    print("=" * 50)
    
    # Load the interpolated daily data
    data_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/DATA"
    input_file = os.path.join(data_dir, "DAILY_INTERPOLATED_STOCK_DATA.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found at {input_file}")
        print("   Please run the interpolation script first.")
        return None
    
    print(f"📁 Loading: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Successfully loaded {len(df):,} daily data points")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Get asset columns (exclude 'date')
    asset_columns = [col for col in df.columns if col != 'date']
    total_assets = len(asset_columns)
    
    print(f"📊 Found {total_assets} assets to process")
    print(f"📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"🔧 Using lookback period: 1,500 days (~6 years)")
    print()
    
    # Process each asset
    all_results = []
    
    for i, asset in enumerate(asset_columns, 1):
        print(f"Processing asset {i}/{total_assets}: {asset}")
        
        # Check if asset has sufficient data
        asset_data_count = df[asset].count()
        if asset_data_count < 2000:
            print(f"  Skipping {asset}: insufficient data ({asset_data_count:,} valid points)")
            continue
        
        # Create asset-specific dataframe with date and this asset's price
        asset_df = df[['date', asset]].copy()
        
        # Calculate IWLS for this asset
        asset_results = calculate_iwls_for_asset(asset_df, asset)
        
        # Add to master results
        all_results.append(asset_results)
        
        print(f"  ✅ Completed {asset}")
        print()
    
    if not all_results:
        print("❌ No assets had sufficient data for processing.")
        return None
    
    # Combine all results
    print("🔄 Combining all results...")
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Create summary statistics
    total_calculations = len(combined_results)
    valid_calculations = combined_results['iwls_annual_growth_rate'].count()
    
    print(f"📊 PROCESSING SUMMARY:")
    print(f"   Total asset-date combinations: {total_calculations:,}")
    print(f"   Valid IWLS calculations: {valid_calculations:,}")
    print(f"   Success rate: {(valid_calculations/total_calculations)*100:.1f}%")
    
    # Show sample statistics
    if valid_calculations > 0:
        valid_data = combined_results.dropna(subset=['iwls_annual_growth_rate'])
        
        print(f"\n📈 IWLS GROWTH RATE STATISTICS:")
        print(f"   Mean annual growth rate: {valid_data['iwls_annual_growth_rate'].mean():.2f}%")
        print(f"   Median annual growth rate: {valid_data['iwls_annual_growth_rate'].median():.2f}%")
        print(f"   Std dev: {valid_data['iwls_annual_growth_rate'].std():.2f}%")
        print(f"   Min: {valid_data['iwls_annual_growth_rate'].min():.2f}%")
        print(f"   Max: {valid_data['iwls_annual_growth_rate'].max():.2f}%")
        
        print(f"\n📉 PRICE DEVIATION STATISTICS:")
        print(f"   Mean deviation: {valid_data['price_deviation_from_trend'].mean():.2f}%")
        print(f"   Median deviation: {valid_data['price_deviation_from_trend'].median():.2f}%")
        print(f"   Std dev: {valid_data['price_deviation_from_trend'].std():.2f}%")
        print(f"   Min deviation: {valid_data['price_deviation_from_trend'].min():.2f}%")
        print(f"   Max deviation: {valid_data['price_deviation_from_trend'].max():.2f}%")
    
    # Save results
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    output_file = os.path.join(results_dir, "IWLS_GROWTH_RATES_ALL_ASSETS.csv")
    combined_results.to_csv(output_file, index=False)
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"   File: {output_file}")
    print(f"   Size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # Create a pivot table version for easier analysis
    print(f"\n🔄 Creating pivot table format...")
    
    # Create separate dataframes for each metric
    metrics = ['price', 'iwls_annual_growth_rate', 'iwls_trend_line_value', 'price_deviation_from_trend']
    
    for metric in metrics:
        pivot_data = combined_results.pivot(index='date', columns='asset', values=metric)
        pivot_file = os.path.join(results_dir, f"IWLS_{metric.upper()}_PIVOT.csv")
        pivot_data.to_csv(pivot_file)
        print(f"   Saved: IWLS_{metric.upper()}_PIVOT.csv")
    
    print(f"\n🎯 SUCCESS!")
    print(f"   IWLS calculations complete for all assets")
    print(f"   Use 'IWLS_GROWTH_RATES_ALL_ASSETS.csv' for comprehensive analysis")
    print(f"   Use pivot tables for time-series analysis by metric")
    
    return combined_results

def show_sample_data():
    """
    Display sample of the calculated data
    """
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    results_file = os.path.join(results_dir, "IWLS_GROWTH_RATES_ALL_ASSETS.csv")
    
    if not os.path.exists(results_file):
        print("No results file found. Run the calculation first.")
        return
    
    print("\n📋 SAMPLE OF CALCULATED DATA:")
    print("-" * 100)
    
    df = pd.read_csv(results_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Show recent data for a few assets
    recent_data = df[df['date'] >= df['date'].max() - pd.Timedelta(days=30)]
    sample_assets = recent_data['asset'].unique()[:3]
    
    for asset in sample_assets:
        asset_data = recent_data[recent_data['asset'] == asset].tail(5)
        print(f"\n{asset} (last 5 trading days):")
        for _, row in asset_data.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')}: "
                  f"Price=${row['price']:.2f}, "
                  f"Growth={row['iwls_annual_growth_rate']:.2f}%, "
                  f"Deviation={row['price_deviation_from_trend']:.2f}%")

if __name__ == "__main__":
    # Calculate IWLS for all assets
    results = calculate_iwls_for_all_assets()
    
    if results is not None:
        # Show sample of the data
        show_sample_data()
        
        print(f"\n✨ Ready for analysis!")
        print(f"   You now have IWLS growth rates and deviations for all assets")
        print(f"   Next step: Analyze correlation between deviations and future performance")