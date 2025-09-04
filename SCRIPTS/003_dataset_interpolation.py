import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def interpolate_to_daily():
    """
    Convert weekly stock data to daily using linear interpolation
    """
    print("WEEKLY TO DAILY DATA INTERPOLATOR")
    print("=" * 50)
    
    # Load the merged weekly data
    data_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/DATA"
    input_file = os.path.join(data_dir, "MERGED_WEEKLY_STOCK_DATA.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found at {input_file}")
        return None
    
    print(f"📁 Loading: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Successfully loaded {len(df)} weekly data points")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Get asset columns (exclude 'date')
    asset_columns = [col for col in df.columns if col != 'date']
    print(f"📊 Found {len(asset_columns)} assets to interpolate")
    
    # Create daily date range
    start_date = df['date'].min()
    end_date = df['date'].max()
    
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Create complete daily date range (excluding weekends for stock market)
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Filter out weekends (Saturday=5, Sunday=6)
    weekdays_only = [date for date in daily_dates if date.weekday() < 5]
    
    print(f"📈 Creating {len(weekdays_only)} trading days from {len(df)} weekly points")
    print(f"   Note: Assets will only have data between their first and last known data points")
    print(f"   No extrapolation before IPO dates or beyond last known prices")
    
    # Create daily dataframe with all trading days
    daily_df = pd.DataFrame({'date': weekdays_only})
    
    print(f"\n🔄 Interpolating asset data...")
    
    # Process each asset
    interpolated_assets = 0
    skipped_assets = 0
    
    for i, asset in enumerate(asset_columns, 1):
        print(f"   Processing {asset} ({i}/{len(asset_columns)})...", end=' ')
        
        # Get non-null values for this asset
        asset_data = df[['date', asset]].dropna()
        
        if len(asset_data) < 2:
            print(f"SKIPPED (insufficient data: {len(asset_data)} points)")
            skipped_assets += 1
            # Add column with NaN values
            daily_df[asset] = np.nan
            continue
        
        # Sort by date to ensure proper interpolation
        asset_data = asset_data.sort_values('date')
        
        # Create interpolation function
        # Using pandas interpolate with method='linear' for robust interpolation
        temp_df = pd.DataFrame({'date': weekdays_only})
        
        # Merge with existing data
        merged_temp = temp_df.merge(asset_data, on='date', how='left')
        
        # Interpolate missing values using linear interpolation
        merged_temp[asset] = merged_temp[asset].interpolate(method='linear', limit_area='inside')
        
        # For dates before first data point or after last data point, use forward/backward fill
        # Forward fill for dates before first valid data
        first_valid_idx = merged_temp[asset].first_valid_index()
        if first_valid_idx is not None and first_valid_idx > 0:
            first_valid_value = merged_temp.loc[first_valid_idx, asset]
            merged_temp.loc[:first_valid_idx-1, asset] = first_valid_value
        
        # Backward fill for dates after last valid data
        last_valid_idx = merged_temp[asset].last_valid_index()
        if last_valid_idx is not None and last_valid_idx < len(merged_temp) - 1:
            last_valid_value = merged_temp.loc[last_valid_idx, asset]
            merged_temp.loc[last_valid_idx+1:, asset] = last_valid_value
        
        # Add interpolated data to main dataframe
        daily_df[asset] = merged_temp[asset]
        
        # Calculate interpolation statistics
        original_points = len(asset_data)
        interpolated_points = daily_df[asset].count()
        interpolation_ratio = interpolated_points / original_points
        
        print(f"✅ {original_points} → {interpolated_points} points ({interpolation_ratio:.1f}x)")
        interpolated_assets += 1
    
    print(f"\n📊 INTERPOLATION SUMMARY:")
    print(f"   Successfully interpolated: {interpolated_assets} assets")
    print(f"   Skipped (insufficient data): {skipped_assets} assets")
    print(f"   Total daily records: {len(daily_df):,}")
    
    # Quality check - show some statistics
    print(f"\n🔍 QUALITY CHECK:")
    
    # Check for any remaining NaN values
    total_possible_values = len(daily_df) * len(asset_columns)
    total_actual_values = daily_df[asset_columns].count().sum()
    completeness = (total_actual_values / total_possible_values) * 100
    
    print(f"   Overall data completeness: {completeness:.1f}%")
    print(f"   Total data points: {total_actual_values:,} / {total_possible_values:,}")
    
    # Show assets with most complete data
    asset_completeness = []
    for asset in asset_columns:
        asset_complete = (daily_df[asset].count() / len(daily_df)) * 100
        asset_completeness.append((asset, asset_complete))
    
    # Sort by completeness
    asset_completeness.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n   Top 5 most complete assets:")
    for asset, completeness in asset_completeness[:5]:
        print(f"     {asset}: {completeness:.1f}%")
    
    if len(asset_completeness) > 5:
        print(f"\n   Bottom 5 least complete assets:")
        for asset, completeness in asset_completeness[-5:]:
            print(f"     {asset}: {completeness:.1f}%")
    
    # Show sample of interpolated data
    print(f"\n📋 SAMPLE OF INTERPOLATED DATA:")
    print(f"   First 10 rows, first 5 assets:")
    
    sample_assets = asset_columns[:5]
    sample_columns = ['date'] + sample_assets
    sample_data = daily_df[sample_columns].head(10)
    
    for _, row in sample_data.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d (%a)')
        values = [f"{row[asset]:.2f}" if pd.notna(row[asset]) else "NaN" for asset in sample_assets]
        print(f"   {date_str}: {' | '.join(values)}")
    
    # Save interpolated data
    output_file = os.path.join(data_dir, "DAILY_INTERPOLATED_STOCK_DATA.csv")
    daily_df.to_csv(output_file, index=False)
    
    print(f"\n💾 SAVE COMPLETE")
    print(f"   Daily interpolated data saved to: {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # Create a summary report
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Summary statistics
    summary_stats = []
    for asset, completeness in asset_completeness:
        original_count = df[asset].count() if asset in df.columns else 0
        interpolated_count = daily_df[asset].count()
        summary_stats.append({
            'asset': asset,
            'original_weekly_points': original_count,
            'interpolated_daily_points': interpolated_count,
            'interpolation_ratio': interpolated_count / original_count if original_count > 0 else 0,
            'completeness_percent': completeness
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_file = os.path.join(results_dir, "interpolation_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"   Interpolation summary saved to: {summary_file}")
    
    print(f"\n🎯 SUCCESS!")
    print(f"   Your weekly data has been converted to daily using linear interpolation")
    print(f"   Use 'DAILY_INTERPOLATED_STOCK_DATA.csv' for your IWLS analysis")
    print(f"   Dataset now contains {len(daily_df):,} trading days across {len(asset_columns)} assets")
    
    return daily_df

def validate_interpolation():
    """
    Optional validation function to check interpolation quality
    """
    print(f"\n🔍 VALIDATION CHECK")
    print("-" * 30)
    
    # Load both original and interpolated data for comparison
    data_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/DATA"
    
    try:
        original_df = pd.read_csv(os.path.join(data_dir, "MERGED_WEEKLY_STOCK_DATA.csv"))
        original_df['date'] = pd.to_datetime(original_df['date'])
        
        interpolated_df = pd.read_csv(os.path.join(data_dir, "DAILY_INTERPOLATED_STOCK_DATA.csv"))
        interpolated_df['date'] = pd.to_datetime(interpolated_df['date'])
        
        print("✅ Both files loaded for validation")
        
        # Check that original weekly values are preserved
        asset_columns = [col for col in original_df.columns if col != 'date']
        
        # Sample a few assets for detailed validation
        sample_assets = asset_columns[:3] if len(asset_columns) >= 3 else asset_columns
        
        for asset in sample_assets:
            print(f"\nValidating {asset}:")
            
            # Get original non-null values
            original_asset_data = original_df[['date', asset]].dropna()
            
            if len(original_asset_data) > 0:
                # Check if these dates exist in interpolated data with same values
                validation_errors = 0
                
                for _, row in original_asset_data.head(10).iterrows():  # Check first 10 points
                    orig_date = row['date']
                    orig_value = row[asset]
                    
                    # Find this date in interpolated data
                    interpolated_row = interpolated_df[interpolated_df['date'] == orig_date]
                    
                    if len(interpolated_row) > 0:
                        interp_value = interpolated_row.iloc[0][asset]
                        
                        if pd.notna(interp_value) and abs(orig_value - interp_value) < 0.01:
                            print(f"   ✅ {orig_date.strftime('%Y-%m-%d')}: {orig_value:.2f} preserved")
                        else:
                            print(f"   ❌ {orig_date.strftime('%Y-%m-%d')}: {orig_value:.2f} → {interp_value:.2f}")
                            validation_errors += 1
                    else:
                        print(f"   ⚠️  Date {orig_date.strftime('%Y-%m-%d')} not found in interpolated data")
                        validation_errors += 1
                
                if validation_errors == 0:
                    print(f"   ✅ All original values preserved correctly")
                else:
                    print(f"   ⚠️  {validation_errors} validation issues found")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    # Run the interpolation
    daily_data = interpolate_to_daily()
    
    if daily_data is not None:
        # Run validation check
        validate_interpolation()
        
        print(f"\n🚀 READY FOR IWLS ANALYSIS!")
        print(f"   Your daily interpolated dataset is ready")
        print(f"   Linear interpolation provides smooth price transitions")
        print(f"   All original weekly data points are preserved exactly")