import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def prepare_spy_market_data():
    """
    Process SPY weekly data to create daily interpolated values with market regime indicators
    """
    print("SPY MARKET REGIME DATA PREPARATION")
    print("=" * 50)
    
    # File paths
    data_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/DATA"
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    spy_file = os.path.join(data_dir, "BATS_SPY, 1W-3.csv")
    
    if not os.path.exists(spy_file):
        print(f"❌ Error: SPY file not found at {spy_file}")
        return None
    
    print(f"📁 Loading SPY data: {spy_file}")
    
    try:
        # Load SPY data
        spy_df = pd.read_csv(spy_file)
        print(f"✅ Successfully loaded {len(spy_df):,} weekly records")
        
        # Convert timestamp to datetime
        spy_df['date'] = pd.to_datetime(spy_df['time'], unit='s')
        spy_df = spy_df.sort_values('date')
        
        print(f"📅 Data range: {spy_df['date'].min().strftime('%Y-%m-%d')} to {spy_df['date'].max().strftime('%Y-%m-%d')}")
        
    except Exception as e:
        print(f"❌ Error loading SPY data: {e}")
        return None
    
    # 1. CREATE DAILY DATE RANGE
    print(f"\n📈 INTERPOLATING TO DAILY DATA...")
    
    start_date = spy_df['date'].min()
    end_date = spy_df['date'].max()
    
    # Create daily date range
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_df = pd.DataFrame({'date': daily_dates})
    
    print(f"   Weekly records: {len(spy_df):,}")
    print(f"   Daily records to create: {len(daily_df):,}")
    
    # 2. INTERPOLATE PRICE AND MA DATA
    print(f"\n🔧 INTERPOLATING PRICE AND MOVING AVERAGE...")
    
    # Merge and interpolate
    spy_df_subset = spy_df[['date', 'close', 'MA']].copy()
    merged_df = pd.merge(daily_df, spy_df_subset, on='date', how='left')
    
    # Interpolate missing values
    merged_df['close_interpolated'] = merged_df['close'].interpolate(method='linear')
    merged_df['ma_interpolated'] = merged_df['MA'].interpolate(method='linear')
    
    # Remove rows where we don't have MA data (beginning of series)
    daily_spy = merged_df.dropna(subset=['ma_interpolated']).copy()
    
    print(f"   ✅ Daily records with interpolated data: {len(daily_spy):,}")
    print(f"   📅 Daily range: {daily_spy['date'].min().strftime('%Y-%m-%d')} to {daily_spy['date'].max().strftime('%Y-%m-%d')}")
    
    # 3. CALCULATE PRICE-TO-MA RATIO
    print(f"\n📊 CALCULATING PRICE-TO-MA RATIO...")
    
    daily_spy['price_to_ma_ratio'] = daily_spy['close_interpolated'] / daily_spy['ma_interpolated']
    daily_spy['price_to_ma_percentage'] = (daily_spy['price_to_ma_ratio'] - 1) * 100
    
    # Show ratio statistics
    ratio_stats = daily_spy['price_to_ma_percentage'].describe()
    print(f"   Price-to-MA Percentage Statistics:")
    print(f"     Mean: {ratio_stats['mean']:>8.2f}%")
    print(f"     Std:  {ratio_stats['std']:>8.2f}%")
    print(f"     Min:  {ratio_stats['min']:>8.2f}%")
    print(f"     Max:  {ratio_stats['max']:>8.2f}%")
    
    # 4. CALCULATE ROLLING STANDARD DEVIATION AND Z-SCORES
    print(f"\n📏 CALCULATING Z-SCORES...")
    
    # Use different rolling windows to test sensitivity
    rolling_windows = [30, 60, 120, 252]  # ~1 month, 2 months, 4 months, 1 year
    
    for window in rolling_windows:
        # Rolling mean and std of price-to-MA ratio
        daily_spy[f'ratio_rolling_mean_{window}d'] = daily_spy['price_to_ma_ratio'].rolling(window=window, min_periods=window//2).mean()
        daily_spy[f'ratio_rolling_std_{window}d'] = daily_spy['price_to_ma_ratio'].rolling(window=window, min_periods=window//2).std()
        
        # Z-score calculation
        daily_spy[f'market_z_score_{window}d'] = (
            daily_spy['price_to_ma_ratio'] - daily_spy[f'ratio_rolling_mean_{window}d']
        ) / daily_spy[f'ratio_rolling_std_{window}d']
        
        # Show z-score statistics for this window
        z_scores = daily_spy[f'market_z_score_{window}d'].dropna()
        if len(z_scores) > 0:
            print(f"   {window:>3}d Z-Score: Mean={z_scores.mean():>6.2f}, Std={z_scores.std():>6.2f}, Min={z_scores.min():>6.2f}, Max={z_scores.max():>6.2f}")
    
    # 5. IDENTIFY EXTREME PERIODS FOR REFERENCE
    print(f"\n🚨 EXTREME MARKET PERIODS (using 60d Z-score):")
    
    # Find most extreme periods
    z_col = 'market_z_score_60d'
    extreme_threshold = 2.0
    
    extreme_high = daily_spy[daily_spy[z_col] > extreme_threshold].copy()
    extreme_low = daily_spy[daily_spy[z_col] < -extreme_threshold].copy()
    
    if len(extreme_high) > 0:
        print(f"   📈 Periods with Z > +{extreme_threshold}: {len(extreme_high):,} days")
        top_highs = extreme_high.nlargest(5, z_col)[['date', 'price_to_ma_percentage', z_col]]
        for _, row in top_highs.iterrows():
            print(f"     {row['date'].strftime('%Y-%m-%d')}: {row['price_to_ma_percentage']:>6.1f}% above MA (Z={row[z_col]:>5.2f})")
    
    if len(extreme_low) > 0:
        print(f"   📉 Periods with Z < -{extreme_threshold}: {len(extreme_low):,} days")
        top_lows = extreme_low.nsmallest(5, z_col)[['date', 'price_to_ma_percentage', z_col]]
        for _, row in top_lows.iterrows():
            print(f"     {row['date'].strftime('%Y-%m-%d')}: {row['price_to_ma_percentage']:>6.1f}% below MA (Z={row[z_col]:>5.2f})")
    
    # 6. SAVE RESULTS
    print(f"\n💾 SAVING RESULTS...")
    
    # Clean up the dataframe for saving
    output_columns = [
        'date', 'close_interpolated', 'ma_interpolated', 
        'price_to_ma_ratio', 'price_to_ma_percentage'
    ]
    
    # Add all the z-score columns
    z_score_columns = [col for col in daily_spy.columns if col.startswith('market_z_score_')]
    rolling_columns = [col for col in daily_spy.columns if col.startswith('ratio_rolling_')]
    
    output_columns.extend(rolling_columns)
    output_columns.extend(z_score_columns)
    
    output_df = daily_spy[output_columns].copy()
    
    # Rename columns for clarity
    output_df = output_df.rename(columns={
        'close_interpolated': 'spy_price',
        'ma_interpolated': 'spy_8week_ma',
        'price_to_ma_ratio': 'price_ma_ratio',
        'price_to_ma_percentage': 'price_ma_pct'
    })
    
    # Save the processed data
    output_file = os.path.join(results_dir, "SPY_DAILY_MARKET_REGIME_DATA.csv")
    output_df.to_csv(output_file, index=False)
    
    print(f"   ✅ Market regime data saved: {output_file}")
    print(f"   📊 Columns included: {len(output_columns)} columns")
    print(f"   📅 Daily records: {len(output_df):,}")
    
    # Show column summary
    print(f"\n📋 OUTPUT COLUMNS:")
    print(f"   📈 Price data: spy_price, spy_8week_ma, price_ma_ratio, price_ma_pct")
    print(f"   📏 Z-scores: market_z_score_30d, 60d, 120d, 252d")
    print(f"   📊 Rolling stats: ratio_rolling_mean/std for each window")
    
    print(f"\n✨ PREPARATION COMPLETE!")
    print(f"   🎯 Ready for market regime classification")
    print(f"   📈 You can now choose Z-score thresholds for:")
    print(f"      • Normal market conditions")
    print(f"      • Market spikes/euphoria") 
    print(f"      • Market crashes/panic")
    print(f"      • Trending up/down within normal bounds")
    
    return output_df

if __name__ == "__main__":
    # Process SPY data for market regime analysis
    spy_data = prepare_spy_market_data()
    
    if spy_data is not None:
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Review the Z-score distributions to choose classification thresholds")
        print(f"   2. Create market regime classifications (normal, spike, crash, etc.)")
        print(f"   3. Merge with your stock deviation analysis")
        print(f"   4. Analyze how stock performance varies by market regime")