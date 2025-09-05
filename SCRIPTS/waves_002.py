import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def enhance_iwls_dataset_with_wave_analysis():
    """
    Enhance existing IWLS dataset with rolling range wave analysis metrics
    """
    print("ENHANCING IWLS DATASET WITH WAVE ANALYSIS")
    print("=" * 50)
    
    # Load the existing IWLS dataset with forward performance
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    input_file = os.path.join(results_dir, "IWLS_WITH_FORWARD_PERFORMANCE.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: IWLS dataset not found at {input_file}")
        print("   Please run the IWLS forward performance script first.")
        return None
    
    print(f"📁 Loading existing IWLS dataset: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Successfully loaded {len(df):,} IWLS records")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Show current dataset info
    assets = df['asset'].unique()
    print(f"📊 Current dataset info:")
    print(f"   Assets: {len(assets)}")
    print(f"   Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Existing columns: {len(df.columns)}")
    
    # Check if we have forward performance data
    forward_cols = [col for col in df.columns if 'forward_return' in col or 'max_gain' in col]
    print(f"   Forward performance metrics: {len(forward_cols)}")
    if forward_cols:
        print(f"     {', '.join(forward_cols[:3])}{'...' if len(forward_cols) > 3 else ''}")
    
    # Define rolling windows (trading days)
    windows = {
        '3_month': 63,   # ~3 months of trading days
        '6_month': 126   # ~6 months of trading days
    }
    
    print(f"\n🔄 Adding wave analysis metrics...")
    print(f"   3-month window: {windows['3_month']} trading days")
    print(f"   6-month window: {windows['6_month']} trading days")
    
    # Sort data for rolling calculations
    df_sorted = df.sort_values(['asset', 'date']).reset_index(drop=True)
    
    # Initialize new columns
    new_columns = [
        'rolling_range_pct_3_month',
        'rolling_range_pct_6_month', 
        'position_pct_3_month',
        'position_pct_6_month',
        'rolling_low_3_month',
        'rolling_high_3_month',
        'rolling_low_6_month',
        'rolling_high_6_month'
    ]
    
    for col in new_columns:
        df_sorted[col] = np.nan
    
    # Process each asset separately
    processed_assets = 0
    total_assets = len(assets)
    
    for asset in assets:
        processed_assets += 1
        print(f"\n📈 Processing asset {processed_assets}/{total_assets}: {asset}")
        
        # Get asset data
        asset_mask = df_sorted['asset'] == asset
        asset_data = df_sorted[asset_mask].copy()
        
        if len(asset_data) < windows['6_month']:
            print(f"  ⚠️  Skipping {asset}: insufficient data ({len(asset_data)} days)")
            continue
        
        print(f"  📊 Processing {len(asset_data):,} data points...")
        
        # Calculate rolling metrics for each window
        for window_name, window_days in windows.items():
            print(f"    🔍 Calculating {window_name} metrics...")
            
            # Rolling min and max prices
            rolling_low = asset_data['price'].rolling(
                window=window_days, 
                min_periods=window_days//2
            ).min()
            
            rolling_high = asset_data['price'].rolling(
                window=window_days, 
                min_periods=window_days//2
            ).max()
            
            # Rolling range as percentage (from low to high)
            rolling_range_pct = ((rolling_high - rolling_low) / rolling_low) * 100
            
            # Current position within the range (0% = at low, 100% = at high)
            range_span = rolling_high - rolling_low
            position_pct = np.where(
                range_span > 0,
                ((asset_data['price'] - rolling_low) / range_span) * 100,
                50.0  # If no range (high = low), set to middle
            )
            
            # Update the main dataframe
            df_sorted.loc[asset_mask, f'rolling_low_{window_name}'] = rolling_low.values
            df_sorted.loc[asset_mask, f'rolling_high_{window_name}'] = rolling_high.values
            df_sorted.loc[asset_mask, f'rolling_range_pct_{window_name}'] = rolling_range_pct.values
            df_sorted.loc[asset_mask, f'position_pct_{window_name}'] = position_pct
        
        # Show sample statistics for this asset
        valid_3m = rolling_range_pct.dropna()
        valid_6m_range = df_sorted.loc[asset_mask, 'rolling_range_pct_6_month'].dropna()
        
        if len(valid_3m) > 0 and len(valid_6m_range) > 0:
            print(f"  📊 Sample stats:")
            print(f"    3m range: avg={valid_3m.mean():.1f}%, max={valid_3m.max():.1f}%")
            print(f"    6m range: avg={valid_6m_range.mean():.1f}%, max={valid_6m_range.max():.1f}%")
    
    # Summary statistics
    print(f"\n📊 ENHANCED DATASET SUMMARY:")
    print(f"   Total records: {len(df_sorted):,}")
    print(f"   Assets processed: {processed_assets}/{total_assets}")
    print(f"   New columns added: {len(new_columns)}")
    print(f"   Total columns now: {len(df_sorted.columns)}")
    
    # Wave analysis statistics
    valid_data = df_sorted.dropna(subset=['rolling_range_pct_3_month', 'rolling_range_pct_6_month'])
    
    if len(valid_data) > 0:
        print(f"\n📈 WAVE ANALYSIS STATISTICS:")
        print(f"   Valid wave records: {len(valid_data):,}")
        
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
        print(f"   6-month positions:")
        print(f"     Average: {valid_data['position_pct_6_month'].mean():.1f}%")
        print(f"     Bottom quartile (<25%): {(valid_data['position_pct_6_month'] < 25).sum():,} records")
        print(f"     Top quartile (>75%): {(valid_data['position_pct_6_month'] > 75).sum():,} records")
    
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
    
    # Save enhanced dataset
    output_file = os.path.join(results_dir, "IWLS_WITH_WAVE_ANALYSIS.csv")
    df_sorted.to_csv(output_file, index=False)
    
    print(f"\n💾 ENHANCED DATASET SAVED:")
    print(f"   File: {output_file}")
    print(f"   Size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # Save asset volatility summary
    summary_file = os.path.join(results_dir, "ASSET_VOLATILITY_SUMMARY.csv")
    asset_avg_ranges.to_csv(summary_file)
    print(f"   Volatility summary: {summary_file}")
    
    print(f"\n✨ DATASET ENHANCEMENT COMPLETE!")
    print(f"🎯 Your enhanced dataset now includes:")
    print(f"   📊 Original IWLS deviation analysis")
    print(f"   📈 Forward performance metrics (6m, 12m, 18m)")
    print(f"   🌊 Wave analysis (rolling ranges & positions)")
    print(f"   🔍 Ready for comprehensive wave physics backtesting!")
    
    return df_sorted

def show_enhanced_dataset_sample():
    """
    Display sample of the enhanced dataset showing all metrics together
    """
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    results_file = os.path.join(results_dir, "IWLS_WITH_WAVE_ANALYSIS.csv")
    
    if not os.path.exists(results_file):
        print("No enhanced dataset found. Run the enhancement first.")
        return
    
    print(f"\n📋 ENHANCED DATASET SAMPLE:")
    print("=" * 100)
    
    df = pd.read_csv(results_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Show recent data for one asset with all metrics
    recent_data = df[df['date'] >= df['date'].max() - pd.Timedelta(days=60)]
    sample_asset = recent_data['asset'].iloc[0]
    
    asset_data = recent_data[recent_data['asset'] == sample_asset].tail(5)
    
    print(f"Sample: {sample_asset} (last 5 records with complete metrics)")
    print()
    print(f"{'Date':>12} | {'Price':>8} | {'IWLS Dev':>8} | {'6m Range':>9} | {'6m Pos':>7} | {'12m Gain':>9}")
    print("-" * 80)
    
    for _, row in asset_data.iterrows():
        iwls_dev = row.get('price_deviation_from_trend', np.nan)
        range_6m = row.get('rolling_range_pct_6_month', np.nan)
        pos_6m = row.get('position_pct_6_month', np.nan)
        gain_12m = row.get('max_gain_12_month', np.nan)
        
        iwls_str = f"{iwls_dev:.1f}%" if not pd.isna(iwls_dev) else "N/A"
        range_str = f"{range_6m:.1f}%" if not pd.isna(range_6m) else "N/A"
        pos_str = f"{pos_6m:.1f}%" if not pd.isna(pos_6m) else "N/A"
        gain_str = f"{gain_12m:.1f}%" if not pd.isna(gain_12m) else "N/A"
        
        print(f"{row['date'].strftime('%Y-%m-%d'):>12} | "
              f"${row['price']:>7.2f} | "
              f"{iwls_str:>7} | "
              f"{range_str:>8} | "
              f"{pos_str:>6} | "
              f"{gain_str:>8}")
    
    print()
    print("Key:")
    print("  IWLS Dev = IWLS deviation from trend")
    print("  6m Range = 6-month rolling range percentage") 
    print("  6m Pos = Position within 6-month range")
    print("  12m Gain = Maximum gain achieved in following 12 months")

def analyze_wave_physics_correlation():
    """
    Quick analysis of correlations between wave metrics and forward performance
    """
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    results_file = os.path.join(results_dir, "IWLS_WITH_WAVE_ANALYSIS.csv")
    
    if not os.path.exists(results_file):
        print("No enhanced dataset found. Run the enhancement first.")
        return
    
    print(f"\n🔄 WAVE PHYSICS CORRELATION ANALYSIS:")
    print("-" * 60)
    
    df = pd.read_csv(results_file)
    
    # Get complete records with all metrics
    complete_data = df.dropna(subset=[
        'rolling_range_pct_6_month', 'position_pct_6_month', 
        'max_gain_12_month', 'forward_return_12_month'
    ])
    
    if len(complete_data) == 0:
        print("No complete records found for correlation analysis.")
        return
    
    print(f"Complete records for analysis: {len(complete_data):,}")
    
    # Correlation analysis
    correlations = {
        '6m Range vs 12m Max Gain': complete_data['rolling_range_pct_6_month'].corr(complete_data['max_gain_12_month']),
        '6m Position vs 12m Max Gain': complete_data['position_pct_6_month'].corr(complete_data['max_gain_12_month']),
        '6m Range vs 12m Return': complete_data['rolling_range_pct_6_month'].corr(complete_data['forward_return_12_month']),
        '6m Position vs 12m Return': complete_data['position_pct_6_month'].corr(complete_data['forward_return_12_month'])
    }
    
    print("Key correlations:")
    for desc, corr in correlations.items():
        direction = "📈 Positive" if corr > 0 else "📉 Negative" if corr < 0 else "➡️  Neutral"
        strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.1 else "Weak"
        print(f"  {desc:<25}: {corr:>6.3f} ({direction}, {strength})")
    
    # High volatility performance
    high_vol_threshold = complete_data['rolling_range_pct_6_month'].quantile(0.75)
    high_vol_data = complete_data[complete_data['rolling_range_pct_6_month'] > high_vol_threshold]
    
    print(f"\nHigh volatility analysis (>75th percentile: {high_vol_threshold:.1f}%):")
    print(f"  Records: {len(high_vol_data):,}")
    print(f"  Avg 12m max gain: {high_vol_data['max_gain_12_month'].mean():.1f}%")
    print(f"  Avg 12m return: {high_vol_data['forward_return_12_month'].mean():.1f}%")
    
    # Bottom quartile position analysis
    bottom_position_data = high_vol_data[high_vol_data['position_pct_6_month'] < 25]
    
    if len(bottom_position_data) > 0:
        print(f"\nHigh volatility + bottom position (<25%) analysis:")
        print(f"  Records: {len(bottom_position_data):,}")
        print(f"  Avg 12m max gain: {bottom_position_data['max_gain_12_month'].mean():.1f}%")
        print(f"  Avg 12m return: {bottom_position_data['forward_return_12_month'].mean():.1f}%")

if __name__ == "__main__":
    # Enhance the existing IWLS dataset with wave analysis
    enhanced_data = enhance_iwls_dataset_with_wave_analysis()
    
    if enhanced_data is not None:
        # Show sample of enhanced dataset
        show_enhanced_dataset_sample()
        
        # Quick correlation analysis
        analyze_wave_physics_correlation()
        
        print(f"\n🚀 READY FOR WAVE PHYSICS BACKTESTING!")
        print(f"   🎯 Test theory: High volatility + low position → exceptional returns")
        print(f"   🌊 All metrics now in single comprehensive dataset")
        print(f"   📊 IWLS + Wave + Forward Performance = Complete picture")