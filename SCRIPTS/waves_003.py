import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_rolling_range_deciles():
    """
    Analyze forward performance by rolling range deciles
    """
    print("ROLLING RANGE DECILE ANALYSIS")
    print("=" * 50)
    
    # Load the enhanced dataset
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    input_file = os.path.join(results_dir, "IWLS_WITH_WAVE_ANALYSIS.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Enhanced dataset not found at {input_file}")
        print("   Please run the wave analysis enhancement script first.")
        return None
    
    print(f"📁 Loading enhanced dataset: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Successfully loaded {len(df):,} records")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Filter to complete records with all required metrics
    required_columns = [
        'rolling_range_pct_6_month', 'position_pct_6_month',
        'forward_return_12_month', 'max_gain_12_month', 'max_drawdown_12_month'
    ]
    
    complete_data = df.dropna(subset=required_columns).copy()
    print(f"📊 Complete records for analysis: {len(complete_data):,}")
    
    if len(complete_data) == 0:
        print("❌ No complete records found for analysis.")
        return None
    
    # Create rolling range deciles
    print(f"\n🔍 Creating rolling range deciles...")
    
    # Check for zero or duplicate values
    range_values = complete_data['rolling_range_pct_6_month']
    zero_count = (range_values == 0).sum()
    unique_values = range_values.nunique()
    
    print(f"   Zero range values: {zero_count:,}")
    print(f"   Unique range values: {unique_values:,}")
    print(f"   Range statistics: min={range_values.min():.2f}%, max={range_values.max():.2f}%")
    
    # Filter out zero or very small ranges (less than 0.1%)
    print(f"   Filtering out ranges < 0.1% to improve decile quality...")
    filtered_data = complete_data[complete_data['rolling_range_pct_6_month'] >= 0.1].copy()
    print(f"   Records after filtering: {len(filtered_data):,}")
    
    if len(filtered_data) < 1000:
        print(f"❌ Insufficient data after filtering for decile analysis")
        return None, None
    
    # Create deciles with duplicate handling
    try:
        filtered_data['range_decile'] = pd.qcut(
            filtered_data['rolling_range_pct_6_month'], 
            q=10, 
            labels=['D1 (Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (High)'],
            precision=1,
            duplicates='drop'  # Handle duplicate bin edges
        )
        complete_data = filtered_data
        print(f"✅ Successfully created 10 deciles")
    except ValueError as e:
        print(f"⚠️  Still having issues with deciles, trying quintiles...")
        try:
            filtered_data['range_decile'] = pd.qcut(
                filtered_data['rolling_range_pct_6_month'], 
                q=5, 
                labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'],
                precision=1,
                duplicates='drop'
            )
            complete_data = filtered_data
            print(f"✅ Successfully created 5 quintiles")
        except ValueError as e2:
            print(f"❌ Cannot create bins: {e2}")
            return None, None
    
    # Show decile boundaries
    decile_boundaries = complete_data.groupby('range_decile')['rolling_range_pct_6_month'].agg(['min', 'max', 'mean', 'count'])
    
    print(f"\n📋 ROLLING RANGE DECILE/QUINTILE BOUNDARIES:")
    print(f"{'Bin':>12} | {'Min':>8} | {'Max':>8} | {'Mean':>8} | {'Count':>8}")
    print("-" * 60)
    
    for bin_name in decile_boundaries.index:
        min_val = decile_boundaries.loc[bin_name, 'min']
        max_val = decile_boundaries.loc[bin_name, 'max']
        mean_val = decile_boundaries.loc[bin_name, 'mean']
        count_val = int(decile_boundaries.loc[bin_name, 'count'])
        
        print(f"{bin_name:>12} | {min_val:>7.1f}% | {max_val:>7.1f}% | {mean_val:>7.1f}% | {count_val:>8,}")
    
    # Analyze forward performance by decile
    print(f"\n📈 FORWARD PERFORMANCE BY ROLLING RANGE BIN:")
    print("-" * 90)
    print(f"{'Bin':>12} | {'Count':>8} | {'Avg Return':>11} | {'Avg Max Gain':>12} | {'Avg Max DD':>11}")
    print("-" * 90)
    
    decile_results = []
    
    for bin_name in decile_boundaries.index:
        bin_data = complete_data[complete_data['range_decile'] == bin_name]
        
        if len(bin_data) > 0:
            avg_return = bin_data['forward_return_12_month'].mean()
            avg_max_gain = bin_data['max_gain_12_month'].mean()
            avg_max_dd = bin_data['max_drawdown_12_month'].mean()
            count = len(bin_data)
            
            print(f"{bin_name:>12} | {count:>8,} | {avg_return:>10.1f}% | {avg_max_gain:>11.1f}% | {avg_max_dd:>10.1f}%")
            
            decile_results.append({
                'bin': bin_name,
                'count': count,
                'avg_forward_return': avg_return,
                'avg_max_gain': avg_max_gain,
                'avg_max_drawdown': avg_max_dd,
                'range_min': decile_boundaries.loc[bin_name, 'min'],
                'range_max': decile_boundaries.loc[bin_name, 'max'],
                'range_mean': decile_boundaries.loc[bin_name, 'mean']
            })
    
    return pd.DataFrame(decile_results), complete_data

def analyze_quintile_position_combinations():
    """
    Analyze performance by rolling range quintiles crossed with position percentages
    """
    print(f"\n" + "="*80)
    print("QUINTILE × POSITION ANALYSIS")
    print("="*80)
    
    # Load the enhanced dataset
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    input_file = os.path.join(results_dir, "IWLS_WITH_WAVE_ANALYSIS.csv")
    
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to complete records
    required_columns = [
        'rolling_range_pct_6_month', 'position_pct_6_month',
        'forward_return_12_month', 'max_gain_12_month', 'max_drawdown_12_month'
    ]
    
    complete_data = df.dropna(subset=required_columns).copy()
    
    # Filter out tiny ranges like above
    complete_data = complete_data[complete_data['rolling_range_pct_6_month'] >= 0.1].copy()
    print(f"📊 Complete records for quintile analysis: {len(complete_data):,}")
    
    # Create rolling range quintiles
    try:
        complete_data['range_quintile'] = pd.qcut(
            complete_data['rolling_range_pct_6_month'], 
            q=5, 
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'],
            precision=1,
            duplicates='drop'
        )
    except ValueError:
        print("⚠️  Using percentile-based quintiles")
        complete_data['range_quintile'] = pd.cut(
            complete_data['rolling_range_pct_6_month'].rank(pct=True),
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'],
            include_lowest=True
        )
    
    # Create position thirds
    try:
        complete_data['position_third'] = pd.qcut(
            complete_data['position_pct_6_month'], 
            q=3, 
            labels=['Bottom Third', 'Middle Third', 'Top Third'],
            precision=1,
            duplicates='drop'
        )
    except ValueError:
        print("⚠️  Using percentile-based position thirds")
        complete_data['position_third'] = pd.cut(
            complete_data['position_pct_6_month'].rank(pct=True),
            bins=[0, 0.333, 0.667, 1.0],
            labels=['Bottom Third', 'Middle Third', 'Top Third'],
            include_lowest=True
        )
    
    print(f"\n🎯 QUINTILE × POSITION COMBINATION ANALYSIS:")
    print("Forward 12-month performance by rolling range quintile and position within range")
    print()
    
    # Create combination analysis
    combo_results = []
    
    print(f"{'Range Quintile':>15} | {'Position':>12} | {'Count':>8} | {'Avg Return':>11} | {'Avg Max Gain':>12}")
    print("-" * 75)
    
    for quintile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']:
        quintile_data = complete_data[complete_data['range_quintile'] == quintile]
        
        if len(quintile_data) == 0:
            continue
            
        for position in ['Bottom Third', 'Middle Third', 'Top Third']:
            combo_data = quintile_data[quintile_data['position_third'] == position]
            
            if len(combo_data) > 0:
                avg_return = combo_data['forward_return_12_month'].mean()
                avg_max_gain = combo_data['max_gain_12_month'].mean()
                count = len(combo_data)
                
                print(f"{quintile:>15} | {position:>12} | {count:>8,} | {avg_return:>10.1f}% | {avg_max_gain:>11.1f}%")
                
                combo_results.append({
                    'range_quintile': quintile,
                    'position_third': position,
                    'count': count,
                    'avg_forward_return': avg_return,
                    'avg_max_gain': avg_max_gain,
                    'avg_max_drawdown': combo_data['max_drawdown_12_month'].mean()
                })
            else:
                print(f"{quintile:>15} | {position:>12} | {'0':>8} | {'N/A':>10} | {'N/A':>11}")
    
    # Highlight best combinations
    if combo_results:
        combo_df = pd.DataFrame(combo_results)
        best_return = combo_df.loc[combo_df['avg_forward_return'].idxmax()]
        best_gain = combo_df.loc[combo_df['avg_max_gain'].idxmax()]
        
        print(f"\n🏆 BEST COMBINATIONS:")
        print(f"   Highest avg return: {best_return['range_quintile']} + {best_return['position_third']} = {best_return['avg_forward_return']:.1f}%")
        print(f"   Highest avg max gain: {best_gain['range_quintile']} + {best_gain['position_third']} = {best_gain['avg_max_gain']:.1f}%")
    
    return pd.DataFrame(combo_results)

def analyze_volatility_persistence():
    """
    Analyze whether high volatility periods persist for individual assets
    """
    print(f"\n" + "="*80)
    print("VOLATILITY PERSISTENCE ANALYSIS")
    print("="*80)
    print("Question: Do assets tend to stay in high/low volatility quartiles consistently?")
    print()
    
    # Load the enhanced dataset
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    input_file = os.path.join(results_dir, "IWLS_WITH_WAVE_ANALYSIS.csv")
    
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to records with rolling range data
    range_data = df.dropna(subset=['rolling_range_pct_6_month']).copy()
    range_data = range_data[range_data['rolling_range_pct_6_month'] >= 0.1].copy()  # Filter tiny ranges
    print(f"📊 Records with rolling range data: {len(range_data):,}")
    
    # Create 6-month periods for analysis
    range_data['period'] = range_data['date'].dt.to_period('6M')
    
    print(f"🕐 Analyzing persistence across 6-month periods...")
    
    # Calculate market-adjusted rolling range to remove market noise
    print(f"📊 Calculating market-adjusted rolling ranges...")
    
    # Calculate market average for each date
    market_averages = range_data.groupby('date')['rolling_range_pct_6_month'].mean().reset_index()
    market_averages.columns = ['date', 'market_avg_range']
    
    # Merge back to get market-adjusted ranges
    range_data = range_data.merge(market_averages, on='date', how='left')
    range_data['market_adjusted_range'] = range_data['rolling_range_pct_6_month'] - range_data['market_avg_range']
    
    print(f"✅ Market noise adjustment complete")
    
    # For each period, calculate average market-adjusted range per asset
    period_asset_ranges = range_data.groupby(['period', 'asset'])['market_adjusted_range'].mean().reset_index()
    period_asset_ranges.columns = ['period', 'asset', 'avg_adjusted_range']
    
    # For each period, assign quartiles
    def assign_quartiles(group):
        try:
            group['volatility_quartile'] = pd.qcut(
                group['avg_adjusted_range'], 
                q=4, 
                labels=['Q1 (Bottom 25%)', 'Q2', 'Q3', 'Q4 (Top 25%)'],
                duplicates='drop'  # Handle duplicate bin edges
            )
        except ValueError:
            # If quartiles fail, use simple ranking approach
            group['volatility_quartile'] = pd.cut(
                group['avg_adjusted_range'].rank(pct=True), 
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['Q1 (Bottom 25%)', 'Q2', 'Q3', 'Q4 (Top 25%)'],
                include_lowest=True
            )
        return group
    
    period_quartiles = period_asset_ranges.groupby('period').apply(assign_quartiles, include_groups=False).reset_index()
    
    print(f"📈 Analyzing persistence patterns...")
    
    # Analyze persistence for each asset
    assets = period_quartiles['asset'].unique()
    persistence_results = []
    
    for asset in assets:
        asset_periods = period_quartiles[period_quartiles['asset'] == asset].sort_values('period')
        
        if len(asset_periods) < 4:  # Need at least 4 periods for meaningful analysis
            continue
        
        # Count quartile distributions
        quartile_counts = asset_periods['volatility_quartile'].value_counts()
        total_periods = len(asset_periods)
        
        # Calculate persistence metrics
        top_quartile_pct = (quartile_counts.get('Q4 (Top 25%)', 0) / total_periods) * 100
        bottom_quartile_pct = (quartile_counts.get('Q1 (Bottom 25%)', 0) / total_periods) * 100
        
        # Check for consecutive periods in same quartile
        quartiles = asset_periods['volatility_quartile'].tolist()
        max_consecutive_top = 0
        max_consecutive_bottom = 0
        current_consecutive_top = 0
        current_consecutive_bottom = 0
        
        for q in quartiles:
            if pd.isna(q):
                continue
            elif q == 'Q4 (Top 25%)':
                current_consecutive_top += 1
                max_consecutive_top = max(max_consecutive_top, current_consecutive_top)
                current_consecutive_bottom = 0
            elif q == 'Q1 (Bottom 25%)':
                current_consecutive_bottom += 1
                max_consecutive_bottom = max(max_consecutive_bottom, current_consecutive_bottom)
                current_consecutive_top = 0
            else:
                current_consecutive_top = 0
                current_consecutive_bottom = 0
        
        persistence_results.append({
            'asset': asset,
            'total_periods': total_periods,
            'top_quartile_pct': top_quartile_pct,
            'bottom_quartile_pct': bottom_quartile_pct,
            'max_consecutive_top': max_consecutive_top,
            'max_consecutive_bottom': max_consecutive_bottom,
            'volatility_pattern': 'High Persistence' if top_quartile_pct > 50 or bottom_quartile_pct > 50 else 'Mixed Pattern'
        })
    
    persistence_df = pd.DataFrame(persistence_results)
    
    # Summary statistics
    print(f"\n📊 PERSISTENCE SUMMARY:")
    print(f"   Assets analyzed: {len(persistence_df)}")
    
    high_vol_persistent = persistence_df[persistence_df['top_quartile_pct'] > 50]
    low_vol_persistent = persistence_df[persistence_df['bottom_quartile_pct'] > 50]
    mixed_pattern = persistence_df[
        (persistence_df['top_quartile_pct'] <= 50) & 
        (persistence_df['bottom_quartile_pct'] <= 50)
    ]
    
    print(f"   High volatility persistent (>50% in top quartile): {len(high_vol_persistent)}")
    print(f"   Low volatility persistent (>50% in bottom quartile): {len(low_vol_persistent)}")
    print(f"   Mixed pattern assets: {len(mixed_pattern)}")
    
    # Show most persistent high volatility assets
    if len(high_vol_persistent) > 0:
        print(f"\n🌊 MOST PERSISTENTLY HIGH VOLATILITY ASSETS:")
        top_persistent = high_vol_persistent.nlargest(10, 'top_quartile_pct')
        
        print(f"{'Asset':>6} | {'Periods':>7} | {'Top Q %':>8} | {'Max Consec':>10} | {'Pattern':>15}")
        print("-" * 60)
        
        for _, row in top_persistent.iterrows():
            print(f"{row['asset']:>6} | {row['total_periods']:>7} | {row['top_quartile_pct']:>7.1f}% | {row['max_consecutive_top']:>10} | {row['volatility_pattern']:>15}")
    
    # Show most stable (low volatility) assets
    if len(low_vol_persistent) > 0:
        print(f"\n📊 MOST CONSISTENTLY LOW VOLATILITY ASSETS:")
        top_stable = low_vol_persistent.nlargest(10, 'bottom_quartile_pct')
        
        print(f"{'Asset':>6} | {'Periods':>7} | {'Bottom Q %':>10} | {'Max Consec':>10} | {'Pattern':>15}")
        print("-" * 65)
        
        for _, row in top_stable.iterrows():
            print(f"{row['asset']:>6} | {row['total_periods']:>7} | {row['bottom_quartile_pct']:>9.1f}% | {row['max_consecutive_bottom']:>10} | {row['volatility_pattern']:>15}")
    
    return persistence_df

def save_all_analysis_results(decile_results, combo_results, persistence_results):
    """
    Save all analysis results to CSV files
    """
    print(f"\n💾 SAVING ANALYSIS RESULTS...")
    
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    
    # Save decile analysis
    if decile_results is not None and len(decile_results) > 0:
        decile_file = os.path.join(results_dir, "ROLLING_RANGE_DECILE_ANALYSIS.csv")
        decile_results.to_csv(decile_file, index=False)
        print(f"✅ Decile analysis: {decile_file}")
    
    # Save combination analysis
    if combo_results is not None and len(combo_results) > 0:
        combo_file = os.path.join(results_dir, "QUINTILE_POSITION_COMBINATION_ANALYSIS.csv")
        combo_results.to_csv(combo_file, index=False)
        print(f"✅ Combination analysis: {combo_file}")
    
    # Save persistence analysis
    if persistence_results is not None and len(persistence_results) > 0:
        persistence_file = os.path.join(results_dir, "VOLATILITY_PERSISTENCE_ANALYSIS.csv")
        persistence_results.to_csv(persistence_file, index=False)
        print(f"✅ Persistence analysis: {persistence_file}")
    
    print(f"📊 All wave physics analysis results saved!")

def main():
    """
    Run complete wave physics analysis suite
    """
    print("COMPREHENSIVE WAVE PHYSICS ANALYSIS")
    print("=" * 60)
    print("1. Rolling range deciles vs. forward performance")
    print("2. Range quintiles × position combinations")
    print("3. Volatility persistence patterns (market-adjusted)")
    print()
    
    # Run all analyses
    decile_results, complete_data = analyze_rolling_range_deciles()
    combo_results = analyze_quintile_position_combinations()
    persistence_results = analyze_volatility_persistence()
    
    # Save all results
    save_all_analysis_results(decile_results, combo_results, persistence_results)
    
    print(f"\n✨ WAVE PHYSICS ANALYSIS COMPLETE!")
    print(f"🎯 Key Questions Answered:")
    print(f"   📊 Do higher rolling ranges predict better forward performance?")
    print(f"   📍 What's the optimal combination of range + position?") 
    print(f"   🔄 Which assets are chronically volatile vs. occasionally spiky?")
    print(f"   🌊 Are your 'loaded springs' persistent or random?")
    
    return decile_results, combo_results, persistence_results

if __name__ == "__main__":
    results = main()