import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_daily_worst_5_sum():
    """
    Calculate the sum of the 5 worst deviations for each trading day
    """
    print("CALCULATING DAILY DEVIATION SUMS (5 WORST ASSETS)")
    print("=" * 60)
    
    # Load the IWLS data
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    input_file = os.path.join(results_dir, "IWLS_WITH_FORWARD_PERFORMANCE.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Data file not found at {input_file}")
        return None
    
    print(f"📁 Loading: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Successfully loaded {len(df):,} records")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Filter to records with deviation data and 12-month forward data
    analysis_data = df.dropna(subset=['price_deviation_from_trend', 'forward_return_12_month']).copy()
    print(f"📊 Records with deviation and 12m forward data: {len(analysis_data):,}")
    print(f"📅 Date range: {analysis_data['date'].min().strftime('%Y-%m-%d')} to {analysis_data['date'].max().strftime('%Y-%m-%d')}")
    
    # Calculate daily deviation sums and collect individual asset data
    print(f"\n🔄 Calculating daily sums and preparing asset-level data...")
    
    daily_sums = []
    individual_records = []
    
    # Get unique dates
    unique_dates = sorted(analysis_data['date'].unique())
    print(f"📈 Processing {len(unique_dates):,} unique dates...")
    
    for i, date in enumerate(unique_dates):
        if (i + 1) % 500 == 0:
            print(f"   Processed {i + 1:,}/{len(unique_dates):,} dates...")
        
        # Get all assets for this date
        date_data = analysis_data[analysis_data['date'] == date]
        
        if len(date_data) >= 5:  # Need at least 5 assets
            # Sort by deviation (most negative first)
            sorted_data = date_data.sort_values('price_deviation_from_trend')
            
            # Get worst 5
            worst_5 = sorted_data.head(5)
            
            # Calculate sum of worst 5 deviations
            sum_worst_5 = worst_5['price_deviation_from_trend'].sum()
            
            # Record the daily sum
            daily_sums.append({
                'date': date,
                'sum_worst_5_deviations': sum_worst_5,
                'num_assets': len(date_data)
            })
            
            # Record individual asset data for the worst 5
            for _, asset_row in worst_5.iterrows():
                individual_records.append({
                    'date': date,
                    'asset': asset_row['asset'],
                    'deviation': asset_row['price_deviation_from_trend'],
                    'forward_return_12m': asset_row['forward_return_12_month'],
                    'sum_worst_5_deviations': sum_worst_5,
                    'rank_in_worst_5': list(worst_5['asset']).index(asset_row['asset']) + 1
                })
    
    # Convert to DataFrames
    daily_sums_df = pd.DataFrame(daily_sums)
    individual_df = pd.DataFrame(individual_records)
    
    print(f"\n📊 DAILY SUMS SUMMARY:")
    print(f"   Valid trading days: {len(daily_sums_df):,}")
    print(f"   Individual asset records: {len(individual_df):,}")
    print(f"   Average sum of worst 5: {daily_sums_df['sum_worst_5_deviations'].mean():.1f}%")
    print(f"   Most extreme day: {daily_sums_df['sum_worst_5_deviations'].min():.1f}%")
    print(f"   Least extreme day: {daily_sums_df['sum_worst_5_deviations'].max():.1f}%")
    
    return daily_sums_df, individual_df

def analyze_forward_performance_by_deciles(daily_sums_df, individual_df):
    """
    Analyze 12-month forward performance by deciles of deviation sum
    """
    print(f"\n📊 DECILE ANALYSIS OF FORWARD PERFORMANCE")
    print("=" * 60)
    
    # Create deciles based on sum of worst 5 deviations
    daily_sums_df['decile'] = pd.qcut(
        daily_sums_df['sum_worst_5_deviations'], 
        q=10, 
        labels=['D1 (Best)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Worst)'],
        precision=1
    )
    
    # Show decile boundaries
    decile_boundaries = daily_sums_df.groupby('decile')['sum_worst_5_deviations'].agg(['min', 'max', 'mean', 'count'])
    
    print(f"\n📋 DECILE BOUNDARIES (Sum of Worst 5 Deviations):")
    print(f"{'Decile':>12} | {'Min':>8} | {'Max':>8} | {'Mean':>8} | {'Days':>6}")
    print("-" * 55)
    
    for decile in decile_boundaries.index:
        min_val = decile_boundaries.loc[decile, 'min']
        max_val = decile_boundaries.loc[decile, 'max']
        mean_val = decile_boundaries.loc[decile, 'mean']
        count_val = int(decile_boundaries.loc[decile, 'count'])
        
        print(f"{decile:>12} | {min_val:>7.1f}% | {max_val:>7.1f}% | {mean_val:>7.1f}% | {count_val:>6,}")
    
    # Merge decile information with individual records
    individual_with_deciles = individual_df.merge(
        daily_sums_df[['date', 'decile']], 
        on='date', 
        how='left'
    )
    
    print(f"\n📈 Individual records with decile assignments: {len(individual_with_deciles):,}")
    
    # Analyze forward performance by decile
    print(f"\n🎯 12-MONTH FORWARD PERFORMANCE BY DECILE")
    print("=" * 70)
    
    decile_performance = individual_with_deciles.groupby('decile').agg({
        'forward_return_12m': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'deviation': 'mean'
    }).round(2)
    
    # Flatten column names
    decile_performance.columns = ['Count', 'Mean_Return', 'Median_Return', 'Std_Return', 'Min_Return', 'Max_Return', 'Avg_Deviation']
    
    print(f"{'Decile':>12} | {'Count':>6} | {'Mean':>8} | {'Median':>8} | {'Std':>7} | {'Min':>8} | {'Max':>8} | {'Avg Dev':>8}")
    print("-" * 85)
    
    performance_summary = []
    
    for decile in decile_performance.index:
        row = decile_performance.loc[decile]
        count = int(row['Count'])
        mean_ret = row['Mean_Return']
        median_ret = row['Median_Return']
        std_ret = row['Std_Return']
        min_ret = row['Min_Return']
        max_ret = row['Max_Return']
        avg_dev = row['Avg_Deviation']
        
        print(f"{decile:>12} | {count:>6,} | {mean_ret:>7.1f}% | {median_ret:>7.1f}% | {std_ret:>6.1f}% | {min_ret:>7.1f}% | {max_ret:>7.1f}% | {avg_dev:>7.1f}%")
        
        performance_summary.append({
            'decile': decile,
            'count': count,
            'mean_return_12m': mean_ret,
            'median_return_12m': median_ret,
            'std_return_12m': std_ret,
            'min_return_12m': min_ret,
            'max_return_12m': max_ret,
            'avg_individual_deviation': avg_dev
        })
    
    # Additional analysis
    print(f"\n📊 ADDITIONAL INSIGHTS:")
    
    # Best vs Worst decile comparison
    best_decile = individual_with_deciles[individual_with_deciles['decile'] == 'D1 (Best)']['forward_return_12m']
    worst_decile = individual_with_deciles[individual_with_deciles['decile'] == 'D10 (Worst)']['forward_return_12m']
    
    if len(best_decile) > 0 and len(worst_decile) > 0:
        best_mean = best_decile.mean()
        worst_mean = worst_decile.mean()
        difference = worst_mean - best_mean
        
        print(f"   Best Decile (D1) avg return: {best_mean:.1f}%")
        print(f"   Worst Decile (D10) avg return: {worst_mean:.1f}%")
        print(f"   Difference (D10 - D1): {difference:+.1f}%")
        
        # Win rates
        best_win_rate = (best_decile > 0).mean() * 100
        worst_win_rate = (worst_decile > 0).mean() * 100
        
        print(f"   Best Decile win rate: {best_win_rate:.1f}%")
        print(f"   Worst Decile win rate: {worst_win_rate:.1f}%")
    
    # Correlation analysis
    correlation = individual_with_deciles['sum_worst_5_deviations'].corr(individual_with_deciles['forward_return_12m'])
    print(f"   Correlation (sum vs forward return): {correlation:.3f}")
    
    return performance_summary, individual_with_deciles

def analyze_by_rank_within_worst_5(individual_with_deciles):
    """
    Analyze performance by rank within the worst 5 assets
    """
    print(f"\n🏆 PERFORMANCE BY RANK WITHIN WORST 5")
    print("=" * 50)
    
    rank_performance = individual_with_deciles.groupby('rank_in_worst_5').agg({
        'forward_return_12m': ['count', 'mean', 'median', 'std'],
        'deviation': 'mean'
    }).round(2)
    
    # Flatten column names
    rank_performance.columns = ['Count', 'Mean_Return', 'Median_Return', 'Std_Return', 'Avg_Deviation']
    
    print(f"{'Rank':>6} | {'Count':>6} | {'Mean':>8} | {'Median':>8} | {'Std':>7} | {'Avg Dev':>8}")
    print("-" * 55)
    
    for rank in sorted(rank_performance.index):
        row = rank_performance.loc[rank]
        count = int(row['Count'])
        mean_ret = row['Mean_Return']
        median_ret = row['Median_Return']
        std_ret = row['Std_Return']
        avg_dev = row['Avg_Deviation']
        
        print(f"{rank:>6} | {count:>6,} | {mean_ret:>7.1f}% | {median_ret:>7.1f}% | {std_ret:>6.1f}% | {avg_dev:>7.1f}%")
    
    print(f"\nNote: Rank 1 = worst performing asset of the day, Rank 5 = 5th worst")

def create_decile_trend_analysis(individual_with_deciles):
    """
    Analyze how decile performance has changed over time
    """
    print(f"\n📈 DECILE PERFORMANCE TRENDS OVER TIME")
    print("=" * 50)
    
    # Add year column
    individual_with_deciles['year'] = pd.to_datetime(individual_with_deciles['date']).dt.year
    
    # Group by year and decile
    yearly_decile_performance = individual_with_deciles.groupby(['year', 'decile'])['forward_return_12m'].mean().unstack(fill_value=np.nan)
    
    # Show performance by decade
    decades = {
        '1990s': [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999],
        '2000s': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
        '2010s': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
        '2020s': [2020, 2021, 2022, 2023, 2024, 2025]
    }
    
    print(f"\nAverage 12-month returns by decade and decile:")
    print(f"{'Decade':>8} | {'D1 (Best)':>9} | {'D5':>8} | {'D10 (Worst)':>11} | {'D10-D1':>8}")
    print("-" * 55)
    
    for decade, years in decades.items():
        decade_data = yearly_decile_performance[yearly_decile_performance.index.isin(years)]
        if len(decade_data) > 0:
            d1_avg = decade_data['D1 (Best)'].mean()
            d5_avg = decade_data['D5'].mean()
            d10_avg = decade_data['D10 (Worst)'].mean()
            
            if not (pd.isna(d1_avg) or pd.isna(d10_avg)):
                difference = d10_avg - d1_avg
                
                print(f"{decade:>8} | {d1_avg:>8.1f}% | {d5_avg:>7.1f}% | {d10_avg:>10.1f}% | {difference:>+7.1f}%")

def save_analysis_results(daily_sums_df, performance_summary, individual_with_deciles):
    """
    Save all analysis results to CSV files
    """
    print(f"\n💾 SAVING ANALYSIS RESULTS...")
    
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    
    # Save daily sums with deciles
    daily_file = os.path.join(results_dir, "DAILY_DEVIATION_SUMS_WITH_DECILES.csv")
    daily_sums_df.to_csv(daily_file, index=False)
    print(f"✅ Daily sums saved: {daily_file}")
    
    # Save decile performance summary
    summary_file = os.path.join(results_dir, "DECILE_FORWARD_PERFORMANCE_SUMMARY.csv")
    pd.DataFrame(performance_summary).to_csv(summary_file, index=False)
    print(f"✅ Performance summary saved: {summary_file}")
    
    # Save individual records with deciles
    individual_file = os.path.join(results_dir, "INDIVIDUAL_WORST5_WITH_DECILES.csv")
    individual_with_deciles.to_csv(individual_file, index=False)
    print(f"✅ Individual records saved: {individual_file}")
    
    print(f"📊 All analysis results saved successfully!")

def main():
    """
    Main function to run the deviation sum forward performance analysis
    """
    print("DEVIATION SUM FORWARD PERFORMANCE ANALYSIS")
    print("=" * 60)
    print("Analyzing 12-month forward returns by deciles of daily worst-5 deviation sums")
    print()
    
    # Step 1: Calculate daily deviation sums
    daily_sums_df, individual_df = calculate_daily_worst_5_sum()
    
    if daily_sums_df is not None and individual_df is not None:
        
        # Step 2: Analyze forward performance by deciles
        performance_summary, individual_with_deciles = analyze_forward_performance_by_deciles(
            daily_sums_df, individual_df
        )
        
        # Step 3: Analyze by rank within worst 5
        analyze_by_rank_within_worst_5(individual_with_deciles)
        
        # Step 4: Trend analysis over time
        create_decile_trend_analysis(individual_with_deciles)
        
        # Step 5: Save results
        save_analysis_results(daily_sums_df, performance_summary, individual_with_deciles)
        
        print(f"\n✨ DEVIATION SUM FORWARD PERFORMANCE ANALYSIS COMPLETE!")
        print(f"🎯 Key Findings:")
        print(f"   📊 Analyzed forward performance across 10 deciles of market stress")
        print(f"   📈 Examined individual asset returns within worst-5 groups")
        print(f"   🏆 Compared performance by rank within worst-performing assets")
        print(f"   📅 Tracked decile performance trends across decades")
        print(f"   💾 Saved comprehensive results for further analysis")
        
        return performance_summary, individual_with_deciles
    
    return None, None

if __name__ == "__main__":
    results = main()