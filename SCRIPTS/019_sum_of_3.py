import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_sum_of_most_deviated_assets():
    """
    Analyze the sum of deviations for the most underperforming assets over time
    """
    print("SUM OF MOST DEVIATION ANALYSIS")
    print("=" * 50)
    
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
    
    # Filter to records with deviation data
    analysis_data = df.dropna(subset=['price_deviation_from_trend']).copy()
    print(f"📊 Records with deviation data: {len(analysis_data):,}")
    print(f"📅 Date range: {analysis_data['date'].min().strftime('%Y-%m-%d')} to {analysis_data['date'].max().strftime('%Y-%m-%d')}")
    
    # Calculate daily metrics
    print(f"\n🔄 Calculating daily deviation metrics...")
    
    daily_metrics = []
    
    # Get unique dates
    unique_dates = sorted(analysis_data['date'].unique())
    print(f"📈 Processing {len(unique_dates):,} unique dates...")
    
    for i, date in enumerate(unique_dates):
        if (i + 1) % 500 == 0:
            print(f"   Processed {i + 1:,}/{len(unique_dates):,} dates...")
        
        # Get all assets for this date
        date_data = analysis_data[analysis_data['date'] == date]
        
        if len(date_data) > 0:
            # Sort by deviation (most negative first)
            sorted_data = date_data.sort_values('price_deviation_from_trend')
            
            # Calculate various metrics
            total_assets = len(date_data)
            
            # Sum of worst 5 deviations
            worst_5 = sorted_data.head(5)['price_deviation_from_trend'].sum()
            
            # Sum of worst 10 deviations
            worst_10 = sorted_data.head(10)['price_deviation_from_trend'].sum()
            
            # Sum of all negative deviations
            negative_devs = sorted_data[sorted_data['price_deviation_from_trend'] < 0]
            sum_all_negative = negative_devs['price_deviation_from_trend'].sum()
            
            # Count of severely negative (< -20%)
            severe_count = len(sorted_data[sorted_data['price_deviation_from_trend'] < -20])
            
            # Average deviation of worst 10%
            worst_10_pct_count = max(1, int(total_assets * 0.1))
            avg_worst_10_pct = sorted_data.head(worst_10_pct_count)['price_deviation_from_trend'].mean()
            
            # Median deviation
            median_deviation = sorted_data['price_deviation_from_trend'].median()
            
            # Most extreme single deviation
            most_extreme = sorted_data['price_deviation_from_trend'].min()
            
            daily_metrics.append({
                'date': date,
                'total_assets': total_assets,
                'sum_worst_5': worst_5,
                'sum_worst_10': worst_10,
                'sum_all_negative': sum_all_negative,
                'count_severe_negative': severe_count,
                'avg_worst_10_percent': avg_worst_10_pct,
                'median_deviation': median_deviation,
                'most_extreme_deviation': most_extreme,
                'assets_below_minus_30': len(sorted_data[sorted_data['price_deviation_from_trend'] < -30]),
                'assets_below_minus_40': len(sorted_data[sorted_data['price_deviation_from_trend'] < -40]),
                'assets_below_minus_50': len(sorted_data[sorted_data['price_deviation_from_trend'] < -50])
            })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(daily_metrics)
    metrics_df = metrics_df.sort_values('date')
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total trading days: {len(metrics_df):,}")
    print(f"   Average assets per day: {metrics_df['total_assets'].mean():.1f}")
    print(f"   Average sum of worst 5: {metrics_df['sum_worst_5'].mean():.1f}%")
    print(f"   Average sum of worst 10: {metrics_df['sum_worst_10'].mean():.1f}%")
    print(f"   Most extreme single day (worst 5): {metrics_df['sum_worst_5'].min():.1f}%")
    
    # Add year and rolling averages
    metrics_df['year'] = metrics_df['date'].dt.year
    metrics_df['rolling_30d_worst_5'] = metrics_df['sum_worst_5'].rolling(window=30, min_periods=15).mean()
    metrics_df['rolling_90d_worst_5'] = metrics_df['sum_worst_5'].rolling(window=90, min_periods=45).mean()
    
    return metrics_df

def create_deviation_plots(metrics_df):
    """
    Create comprehensive plots of deviation metrics
    """
    print(f"\n📈 CREATING DEVIATION PLOTS...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Sum of Most Deviated Assets Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Sum of Worst 5 Deviations Over Time
    ax1 = axes[0, 0]
    ax1.plot(metrics_df['date'], metrics_df['sum_worst_5'], alpha=0.6, linewidth=0.8, label='Daily')
    ax1.plot(metrics_df['date'], metrics_df['rolling_30d_worst_5'], color='red', linewidth=2, label='30-day average')
    ax1.set_title('Sum of 5 Worst Daily Deviations', fontweight='bold')
    ax1.set_ylabel('Sum of Deviations (%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight extreme periods
    extreme_threshold = metrics_df['sum_worst_5'].quantile(0.05)  # Bottom 5%
    extreme_dates = metrics_df[metrics_df['sum_worst_5'] <= extreme_threshold]
    ax1.scatter(extreme_dates['date'], extreme_dates['sum_worst_5'], 
               color='red', s=20, alpha=0.7, label=f'Extreme (<{extreme_threshold:.0f}%)')
    
    # Plot 2: Count of Severely Negative Assets
    ax2 = axes[0, 1]
    ax2.plot(metrics_df['date'], metrics_df['count_severe_negative'], alpha=0.7, linewidth=1)
    ax2.fill_between(metrics_df['date'], metrics_df['count_severe_negative'], alpha=0.3)
    ax2.set_title('Count of Assets with Deviation < -20%', fontweight='bold')
    ax2.set_ylabel('Number of Assets')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution by Severity Levels
    ax3 = axes[1, 0]
    ax3.plot(metrics_df['date'], metrics_df['assets_below_minus_30'], label='< -30%', linewidth=1.5)
    ax3.plot(metrics_df['date'], metrics_df['assets_below_minus_40'], label='< -40%', linewidth=1.5)
    ax3.plot(metrics_df['date'], metrics_df['assets_below_minus_50'], label='< -50%', linewidth=1.5)
    ax3.set_title('Assets by Deviation Severity Levels', fontweight='bold')
    ax3.set_ylabel('Number of Assets')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Yearly Boxplot of Worst 5 Sum
    ax4 = axes[1, 1]
    yearly_data = [metrics_df[metrics_df['year'] == year]['sum_worst_5'].values 
                   for year in sorted(metrics_df['year'].unique())]
    years = sorted(metrics_df['year'].unique())
    
    ax4.boxplot(yearly_data, labels=years)
    ax4.set_title('Annual Distribution of Worst 5 Sum', fontweight='bold')
    ax4.set_ylabel('Sum of 5 Worst Deviations (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis for time series plots
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    plot_file = os.path.join(results_dir, "sum_of_most_deviation_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved: {plot_file}")
    
    plt.show()

def create_crisis_analysis(metrics_df):
    """
    Identify and analyze crisis periods based on deviation metrics
    """
    print(f"\n🚨 CRISIS PERIOD ANALYSIS")
    print("-" * 50)
    
    # Define crisis threshold (bottom 5% of worst 5 sum)
    crisis_threshold = metrics_df['sum_worst_5'].quantile(0.05)
    print(f"Crisis threshold (sum of worst 5): {crisis_threshold:.1f}%")
    
    # Find crisis periods
    crisis_data = metrics_df[metrics_df['sum_worst_5'] <= crisis_threshold].copy()
    print(f"Crisis days identified: {len(crisis_data):,}")
    
    if len(crisis_data) > 0:
        # Group consecutive crisis days
        crisis_data['date_diff'] = crisis_data['date'].diff().dt.days
        crisis_data['new_period'] = (crisis_data['date_diff'] > 30) | (crisis_data['date_diff'].isna())
        crisis_data['period_id'] = crisis_data['new_period'].cumsum()
        
        # Analyze each crisis period
        crisis_periods = []
        for period_id in crisis_data['period_id'].unique():
            period_data = crisis_data[crisis_data['period_id'] == period_id]
            
            crisis_periods.append({
                'period_id': period_id,
                'start_date': period_data['date'].min(),
                'end_date': period_data['date'].max(),
                'duration_days': (period_data['date'].max() - period_data['date'].min()).days + 1,
                'worst_sum_5': period_data['sum_worst_5'].min(),
                'avg_sum_5': period_data['sum_worst_5'].mean(),
                'max_severe_count': period_data['count_severe_negative'].max(),
                'total_crisis_days': len(period_data)
            })
        
        # Display crisis periods
        print(f"\n🔴 IDENTIFIED CRISIS PERIODS:")
        print(f"{'Period':>6} | {'Start Date':>12} | {'End Date':>12} | {'Days':>5} | {'Worst Sum':>10} | {'Avg Sum':>8}")
        print("-" * 70)
        
        for crisis in sorted(crisis_periods, key=lambda x: x['worst_sum_5']):
            print(f"{crisis['period_id']:>6} | {crisis['start_date'].strftime('%Y-%m-%d'):>12} | "
                  f"{crisis['end_date'].strftime('%Y-%m-%d'):>12} | {crisis['duration_days']:>5} | "
                  f"{crisis['worst_sum_5']:>9.1f}% | {crisis['avg_sum_5']:>7.1f}%")
        
        return crisis_periods
    
    return []

def analyze_market_stress_levels(metrics_df):
    """
    Create market stress level classification
    """
    print(f"\n📊 MARKET STRESS LEVEL ANALYSIS")
    print("-" * 50)
    
    # Define stress levels based on sum of worst 5
    def classify_stress_level(sum_worst_5):
        if sum_worst_5 >= -50:
            return "Low Stress"
        elif sum_worst_5 >= -100:
            return "Moderate Stress"
        elif sum_worst_5 >= -150:
            return "High Stress"
        elif sum_worst_5 >= -200:
            return "Severe Stress"
        else:
            return "Crisis"
    
    metrics_df['stress_level'] = metrics_df['sum_worst_5'].apply(classify_stress_level)
    
    # Show stress level distribution
    stress_counts = metrics_df['stress_level'].value_counts()
    total_days = len(metrics_df)
    
    print("Market stress level distribution:")
    stress_order = ["Low Stress", "Moderate Stress", "High Stress", "Severe Stress", "Crisis"]
    
    for level in stress_order:
        if level in stress_counts:
            count = stress_counts[level]
            percentage = (count / total_days) * 100
            print(f"   {level:>15}: {count:>6,} days ({percentage:>5.1f}%)")
    
    # Yearly stress analysis
    print(f"\nStress levels by year:")
    yearly_stress = metrics_df.groupby(['year', 'stress_level']).size().unstack(fill_value=0)
    
    for year in sorted(metrics_df['year'].unique()):
        if year in yearly_stress.index:
            year_data = yearly_stress.loc[year]
            year_total = year_data.sum()
            crisis_days = year_data.get('Crisis', 0)
            severe_days = year_data.get('Severe Stress', 0)
            
            print(f"   {year}: {crisis_days:>3} crisis, {severe_days:>3} severe (of {year_total:>3} total days)")
    
    return metrics_df

def save_results(metrics_df, crisis_periods):
    """
    Save all analysis results
    """
    print(f"\n💾 SAVING RESULTS...")
    
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    
    # Save daily metrics
    daily_file = os.path.join(results_dir, "DAILY_DEVIATION_METRICS.csv")
    metrics_df.to_csv(daily_file, index=False)
    print(f"✅ Daily metrics saved: {daily_file}")
    
    # Save crisis periods
    if crisis_periods:
        crisis_file = os.path.join(results_dir, "CRISIS_PERIODS_ANALYSIS.csv")
        pd.DataFrame(crisis_periods).to_csv(crisis_file, index=False)
        print(f"✅ Crisis periods saved: {crisis_file}")
    
    print(f"📊 Analysis complete!")

def main():
    """
    Main function to run the sum of most deviation analysis
    """
    print("STARTING SUM OF MOST DEVIATION ANALYSIS")
    print("Analyzing daily sums of worst-performing asset deviations")
    print()
    
    # Run the analysis
    metrics_df = analyze_sum_of_most_deviated_assets()
    
    if metrics_df is not None:
        # Create plots
        create_deviation_plots(metrics_df)
        
        # Crisis analysis
        crisis_periods = create_crisis_analysis(metrics_df)
        
        # Market stress analysis
        metrics_df = analyze_market_stress_levels(metrics_df)
        
        # Save results
        save_results(metrics_df, crisis_periods)
        
        print(f"\n✨ SUM OF MOST DEVIATION ANALYSIS COMPLETE!")
        print(f"   📈 Daily deviation metrics calculated and plotted")
        print(f"   🚨 Crisis periods identified and analyzed")
        print(f"   📊 Market stress levels classified")
        print(f"   💾 All results saved to CSV files")
        print(f"   🎯 Focus: Worst 3 assets per day, 40%+ deviation thresholds")
    
    return metrics_df

if __name__ == "__main__":
    results = main()