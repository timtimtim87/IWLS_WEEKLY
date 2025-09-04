import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

class OutputLogger:
    """Class to capture and log all output to both console and file"""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        
    def write(self, message):
        # Write to both console and file
        self.stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate writing
        
    def flush(self):
        self.stdout.flush()
        self.log_file.flush()
        
    def close(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()
        sys.stdout = self.stdout

def assign_negative_deviation_bin(deviation):
    """
    Assign deviation to 5% magnitude bins for negative values only
    Focus on -40% to -75% range with 5% increments
    """
    if pd.isna(deviation):
        return 'No Data'
    
    # Only process negative deviations in our range of interest
    if deviation >= -40:
        return 'Above -40%'  # Not in our analysis range
    elif deviation >= -45:
        return "-40% to -45%"
    elif deviation >= -50:
        return "-45% to -50%"
    elif deviation >= -55:
        return "-50% to -55%"
    elif deviation >= -60:
        return "-55% to -60%"
    elif deviation >= -65:
        return "-60% to -65%"
    elif deviation >= -70:
        return "-65% to -70%"
    elif deviation >= -75:
        return "-70% to -75%"
    else:
        return "Below -75%"

def analyze_negative_deviation_bins_by_year():
    """
    Analyze negative IWLS deviation bins (5% increments) by year
    """
    print("NEGATIVE IWLS DEVIATION ANALYSIS BY YEAR (5% BINS)")
    print("=" * 60)
    
    # Load the forward performance dataset
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    input_file = os.path.join(results_dir, "IWLS_WITH_FORWARD_PERFORMANCE.csv")
    
    if not os.path.exists(input_file):
        print(f"⚠️ Error: Forward performance file not found at {input_file}")
        return None
    
    print(f"📂 Loading: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Successfully loaded {len(df):,} records")
    except Exception as e:
        print(f"⚠️ Error loading file: {e}")
        return None
    
    # Filter and prepare data
    analysis_data = df.dropna(subset=['price_deviation_from_trend']).copy()
    print(f"📊 Records with deviation data: {len(analysis_data):,}")
    
    # Add year column
    analysis_data['year'] = analysis_data['date'].dt.year
    
    # Assign negative deviation bins
    analysis_data['negative_dev_bin'] = analysis_data['price_deviation_from_trend'].apply(assign_negative_deviation_bin)
    
    # Filter to only our bins of interest (exclude 'Above -40%')
    target_bins = ["-40% to -45%", "-45% to -50%", "-50% to -55%", "-55% to -60%", 
                   "-60% to -65%", "-65% to -70%", "-70% to -75%", "Below -75%"]
    
    negative_data = analysis_data[analysis_data['negative_dev_bin'].isin(target_bins)].copy()
    print(f"📉 Records in negative deviation bins (-40% to <-75%): {len(negative_data):,}")
    
    if len(negative_data) == 0:
        print("⚠️ No records found in the target deviation range!")
        return None
    
    # Show overall distribution
    print(f"\n📈 OVERALL NEGATIVE DEVIATION BIN DISTRIBUTION:")
    bin_counts = negative_data['negative_dev_bin'].value_counts()
    total_records = len(negative_data)
    
    for bin_name in target_bins:
        if bin_name in bin_counts:
            count = bin_counts[bin_name]
            percentage = (count / total_records) * 100
            print(f"   {bin_name:>15}: {count:>8,} records ({percentage:>5.1f}%)")
        else:
            print(f"   {bin_name:>15}: {0:>8,} records ({0:>5.1f}%)")
    
    # Show year distribution
    year_counts = negative_data['year'].value_counts().sort_index()
    print(f"\n📅 YEAR DISTRIBUTION:")
    print(f"   Date range: {negative_data['date'].min().strftime('%Y-%m-%d')} to {negative_data['date'].max().strftime('%Y-%m-%d')}")
    for year, count in year_counts.items():
        percentage = (count / len(negative_data)) * 100
        print(f"   {year}: {count:>6,} records ({percentage:>5.1f}%)")
    
    # Define periods for analysis
    periods = ['6_month', '12_month', '18_month']
    
    # Analysis by year and deviation bin
    all_results = []
    
    print(f"\n🔍 ANALYZING BY YEAR AND DEVIATION BIN...")
    print("=" * 100)
    
    for year in sorted(year_counts.index):
        year_data = negative_data[negative_data['year'] == year]
        
        print(f"\n📅 YEAR {year} - NEGATIVE DEVIATION ANALYSIS")
        print("-" * 80)
        print(f"Total records for {year}: {len(year_data):,}")
        
        # Show bin distribution for this year
        year_bin_counts = year_data['negative_dev_bin'].value_counts()
        print(f"\nDeviation bin distribution for {year}:")
        
        for bin_name in target_bins:
            if bin_name in year_bin_counts:
                count = year_bin_counts[bin_name]
                pct = (count / len(year_data)) * 100
                print(f"   {bin_name:>15}: {count:>4,} ({pct:>5.1f}%)")
        
        # Analyze each deviation bin for this year
        for period in periods:
            period_display = period.replace('_', '-').upper()
            print(f"\n🔍 {year} - {period_display} FORWARD PERFORMANCE:")
            print("-" * 85)
            print(f"{'Deviation Bin':>15} | {'Count':>5} | {'Avg Return':>10} | {'Avg Max Gain':>12} | {'Avg Max DD':>10}")
            print("-" * 85)
            
            for bin_name in target_bins:
                bin_data = year_data[year_data['negative_dev_bin'] == bin_name]
                
                if len(bin_data) > 0:
                    return_col = f'forward_return_{period}'
                    gain_col = f'max_gain_{period}'
                    dd_col = f'max_drawdown_{period}'
                    
                    return_data = bin_data[return_col].dropna()
                    gain_data = bin_data[gain_col].dropna()
                    dd_data = bin_data[dd_col].dropna()
                    
                    if len(return_data) > 0:
                        count = len(return_data)
                        avg_return = return_data.mean()
                        avg_gain = gain_data.mean() if len(gain_data) > 0 else np.nan
                        avg_dd = dd_data.mean() if len(dd_data) > 0 else np.nan
                        
                        print(f"{bin_name:>15} | {count:>5,} | {avg_return:>9.1f}% | {avg_gain:>11.1f}% | {avg_dd:>9.1f}%")
                        
                        # Store detailed results
                        all_results.append({
                            'year': year,
                            'deviation_bin': bin_name,
                            'period': period,
                            'count': count,
                            'avg_forward_return': avg_return,
                            'median_forward_return': return_data.median(),
                            'std_forward_return': return_data.std(),
                            'avg_max_gain': avg_gain,
                            'avg_max_drawdown': avg_dd,
                            'min_return': return_data.min(),
                            'max_return': return_data.max(),
                            'total_opportunities': len(bin_data)  # Including those without forward data
                        })
                    else:
                        print(f"{bin_name:>15} | {'0':>5} | {'N/A':>9} | {'N/A':>11} | {'N/A':>9}")
                        
                        # Still record the opportunity count
                        all_results.append({
                            'year': year,
                            'deviation_bin': bin_name,
                            'period': period,
                            'count': 0,
                            'avg_forward_return': np.nan,
                            'median_forward_return': np.nan,
                            'std_forward_return': np.nan,
                            'avg_max_gain': np.nan,
                            'avg_max_drawdown': np.nan,
                            'min_return': np.nan,
                            'max_return': np.nan,
                            'total_opportunities': len(bin_data)
                        })
                else:
                    print(f"{bin_name:>15} | {'0':>5} | {'N/A':>9} | {'N/A':>11} | {'N/A':>9}")
                    
                    # Record zero opportunities
                    all_results.append({
                        'year': year,
                        'deviation_bin': bin_name,
                        'period': period,
                        'count': 0,
                        'avg_forward_return': np.nan,
                        'median_forward_return': np.nan,
                        'std_forward_return': np.nan,
                        'avg_max_gain': np.nan,
                        'avg_max_drawdown': np.nan,
                        'min_return': np.nan,
                        'max_return': np.nan,
                        'total_opportunities': 0
                    })
    
    # SUMMARY ANALYSIS
    print(f"\n📊 SUMMARY ANALYSIS - TRADING OPPORTUNITIES BY BIN")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    
    # Trading opportunity summary (12-month data)
    print(f"\n🎯 12-MONTH TRADING OPPORTUNITIES SUMMARY:")
    print("-" * 60)
    
    opportunity_summary = []
    for bin_name in target_bins:
        bin_results = results_df[(results_df['deviation_bin'] == bin_name) & 
                                (results_df['period'] == '12_month')]
        
        total_opportunities = bin_results['total_opportunities'].sum()
        total_with_forward_data = bin_results['count'].sum()
        years_with_opportunities = (bin_results['total_opportunities'] > 0).sum()
        
        if total_with_forward_data > 0:
            avg_return = bin_results[bin_results['count'] > 0]['avg_forward_return'].mean()
            best_year_return = bin_results[bin_results['count'] > 0]['avg_forward_return'].max()
            worst_year_return = bin_results[bin_results['count'] > 0]['avg_forward_return'].min()
        else:
            avg_return = np.nan
            best_year_return = np.nan
            worst_year_return = np.nan
        
        print(f"\n{bin_name}:")
        print(f"   Total opportunities: {total_opportunities:,}")
        print(f"   Opportunities with forward data: {total_with_forward_data:,}")
        print(f"   Years with opportunities: {years_with_opportunities}")
        if not pd.isna(avg_return):
            print(f"   Average 12m return: {avg_return:.1f}%")
            print(f"   Best year average: {best_year_return:.1f}%")
            print(f"   Worst year average: {worst_year_return:.1f}%")
        
        opportunity_summary.append({
            'deviation_bin': bin_name,
            'total_opportunities': total_opportunities,
            'opportunities_with_data': total_with_forward_data,
            'years_with_opportunities': years_with_opportunities,
            'avg_12m_return': avg_return,
            'best_year_return': best_year_return,
            'worst_year_return': worst_year_return
        })
    
    # Year-over-year comparison for most active bins
    print(f"\n📈 YEAR-OVER-YEAR COMPARISON (12-MONTH RETURNS)")
    print("=" * 90)
    
    # Find bins with most opportunities
    opp_df = pd.DataFrame(opportunity_summary)
    top_bins = opp_df.nlargest(3, 'opportunities_with_data')['deviation_bin'].tolist()
    
    for bin_name in top_bins:
        print(f"\n{bin_name} - Year-over-year 12m returns:")
        bin_yearly = results_df[(results_df['deviation_bin'] == bin_name) & 
                               (results_df['period'] == '12_month') & 
                               (results_df['count'] > 0)]
        
        if len(bin_yearly) > 0:
            print("   Year | Count | Avg Return | Med Return | Std Dev")
            print("   -----|-------|------------|------------|--------")
            
            for _, row in bin_yearly.sort_values('year').iterrows():
                year = int(row['year'])
                count = int(row['count'])
                avg_ret = row['avg_forward_return']
                med_ret = row['median_forward_return']
                std_ret = row['std_forward_return']
                
                print(f"   {year} | {count:>5,} | {avg_ret:>9.1f}% | {med_ret:>9.1f}% | {std_ret:>6.1f}%")
    
    # Save detailed results
    print(f"\n💾 SAVING RESULTS...")
    
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    output_file = os.path.join(results_dir, "NEGATIVE_DEVIATION_YEARLY_ANALYSIS.csv")
    results_df.to_csv(output_file, index=False)
    
    # Save opportunity summary
    summary_file = os.path.join(results_dir, "NEGATIVE_DEVIATION_OPPORTUNITY_SUMMARY.csv")
    pd.DataFrame(opportunity_summary).to_csv(summary_file, index=False)
    
    print(f"✅ Detailed results saved: {output_file}")
    print(f"✅ Opportunity summary saved: {summary_file}")
    print(f"📊 Total result rows: {len(results_df):,}")
    
    print(f"\n✨ NEGATIVE DEVIATION ANALYSIS COMPLETE!")
    print(f"🎯 Key findings:")
    print(f"   📊 Analyzed {len(negative_data):,} records across {len(year_counts)} years")
    print(f"   📉 Focus on severe underperformance: -40% to -75%+ deviation")
    print(f"   🔍 5% increment bins for detailed analysis")
    print(f"   📈 Year-over-year performance comparison available")
    
    return results_df, opportunity_summary

def show_sample_extreme_cases():
    """
    Show some sample extreme cases for context
    """
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    input_file = os.path.join(results_dir, "IWLS_WITH_FORWARD_PERFORMANCE.csv")
    
    if not os.path.exists(input_file):
        print("No data file found for sample analysis.")
        return
    
    print(f"\n📋 SAMPLE EXTREME NEGATIVE DEVIATION CASES:")
    print("-" * 70)
    
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Find some extreme cases
    extreme_cases = df[df['price_deviation_from_trend'] < -60].copy()
    
    if len(extreme_cases) > 0:
        print(f"Found {len(extreme_cases):,} cases with deviation < -60%")
        
        # Show a few examples
        sample_cases = extreme_cases.nlargest(10, 'price_deviation_from_trend').copy()
        sample_cases = sample_cases.dropna(subset=['forward_return_12_month'])
        
        if len(sample_cases) > 0:
            print(f"\nTop 10 most extreme cases with 12m forward data:")
            print("Asset | Date | Deviation | 12m Return | Max Gain | Max DD")
            print("------|------|-----------|------------|----------|-------")
            
            for _, row in sample_cases.head(5).iterrows():
                asset = row['asset']
                date = row['date'].strftime('%Y-%m-%d')
                deviation = row['price_deviation_from_trend']
                ret_12m = row['forward_return_12_month']
                max_gain = row['max_gain_12_month']
                max_dd = row['max_drawdown_12_month']
                
                print(f"{asset:>5} | {date} | {deviation:>8.1f}% | {ret_12m:>9.1f}% | {max_gain:>7.1f}% | {max_dd:>6.1f}%")

if __name__ == "__main__":
    # Set up output logging
    docs_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/DOCS"
    
    # Create DOCS directory if it doesn't exist
    os.makedirs(docs_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(docs_dir, f"negative_deviation_analysis_{timestamp}.txt")
    
    # Set up output logger
    logger = OutputLogger(log_file_path)
    sys.stdout = logger
    
    try:
        # Add header to the log file
        print(f"NEGATIVE IWLS DEVIATION ANALYSIS REPORT")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Script: 013_deviation_yearly_analysis.py")
        print("=" * 80)
        print()
        
        # Run the negative deviation analysis
        results, summary = analyze_negative_deviation_bins_by_year()
        
        if results is not None:
            # Show some sample extreme cases for context
            show_sample_extreme_cases()
            
            print(f"\n🚀 ANALYSIS COMPLETE!")
            print(f"   🔍 Use 'NEGATIVE_DEVIATION_YEARLY_ANALYSIS.csv' for detailed data")
            print(f"   📊 Use 'NEGATIVE_DEVIATION_OPPORTUNITY_SUMMARY.csv' for overview")
            print(f"   🎯 Focus on bins with consistent opportunities across years")
            print(f"   📈 Look for bins with improving/declining performance over time")
            
            print(f"\n📄 REPORT SAVED TO: {log_file_path}")
            print("=" * 80)
    
    except Exception as e:
        print(f"⚠️ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close the logger and restore normal stdout
        logger.close()
        
        # Print final message to console only
        print(f"\n✅ Analysis complete! Full report saved to:")
        print(f"   📄 {log_file_path}")