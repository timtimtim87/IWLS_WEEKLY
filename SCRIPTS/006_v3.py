import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def assign_deviation_bin(deviation):
    """
    Assign deviation to 10% magnitude bins
    """
    if pd.isna(deviation):
        return 'No Data'
    
    # Round to nearest 10% bin
    if deviation >= 50:
        return ">+50%"
    elif deviation >= 40:
        return "+40% to +50%"
    elif deviation >= 30:
        return "+30% to +40%"
    elif deviation >= 20:
        return "+20% to +30%"
    elif deviation >= 10:
        return "+10% to +20%"
    elif deviation >= -10:
        return "-10% to +10%"
    elif deviation >= -20:
        return "-20% to -10%"
    elif deviation >= -30:
        return "-30% to -20%"
    elif deviation >= -40:
        return "-40% to -30%"
    elif deviation >= -50:
        return "-50% to -40%"
    else:
        return "<-50%"

def assign_5_year_period(date):
    """
    Assign date to 5-year periods starting from pre-1990
    """
    year = date.year
    if year < 1990:
        return "Pre-1990"
    elif year < 1995:
        return "1990-1994"
    elif year < 2000:
        return "1995-1999"
    elif year < 2005:
        return "2000-2004"
    elif year < 2010:
        return "2005-2009"
    elif year < 2015:
        return "2010-2014"
    elif year < 2020:
        return "2015-2019"
    elif year < 2025:
        return "2020-2024"
    else:
        return "2025+"

def display_deviation_bin_tables(data, title_prefix=""):
    """Display the deviation bin performance tables exactly like you want"""
    
    periods = ['6_month', '12_month', '18_month']
    bin_order = [">+50%", "+40% to +50%", "+30% to +40%", "+20% to +30%", "+10% to +20%",
                 "-10% to +10%", "-20% to -10%", "-30% to -20%", "-40% to -30%", 
                 "-50% to -40%", "<-50%"]
    
    bin_results = {}
    
    # Calculate stats for each bin
    for bin_name in bin_order:
        bin_data = data[data['deviation_bin'] == bin_name]
        if len(bin_data) > 0:
            bin_stats = {'bin_name': bin_name}
            
            for period in periods:
                for metric in ['forward_return', 'max_gain', 'max_drawdown']:
                    column_name = f'{metric}_{period}'
                    if column_name in data.columns:
                        metric_data = bin_data[column_name].dropna()
                        if len(metric_data) > 0:
                            bin_stats[f'{metric}_{period}_count'] = len(metric_data)
                            bin_stats[f'{metric}_{period}_mean'] = metric_data.mean()
                        else:
                            bin_stats[f'{metric}_{period}_count'] = 0
                            bin_stats[f'{metric}_{period}_mean'] = np.nan
            
            bin_results[bin_name] = bin_stats
    
    # Display the tables
    for period in periods:
        period_display = period.upper().replace('_', '-')
        print(f"\n🔍 {title_prefix}{period_display} FORWARD PERFORMANCE BY DEVIATION BIN:")
        print("-" * 85)
        print(f"{'Deviation Bin':>15} | {'Count':>6} | {'Avg Return':>10} | {'Avg Max Gain':>12} | {'Avg Max DD':>10}")
        print("-" * 85)
        
        for bin_name in bin_order:
            if bin_name in bin_results:
                stats = bin_results[bin_name]
                count = stats.get(f'forward_return_{period}_count', 0)
                
                if count > 0:
                    avg_return = stats.get(f'forward_return_{period}_mean', 0)
                    avg_max_gain = stats.get(f'max_gain_{period}_mean', 0)
                    avg_max_dd = stats.get(f'max_drawdown_{period}_mean', 0)
                    
                    print(f"{bin_name:>15} | {count:>6,} | {avg_return:>9.1f}% | {avg_max_gain:>11.1f}% | {avg_max_dd:>9.1f}%")
                else:
                    print(f"{bin_name:>15} | {count:>6,} | {'N/A':>9} | {'N/A':>11} | {'N/A':>9}")
    
    return bin_results

def analyze_deviation_bins():
    """
    Analyze deviation bins overall and by 5-year periods
    """
    print("DEVIATION BIN ANALYSIS BY PERIOD")
    print("=" * 50)
    
    # Load the forward performance dataset
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    input_file = os.path.join(results_dir, "IWLS_WITH_FORWARD_PERFORMANCE.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Forward performance file not found at {input_file}")
        return None
    
    print(f"📁 Loading: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Successfully loaded {len(df):,} records")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Filter and prepare data
    analysis_data = df.dropna(subset=['price_deviation_from_trend']).copy()
    print(f"📊 Records with deviation data: {len(analysis_data):,}")
    
    analysis_data['deviation_bin'] = analysis_data['price_deviation_from_trend'].apply(assign_deviation_bin)
    analysis_data['five_year_period'] = analysis_data['date'].apply(assign_5_year_period)
    
    # 1. OVERALL DEVIATION BIN ANALYSIS
    print(f"\n🌍 OVERALL DATASET - DEVIATION BIN ANALYSIS")
    print("=" * 60)
    
    bin_order = [">+50%", "+40% to +50%", "+30% to +40%", "+20% to +30%", "+10% to +20%",
                 "-10% to +10%", "-20% to -10%", "-30% to -20%", "-40% to -30%", 
                 "-50% to -40%", "<-50%"]
    
    # Show overall distribution
    bin_counts = analysis_data['deviation_bin'].value_counts()
    print(f"\n📈 OVERALL DEVIATION BIN DISTRIBUTION:")
    total_records = len(analysis_data)
    for bin_name in bin_order:
        if bin_name in bin_counts:
            count = bin_counts[bin_name]
            percentage = (count / total_records) * 100
            print(f"   {bin_name:>15}: {count:>8,} records ({percentage:>5.1f}%)")
    
    # Display overall deviation bin tables
    overall_results = display_deviation_bin_tables(analysis_data, "OVERALL - ")
    
    # 2. DEVIATION BIN ANALYSIS FOR EACH 5-YEAR PERIOD
    period_counts = analysis_data['five_year_period'].value_counts().sort_index()
    
    print(f"\n📅 5-YEAR PERIOD BREAKDOWN:")
    for period_name, count in period_counts.items():
        percentage = (count / len(analysis_data)) * 100
        print(f"   {period_name:>12}: {count:>8,} records ({percentage:>5.1f}%)")
    
    period_results = {}
    
    for period_name in sorted(period_counts.index):
        print(f"\n" + "="*80)
        print(f"🕐 {period_name} - DEVIATION BIN ANALYSIS")
        print("="*80)
        
        period_data = analysis_data[analysis_data['five_year_period'] == period_name]
        
        # Show distribution for this period
        period_bin_counts = period_data['deviation_bin'].value_counts()
        print(f"\n📈 {period_name} DEVIATION BIN DISTRIBUTION:")
        for bin_name in bin_order:
            if bin_name in period_bin_counts:
                count = period_bin_counts[bin_name]
                percentage = (count / len(period_data)) * 100
                print(f"   {bin_name:>15}: {count:>8,} records ({percentage:>5.1f}%)")
        
        # Display deviation bin tables for this period
        period_bin_results = display_deviation_bin_tables(period_data, f"{period_name} - ")
        period_results[period_name] = period_bin_results
    
    # 3. SAVE RESULTS
    print(f"\n💾 SAVING RESULTS...")
    
    all_results = []
    
    # Overall results
    for bin_name, stats in overall_results.items():
        row = stats.copy()
        row['period_name'] = 'Overall'
        all_results.append(row)
    
    # Period results
    for period_name, period_bins in period_results.items():
        for bin_name, stats in period_bins.items():
            row = stats.copy()
            row['period_name'] = period_name
            all_results.append(row)
    
    results_df = pd.DataFrame(all_results)
    
    output_file = os.path.join(results_dir, "DEVIATION_BINS_BY_PERIOD.csv")
    results_df.to_csv(output_file, index=False)
    
    print(f"✅ Results saved: {output_file}")
    print(f"\n✨ DEVIATION BIN ANALYSIS COMPLETE!")
    print(f"📊 You now have deviation bin performance for overall + each 5-year period")
    
    return results_df

if __name__ == "__main__":
    results = analyze_deviation_bins()