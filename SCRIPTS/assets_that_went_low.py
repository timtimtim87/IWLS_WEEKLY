import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
import warnings
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

def analyze_asset_counts_by_deviation_bins():
    """
    Analyze how many unique assets fall into each deviation bin by year
    """
    print("UNIQUE ASSET COUNT ANALYSIS BY DEVIATION BINS")
    print("=" * 60)
    print("Analysis: How many different assets experienced each level of underperformance")
    print()
    
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
    
    # Get unique assets and years
    unique_assets = sorted(negative_data['asset'].unique())
    years = sorted(negative_data['year'].unique())
    
    print(f"🏢 Total unique assets in dataset: {len(unique_assets):,}")
    print(f"📅 Year range: {min(years)} - {max(years)}")
    print(f"📊 Assets with negative deviation records: {', '.join(map(str, unique_assets))}")
    print()
    
    # MAIN ANALYSIS: Asset counts by year and bin
    print("🎯 UNIQUE ASSET COUNTS BY DEVIATION BIN AND YEAR")
    print("=" * 80)
    
    # Create summary data structure
    asset_count_results = []
    
    # Header for the main table
    header = f"{'Year':>6} |"
    for bin_name in target_bins:
        # Shorten bin names for display
        short_name = bin_name.replace("% to ", "-").replace("%", "")
        header += f" {short_name:>9} |"
    header += f" {'Total':>7}"
    
    print(header)
    print("-" * len(header))
    
    # Analyze each year
    for year in years:
        year_data = negative_data[negative_data['year'] == year]
        
        row = f"{year:>6} |"
        year_total_assets = set()
        
        year_results = {'year': year}
        
        for bin_name in target_bins:
            bin_data = year_data[year_data['negative_dev_bin'] == bin_name]
            unique_assets_in_bin = bin_data['asset'].nunique()
            
            # Track all assets for year total
            year_total_assets.update(bin_data['asset'].unique())
            
            row += f" {unique_assets_in_bin:>9,} |"
            
            # Store results
            year_results[bin_name] = unique_assets_in_bin
            
            # Store detailed info for later analysis
            if unique_assets_in_bin > 0:
                asset_count_results.append({
                    'year': year,
                    'deviation_bin': bin_name,
                    'unique_asset_count': unique_assets_in_bin,
                    'total_records': len(bin_data),
                    'assets_list': sorted(bin_data['asset'].unique().tolist())
                })
        
        year_total = len(year_total_assets)
        row += f" {year_total:>7,}"
        year_results['total_unique_assets'] = year_total
        
        print(row)
    
    print("-" * len(header))
    
    # SUMMARY STATISTICS
    print(f"\n📈 SUMMARY STATISTICS")
    print("=" * 50)
    
    results_df = pd.DataFrame(asset_count_results)
    
    if len(results_df) > 0:
        print(f"\n🏆 MOST ACTIVE DEVIATION BINS (by total unique assets):")
        bin_totals = results_df.groupby('deviation_bin')['unique_asset_count'].sum().sort_values(ascending=False)
        
        for bin_name, total_count in bin_totals.items():
            avg_per_year = results_df[results_df['deviation_bin'] == bin_name]['unique_asset_count'].mean()
            years_active = (results_df[results_df['deviation_bin'] == bin_name]['unique_asset_count'] > 0).sum()
            
            print(f"   {bin_name:>15}: {total_count:>3,} total assets | {avg_per_year:>4.1f} avg/year | {years_active} years active")
        
        print(f"\n📊 YEARLY TOTALS:")
        yearly_totals = negative_data.groupby('year')['asset'].nunique().sort_index()
        for year, count in yearly_totals.items():
            total_records = len(negative_data[negative_data['year'] == year])
            avg_records_per_asset = total_records / count if count > 0 else 0
            print(f"   {year}: {count:>2,} unique assets | {total_records:>4,} total records | {avg_records_per_asset:>5.1f} records/asset")
    
    # DETAILED BREAKDOWN FOR SPECIFIC BINS
    print(f"\n🔍 DETAILED ASSET BREAKDOWN FOR KEY BINS")
    print("=" * 60)
    
    # Focus on bins with decent activity
    key_bins = ["-45% to -50%", "-50% to -55%", "-55% to -60%"]
    
    for bin_name in key_bins:
        print(f"\n📉 {bin_name} - Asset Details by Year:")
        print("-" * 50)
        
        bin_results = results_df[results_df['deviation_bin'] == bin_name]
        
        if len(bin_results) > 0:
            print(f"{'Year':>6} | {'Count':>5} | {'Records':>7} | Assets")
            print("-" * 50)
            
            for _, row in bin_results.sort_values('year').iterrows():
                year = int(row['year'])
                count = int(row['unique_asset_count'])
                records = int(row['total_records'])
                assets = ', '.join(map(str, row['assets_list']))
                
                print(f"{year:>6} | {count:>5,} | {records:>7,} | {assets}")
        else:
            print("   No records found for this bin")
    
    # CONCENTRATION ANALYSIS
    print(f"\n🎯 CONCENTRATION ANALYSIS")
    print("=" * 40)
    
    print(f"\nAsset concentration by year (how many assets dominate the opportunities):")
    print(f"{'Year':>6} | {'Total Assets':>12} | {'Top Asset %':>11} | {'Top 3 Assets %':>13}")
    print("-" * 50)
    
    concentration_results = []
    
    for year in years:
        year_data = negative_data[negative_data['year'] == year]
        
        if len(year_data) > 0:
            asset_record_counts = year_data['asset'].value_counts()
            total_records = len(year_data)
            unique_assets = len(asset_record_counts)
            
            # Top asset percentage
            top_asset_pct = (asset_record_counts.iloc[0] / total_records) * 100
            
            # Top 3 assets percentage
            top_3_pct = (asset_record_counts.head(3).sum() / total_records) * 100
            
            print(f"{year:>6} | {unique_assets:>12,} | {top_asset_pct:>10.1f}% | {top_3_pct:>12.1f}%")
            
            concentration_results.append({
                'year': year,
                'total_unique_assets': unique_assets,
                'total_records': total_records,
                'top_asset_percentage': top_asset_pct,
                'top_3_assets_percentage': top_3_pct,
                'most_active_asset': asset_record_counts.index[0],
                'most_active_asset_records': asset_record_counts.iloc[0]
            })
    
    # SAVE RESULTS
    print(f"\n💾 SAVING RESULTS...")
    
    # Save detailed asset count results
    if len(asset_count_results) > 0:
        asset_count_file = os.path.join(results_dir, "ASSET_COUNT_BY_DEVIATION_BINS.csv")
        pd.DataFrame(asset_count_results).to_csv(asset_count_file, index=False)
        print(f"✅ Asset count details saved: {asset_count_file}")
    
    # Save concentration analysis
    if len(concentration_results) > 0:
        concentration_file = os.path.join(results_dir, "ASSET_CONCENTRATION_ANALYSIS.csv")
        pd.DataFrame(concentration_results).to_csv(concentration_file, index=False)
        print(f"✅ Concentration analysis saved: {concentration_file}")
    
    print(f"\n✨ ASSET COUNT ANALYSIS COMPLETE!")
    print(f"🎯 Key Insights:")
    print(f"   📊 Analyzed {len(unique_assets)} unique assets across {len(years)} years")
    print(f"   📉 Tracked asset distribution across 8 deviation severity bins")
    print(f"   🏢 Identified concentration patterns and dominant assets")
    print(f"   📈 Created year-over-year asset participation trends")
    
    return results_df, concentration_results

def show_most_frequent_underperformers():
    """
    Show which assets most frequently appear in severe deviation bins
    """
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    input_file = os.path.join(results_dir, "IWLS_WITH_FORWARD_PERFORMANCE.csv")
    
    if not os.path.exists(input_file):
        print("No data file found for frequency analysis.")
        return
    
    print(f"\n🏆 MOST FREQUENT UNDERPERFORMERS")
    print("-" * 50)
    print("Assets that most often experience severe negative deviations")
    print()
    
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['negative_dev_bin'] = df['price_deviation_from_trend'].apply(assign_negative_deviation_bin)
    
    # Filter to target bins
    target_bins = ["-40% to -45%", "-45% to -50%", "-50% to -55%", "-55% to -60%", 
                   "-60% to -65%", "-65% to -70%", "-70% to -75%", "Below -75%"]
    
    severe_data = df[df['negative_dev_bin'].isin(target_bins)]
    
    if len(severe_data) > 0:
        # Asset frequency analysis
        asset_frequency = severe_data['asset'].value_counts()
        
        print(f"🔥 Assets by frequency of severe underperformance:")
        print(f"{'Asset':>6} | {'Total Records':>13} | {'Years Active':>12} | {'Avg Records/Year':>16}")
        print("-" * 60)
        
        for asset in asset_frequency.head(10).index:
            asset_data = severe_data[severe_data['asset'] == asset]
            total_records = len(asset_data)
            years_active = asset_data['year'].nunique()
            avg_per_year = total_records / years_active
            
            print(f"{asset:>6} | {total_records:>13,} | {years_active:>12,} | {avg_per_year:>15.1f}")
        
        # Worst deviation by asset
        print(f"\n❄️ Worst recorded deviations by asset:")
        print(f"{'Asset':>6} | {'Worst Deviation':>15} | {'Date':>12} | {'12m Return':>11}")
        print("-" * 50)
        
        for asset in asset_frequency.head(5).index:
            asset_data = df[df['asset'] == asset].dropna(subset=['price_deviation_from_trend'])
            if len(asset_data) > 0:
                worst_idx = asset_data['price_deviation_from_trend'].idxmin()
                worst_row = asset_data.loc[worst_idx]
                
                worst_dev = worst_row['price_deviation_from_trend']
                worst_date = worst_row['date'].strftime('%Y-%m-%d')
                forward_return = worst_row.get('forward_return_12_month', np.nan)
                
                return_str = f"{forward_return:.1f}%" if not pd.isna(forward_return) else "N/A"
                
                print(f"{asset:>6} | {worst_dev:>14.1f}% | {worst_date:>12} | {return_str:>10}")

if __name__ == "__main__":
    # Set up output logging
    docs_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/DOCS"
    
    # Create DOCS directory if it doesn't exist
    os.makedirs(docs_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(docs_dir, f"asset_count_deviation_analysis_{timestamp}.txt")
    
    # Set up output logger
    logger = OutputLogger(log_file_path)
    sys.stdout = logger
    
    try:
        # Add header to the log file
        print(f"ASSET COUNT BY DEVIATION BINS ANALYSIS REPORT")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Purpose: Identify how many unique assets fall into each deviation severity bin")
        print("=" * 80)
        print()
        
        # Run the asset count analysis
        results, concentration = analyze_asset_counts_by_deviation_bins()
        
        if results is not None:
            # Show most frequent underperformers
            show_most_frequent_underperformers()
            
            print(f"\n🚀 ASSET COUNT ANALYSIS COMPLETE!")
            print(f"   📊 Use 'ASSET_COUNT_BY_DEVIATION_BINS.csv' for detailed breakdown")
            print(f"   🎯 Use 'ASSET_CONCENTRATION_ANALYSIS.csv' for concentration metrics")
            print(f"   🏢 Understand asset diversification across deviation severities")
            
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
        print(f"\n✅ Asset count analysis complete! Full report saved to:")
        print(f"   📄 {log_file_path}")