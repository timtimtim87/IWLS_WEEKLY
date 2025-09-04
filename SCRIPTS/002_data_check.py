import pandas as pd
import numpy as np
import os
from datetime import datetime

def analyze_merged_weekly_data():
    """
    Comprehensive analysis of the merged weekly stock data
    """
    print("WEEKLY STOCK DATA ANALYSIS")
    print("=" * 60)
    
    # Load the merged data
    data_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/DATA"
    file_path = os.path.join(data_dir, "MERGED_WEEKLY_STOCK_DATA.csv")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return
    
    print(f"📁 Loading: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Successfully loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    print()
    
    # Basic dataset info
    print("DATASET OVERVIEW")
    print("-" * 30)
    print(f"Total rows (weeks): {len(df):,}")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    # Calculate total duration
    duration_days = (df['date'].max() - df['date'].min()).days
    duration_years = duration_days / 365.25
    print(f"Duration: {duration_days:,} days ({duration_years:.1f} years)")
    
    # Get asset columns (exclude 'date')
    asset_columns = [col for col in df.columns if col != 'date']
    total_assets = len(asset_columns)
    
    print(f"Total assets: {total_assets}")
    print()
    
    # Check for duplicate asset names
    print("DUPLICATE ASSET CHECK")
    print("-" * 30)
    
    # Check for exact duplicates
    exact_duplicates = []
    seen_assets = set()
    for asset in asset_columns:
        if asset in seen_assets:
            exact_duplicates.append(asset)
        else:
            seen_assets.add(asset)
    
    if exact_duplicates:
        print(f"⚠️  Found {len(exact_duplicates)} exact duplicate asset names:")
        for dup in exact_duplicates:
            print(f"   - {dup}")
    else:
        print("✅ No exact duplicate asset names found")
    
    # Check for similar assets (case-insensitive and with/without common variations)
    print("\nSimilar asset names check:")
    asset_upper = [asset.upper().strip() for asset in asset_columns]
    similar_groups = {}
    
    for i, asset1 in enumerate(asset_upper):
        for j, asset2 in enumerate(asset_upper):
            if i < j and asset1 == asset2:
                if asset1 not in similar_groups:
                    similar_groups[asset1] = []
                similar_groups[asset1].extend([asset_columns[i], asset_columns[j]])
    
    if similar_groups:
        print(f"⚠️  Found {len(similar_groups)} groups of similar asset names:")
        for group_key, group_assets in similar_groups.items():
            unique_assets = list(set(group_assets))
            print(f"   {group_key}: {unique_assets}")
    else:
        print("✅ No similar asset names found")
    
    print()
    
    # Detailed analysis for each asset
    print("INDIVIDUAL ASSET ANALYSIS")
    print("-" * 40)
    
    asset_stats = []
    
    for asset in sorted(asset_columns):
        asset_data = df[asset]
        
        # Basic stats
        total_points = len(asset_data)
        valid_points = asset_data.count()
        missing_points = total_points - valid_points
        completeness_pct = (valid_points / total_points) * 100
        
        # Date range for this asset
        valid_dates = df[asset_data.notna()]['date']
        if len(valid_dates) > 0:
            first_date = valid_dates.min()
            last_date = valid_dates.max()
            asset_duration_days = (last_date - first_date).days
            asset_duration_years = asset_duration_days / 365.25
        else:
            first_date = None
            last_date = None
            asset_duration_days = 0
            asset_duration_years = 0
        
        # Store stats
        asset_stats.append({
            'asset': asset,
            'total_weeks': total_points,
            'valid_weeks': valid_points,
            'missing_weeks': missing_points,
            'completeness_pct': completeness_pct,
            'first_date': first_date,
            'last_date': last_date,
            'duration_days': asset_duration_days,
            'duration_years': asset_duration_years
        })
        
        # Print individual asset info
        print(f"{asset:>8}: {valid_points:>4}/{total_points:>4} weeks ({completeness_pct:>5.1f}%) | "
              f"{asset_duration_years:>5.1f} years | "
              f"Missing: {missing_points:>3}")
        if first_date and last_date:
            print(f"         {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
        else:
            print(f"         No valid data")
        print()
    
    # Summary statistics
    print("SUMMARY STATISTICS")
    print("-" * 30)
    
    stats_df = pd.DataFrame(asset_stats)
    
    print(f"Assets with complete data (100%): {len(stats_df[stats_df['completeness_pct'] == 100])}")
    print(f"Assets with >90% data: {len(stats_df[stats_df['completeness_pct'] > 90])}")
    print(f"Assets with >80% data: {len(stats_df[stats_df['completeness_pct'] > 80])}")
    print(f"Assets with >50% data: {len(stats_df[stats_df['completeness_pct'] > 50])}")
    print(f"Assets with <50% data: {len(stats_df[stats_df['completeness_pct'] < 50])}")
    
    print(f"\nData completeness statistics:")
    print(f"  Average completeness: {stats_df['completeness_pct'].mean():.1f}%")
    print(f"  Median completeness: {stats_df['completeness_pct'].median():.1f}%")
    print(f"  Min completeness: {stats_df['completeness_pct'].min():.1f}%")
    print(f"  Max completeness: {stats_df['completeness_pct'].max():.1f}%")
    
    print(f"\nDuration statistics:")
    valid_durations = stats_df[stats_df['duration_years'] > 0]['duration_years']
    if len(valid_durations) > 0:
        print(f"  Average duration: {valid_durations.mean():.1f} years")
        print(f"  Median duration: {valid_durations.median():.1f} years")
        print(f"  Min duration: {valid_durations.min():.1f} years")
        print(f"  Max duration: {valid_durations.max():.1f} years")
    
    # Assets with most/least missing data
    print(f"\nAssets with most missing data:")
    worst_assets = stats_df.nsmallest(5, 'completeness_pct')
    for _, row in worst_assets.iterrows():
        print(f"  {row['asset']:>8}: {row['missing_weeks']:>3} missing ({row['completeness_pct']:>5.1f}%)")
    
    print(f"\nAssets with least missing data:")
    best_assets = stats_df.nlargest(5, 'completeness_pct')
    for _, row in best_assets.iterrows():
        print(f"  {row['asset']:>8}: {row['missing_weeks']:>3} missing ({row['completeness_pct']:>5.1f}%)")
    
    # Time coverage analysis
    print(f"\nTIME COVERAGE ANALYSIS")
    print("-" * 30)
    
    # Find the date range with most assets having data
    date_asset_counts = []
    for _, row in df.iterrows():
        date = row['date']
        valid_count = sum(1 for col in asset_columns if pd.notna(row[col]))
        date_asset_counts.append({'date': date, 'valid_assets': valid_count})
    
    coverage_df = pd.DataFrame(date_asset_counts)
    coverage_df['year'] = coverage_df['date'].dt.year
    
    # Yearly coverage summary
    yearly_coverage = coverage_df.groupby('year')['valid_assets'].agg(['mean', 'min', 'max']).round(1)
    print("Assets available by year (avg/min/max):")
    for year, row in yearly_coverage.iterrows():
        print(f"  {year}: {row['mean']:>5.1f} / {row['min']:>2.0f} / {row['max']:>2.0f}")
    
    # Save detailed analysis to CSV
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    stats_output = os.path.join(results_dir, "asset_analysis_summary.csv")
    stats_df.to_csv(stats_output, index=False)
    
    coverage_output = os.path.join(results_dir, "time_coverage_analysis.csv")
    coverage_df.to_csv(coverage_output, index=False)
    
    print(f"\n📊 ANALYSIS COMPLETE")
    print(f"   Detailed statistics saved to: {stats_output}")
    print(f"   Time coverage saved to: {coverage_output}")
    
    return stats_df, coverage_df

if __name__ == "__main__":
    asset_stats, coverage_stats = analyze_merged_weekly_data()
    
    if asset_stats is not None:
        print(f"\n🎯 Analysis complete!")
        print(f"   Found {len(asset_stats)} unique assets")
        print(f"   Use the generated CSV files for further analysis")