import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def step1_create_spy_regime_file():
    """
    Step 1: Create a simple file with date and SPY market regime
    """
    print("STEP 1: CREATING SPY MARKET REGIME FILE")
    print("=" * 50)
    
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    spy_file = os.path.join(results_dir, "SPY_DAILY_MARKET_REGIME_DATA.csv")
    
    if not os.path.exists(spy_file):
        print(f"❌ SPY file not found: {spy_file}")
        return None
    
    # Load SPY data
    spy_df = pd.read_csv(spy_file)
    spy_df['date'] = pd.to_datetime(spy_df['date']).dt.date  # Convert to date only, no time
    
    print(f"✅ SPY records loaded: {len(spy_df):,}")
    print(f"📅 Date range: {spy_df['date'].min()} to {spy_df['date'].max()}")
    
    # Filter to records with Z-scores
    spy_clean = spy_df.dropna(subset=['market_z_score_60d']).copy()
    print(f"📊 Records with Z-scores: {len(spy_clean):,}")
    
    # Assign Z-score bins
    def assign_z_bin(z_score):
        if pd.isna(z_score):
            return 'No Data'
        elif z_score > 2:
            return "Z > +2"
        elif z_score > 1:
            return "Z +1 to +2"  
        elif z_score > 0:
            return "Z 0 to +1"
        elif z_score > -1:
            return "Z -1 to 0"
        elif z_score > -2:
            return "Z -2 to -1"
        else:
            return "Z < -2"
    
    spy_clean['z_bin'] = spy_clean['market_z_score_60d'].apply(assign_z_bin)
    
    # Determine trend - simple 20-day comparison
    spy_clean = spy_clean.sort_values('date')
    spy_clean['ma_20d_ago'] = spy_clean['spy_8week_ma'].shift(20)
    spy_clean['trend'] = np.where(
        spy_clean['spy_8week_ma'] > spy_clean['ma_20d_ago'], 
        'UPTREND', 
        'DOWNTREND'
    )
    
    # Create combined regime
    spy_clean['spy_regime'] = spy_clean['trend'] + ' - ' + spy_clean['z_bin']
    
    # Show regime distribution
    regime_counts = spy_clean['spy_regime'].value_counts()
    print(f"\n📊 SPY REGIME DISTRIBUTION:")
    for regime, count in regime_counts.items():
        pct = (count / len(spy_clean)) * 100
        print(f"   {regime:<25}: {count:>6,} ({pct:>4.1f}%)")
    
    # Create simple output file
    spy_regime_simple = spy_clean[['date', 'spy_regime']].copy()
    
    # Save the simple regime file
    output_file = os.path.join(results_dir, "SPY_REGIME_BY_DATE.csv")
    spy_regime_simple.to_csv(output_file, index=False)
    
    print(f"\n💾 SPY regime file saved: {output_file}")
    print(f"📊 Records: {len(spy_regime_simple):,}")
    print(f"📅 Date range: {spy_regime_simple['date'].min()} to {spy_regime_simple['date'].max()}")
    
    return spy_regime_simple

def step2_analyze_stock_performance():
    """
    Step 2: Merge IWLS data with SPY regime and analyze performance
    """
    print("\nSTEP 2: ANALYZING STOCK PERFORMANCE BY SPY REGIME")
    print("=" * 60)
    
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    
    # Load files
    iwls_file = os.path.join(results_dir, "IWLS_WITH_FORWARD_PERFORMANCE.csv")
    spy_regime_file = os.path.join(results_dir, "SPY_REGIME_BY_DATE.csv")
    
    if not os.path.exists(iwls_file):
        print(f"❌ IWLS file not found: {iwls_file}")
        return None
        
    if not os.path.exists(spy_regime_file):
        print(f"❌ SPY regime file not found: {spy_regime_file}")
        return None
    
    # Load data
    iwls_df = pd.read_csv(iwls_file)
    spy_regime_df = pd.read_csv(spy_regime_file)
    
    iwls_df['date'] = pd.to_datetime(iwls_df['date']).dt.date  # Convert to date only
    spy_regime_df['date'] = pd.to_datetime(spy_regime_df['date']).dt.date  # Convert to date only
    
    print(f"✅ IWLS records: {len(iwls_df):,}")
    print(f"✅ SPY regime records: {len(spy_regime_df):,}")
    
    # Filter IWLS to records with stock deviation
    iwls_clean = iwls_df.dropna(subset=['price_deviation_from_trend']).copy()
    print(f"📊 IWLS records with deviation: {len(iwls_clean):,}")
    
    # Assign stock deviation bins
    def assign_stock_bin(deviation):
        if pd.isna(deviation):
            return 'No Data'
        elif deviation >= 50:
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
    
    iwls_clean['stock_dev_bin'] = iwls_clean['price_deviation_from_trend'].apply(assign_stock_bin)
    
    # Merge datasets
    print(f"\n🔗 Merging datasets...")
    merged_data = pd.merge(iwls_clean, spy_regime_df, on='date', how='inner')
    
    print(f"📊 Merged records: {len(merged_data):,}")
    
    if len(merged_data) == 0:
        print("❌ No merged records!")
        # Debug the merge
        iwls_dates = set(iwls_clean['date'])
        spy_dates = set(spy_regime_df['date'])
        overlap = iwls_dates.intersection(spy_dates)
        print(f"Overlapping dates: {len(overlap)}")
        print(f"IWLS sample dates: {sorted(list(iwls_dates))[:5]}")
        print(f"SPY sample dates: {sorted(list(spy_dates))[:5]}")
        return None
    
    # Show regime distribution in merged data
    regime_counts = merged_data['spy_regime'].value_counts()
    print(f"\n📊 SPY REGIME DISTRIBUTION IN MERGED DATA:")
    for regime, count in regime_counts.items():
        pct = (count / len(merged_data)) * 100
        print(f"   {regime:<25}: {count:>8,} ({pct:>4.1f}%)")
    
    # Define analysis parameters
    stock_dev_bins = [">+50%", "+40% to +50%", "+30% to +40%", "+20% to +30%", "+10% to +20%",
                      "-10% to +10%", "-20% to -10%", "-30% to -20%", "-40% to -30%", 
                      "-50% to -40%", "<-50%"]
    
    periods = ['6_month', '12_month', '18_month']
    
    # Sort regimes for display
    z_order = ["Z > +2", "Z +1 to +2", "Z 0 to +1", "Z -1 to 0", "Z -2 to -1", "Z < -2"]
    
    def sort_regimes(regimes):
        uptrends = [r for r in regimes if 'UPTREND' in r]
        downtrends = [r for r in regimes if 'DOWNTREND' in r]
        
        def sort_by_z(regime_list):
            def get_z_order(regime):
                z_part = regime.split(' - ')[1]
                try:
                    return z_order.index(z_part)
                except ValueError:
                    return 999
            return sorted(regime_list, key=get_z_order)
        
        return sort_by_z(uptrends) + sort_by_z(downtrends)
    
    sorted_regimes = sort_regimes(list(regime_counts.index))
    
    # Analyze each regime
    print(f"\n🔍 ANALYZING PERFORMANCE BY SPY REGIME...")
    all_results = []
    
    for regime in sorted_regimes:
        regime_count = regime_counts[regime]
        
        # Skip regimes with too few samples
        if regime_count < 100:
            print(f"\n⚠️  Skipping {regime} - only {regime_count:,} records")
            continue
            
        print(f"\n" + "="*80)
        print(f"📊 {regime}")
        print(f"Total Records: {regime_count:,}")
        print("="*80)
        
        regime_data = merged_data[merged_data['spy_regime'] == regime]
        
        # Show stock deviation distribution
        stock_bin_counts = regime_data['stock_dev_bin'].value_counts()
        print(f"\nStock Deviation Distribution in {regime}:")
        for bin_name in stock_dev_bins:
            if bin_name in stock_bin_counts:
                count = stock_bin_counts[bin_name]
                pct = (count / len(regime_data)) * 100
                print(f"   {bin_name:>15}: {count:>6,} ({pct:>4.1f}%)")
        
        # Performance analysis by period
        for period in periods:
            print(f"\n🔍 {regime} - {period.upper().replace('_', '-')} FORWARD PERFORMANCE:")
            print("-" * 85)
            print(f"{'Stock Dev Bin':>15} | {'Count':>6} | {'Avg Return':>10} | {'Avg Max Gain':>12} | {'Avg Max DD':>10}")
            print("-" * 85)
            
            for stock_bin in stock_dev_bins:
                bin_data = regime_data[regime_data['stock_dev_bin'] == stock_bin]
                
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
                        
                        print(f"{stock_bin:>15} | {count:>6,} | {avg_return:>9.1f}% | {avg_gain:>11.1f}% | {avg_dd:>9.1f}%")
                        
                        # Store results
                        all_results.append({
                            'spy_regime': regime,
                            'stock_deviation_bin': stock_bin,
                            'period': period,
                            'count': count,
                            'avg_forward_return': avg_return,
                            'avg_max_gain': avg_gain,
                            'avg_max_drawdown': avg_dd
                        })
                    else:
                        print(f"{stock_bin:>15} | {'0':>6} | {'N/A':>9} | {'N/A':>11} | {'N/A':>9}")
                else:
                    print(f"{stock_bin:>15} | {'0':>6} | {'N/A':>9} | {'N/A':>11} | {'N/A':>9}")
    
    # Save results
    print(f"\n💾 SAVING RESULTS...")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = os.path.join(results_dir, "SPY_REGIME_STOCK_ANALYSIS.csv")
        results_df.to_csv(output_file, index=False)
        print(f"✅ Results saved: {output_file}")
        print(f"📊 Total result rows: {len(results_df):,}")
    
    print(f"\n✨ ANALYSIS COMPLETE!")
    return merged_data

def main():
    """
    Main function to run both steps
    """
    # Step 1: Create SPY regime file
    spy_regime_data = step1_create_spy_regime_file()
    
    if spy_regime_data is not None:
        # Step 2: Analyze stock performance
        merged_data = step2_analyze_stock_performance()
        
        if merged_data is not None:
            print(f"\n🎯 SUCCESS!")
            print(f"   📊 Final merged dataset: {len(merged_data):,} records")
            print(f"   🎯 Ready to explore regime-specific stock strategies!")

if __name__ == "__main__":
    main()