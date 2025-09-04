import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def create_simple_spy_regimes():
    """
    Create SPY market regimes using simple 2% deviation bins + trend direction
    """
    print("SPY REGIME ANALYSIS - SIMPLE 2% DEVIATION BINS")
    print("=" * 60)
    
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    spy_file = os.path.join(results_dir, "SPY_DAILY_MARKET_REGIME_DATA.csv")
    
    if not os.path.exists(spy_file):
        print(f"⛔ SPY file not found: {spy_file}")
        return None
    
    # Load SPY data
    spy_df = pd.read_csv(spy_file)
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    
    # Filter and sort data
    spy_clean = spy_df.dropna(subset=['price_ma_pct']).copy()
    spy_clean = spy_clean.sort_values('date')
    
    print(f"✅ SPY records with price-MA data: {len(spy_clean):,}")
    print(f"📅 Date range: {spy_clean['date'].min().strftime('%Y-%m-%d')} to {spy_clean['date'].max().strftime('%Y-%m-%d')}")
    
    # Show current price-to-MA distribution
    print(f"\n📊 CURRENT PRICE-TO-MA PERCENTAGE DISTRIBUTION:")
    price_ma_stats = spy_clean['price_ma_pct'].describe()
    print(f"   Mean: {price_ma_stats['mean']:>6.2f}%")
    print(f"   Std:  {price_ma_stats['std']:>6.2f}%")
    print(f"   Min:  {price_ma_stats['min']:>6.2f}%")
    print(f"   Max:  {price_ma_stats['max']:>6.2f}%")
    
    # Create deviation bins (2% increments)
    def assign_deviation_bin(price_ma_pct_val):
        """Assign deviation to 2% magnitude bins"""
        if pd.isna(price_ma_pct_val):
            return 'No Data'
        
        if price_ma_pct_val >= 12:
            return ">+12%"
        elif price_ma_pct_val >= 10:
            return "+10% to +12%"
        elif price_ma_pct_val >= 8:
            return "+8% to +10%"
        elif price_ma_pct_val >= 6:
            return "+6% to +8%"
        elif price_ma_pct_val >= 4:
            return "+4% to +6%"
        elif price_ma_pct_val >= 2:
            return "+2% to +4%"
        elif price_ma_pct_val >= 0:
            return "0% to +2%"
        elif price_ma_pct_val >= -2:
            return "-2% to 0%"
        elif price_ma_pct_val >= -4:
            return "-4% to -2%"
        elif price_ma_pct_val >= -6:
            return "-6% to -4%"
        elif price_ma_pct_val >= -8:
            return "-8% to -6%"
        elif price_ma_pct_val >= -10:
            return "-10% to -8%"
        elif price_ma_pct_val >= -12:
            return "-12% to -10%"
        else:
            return "<-12%"
    
    spy_clean['deviation_bin'] = spy_clean['price_ma_pct'].apply(assign_deviation_bin)
    
    # Determine trend direction
    print(f"\n📈 Calculating trend direction...")
    spy_clean['ma_20d_ago'] = spy_clean['spy_8week_ma'].shift(20)
    spy_clean['trend'] = np.where(
        spy_clean['spy_8week_ma'] > spy_clean['ma_20d_ago'], 
        'UPTREND', 
        'DOWNTREND'
    )
    
    # Create combined regime
    spy_clean['spy_regime'] = spy_clean['trend'] + ' - ' + spy_clean['deviation_bin']
    
    # Convert date back to date-only for merging
    spy_clean['date'] = spy_clean['date'].dt.date
    
    # Show deviation bin distribution
    deviation_counts = spy_clean['deviation_bin'].value_counts()
    print(f"\n📊 DEVIATION BIN DISTRIBUTION:")
    dev_bin_order = [
        ">+12%", "+10% to +12%", "+8% to +10%", "+6% to +8%", 
        "+4% to +6%", "+2% to +4%", "0% to +2%", "-2% to 0%", 
        "-4% to -2%", "-6% to -4%", "-8% to -6%", 
        "-10% to -8%", "-12% to -10%", "<-12%"
    ]
    
    for bin_name in dev_bin_order:
        if bin_name in deviation_counts:
            count = deviation_counts[bin_name]
            pct = (count / len(spy_clean)) * 100
            print(f"   {bin_name:>15}: {count:>6,} ({pct:>4.1f}%)")
    
    # Show trend distribution
    trend_counts = spy_clean['trend'].value_counts()
    print(f"\n📈 TREND DISTRIBUTION:")
    for trend, count in trend_counts.items():
        pct = (count / len(spy_clean)) * 100
        print(f"   {trend:>12}: {count:>6,} ({pct:>4.1f}%)")
    
    # Show detailed breakdown by trend and deviation
    print(f"\n🔍 DETAILED BREAKDOWN BY TREND + DEVIATION:")
    
    for trend in ['UPTREND', 'DOWNTREND']:
        trend_data = spy_clean[spy_clean['trend'] == trend]
        print(f"\n   {trend} ({len(trend_data):,} total days):")
        
        trend_dev_counts = trend_data['deviation_bin'].value_counts()
        for bin_name in dev_bin_order:
            if bin_name in trend_dev_counts:
                count = trend_dev_counts[bin_name]
                pct_of_trend = (count / len(trend_data)) * 100
                pct_of_total = (count / len(spy_clean)) * 100
                print(f"     {bin_name:>15}: {count:>5,} ({pct_of_trend:>4.1f}% of {trend.lower()}, {pct_of_total:>4.1f}% of all)")
    
    # Show final regime distribution
    regime_counts = spy_clean['spy_regime'].value_counts()
    print(f"\n📊 SPY REGIME DISTRIBUTION (SIMPLE 2% BINS):")
    
    # Sort regimes logically
    def sort_regimes(regimes):
        uptrends = [r for r in regimes if 'UPTREND' in r]
        downtrends = [r for r in regimes if 'DOWNTREND' in r]
        
        def sort_by_deviation(regime_list):
            def get_dev_order(regime):
                dev_part = regime.split(' - ')[1]
                try:
                    return dev_bin_order.index(dev_part)
                except ValueError:
                    return 999
            return sorted(regime_list, key=get_dev_order)
        
        return sort_by_deviation(uptrends) + sort_by_deviation(downtrends)
    
    sorted_regimes = sort_regimes(list(regime_counts.index))
    
    for regime in sorted_regimes:
        count = regime_counts[regime]
        pct = (count / len(spy_clean)) * 100
        print(f"   {regime:<35}: {count:>6,} ({pct:>4.1f}%)")
    
    # Save the simple regime file
    spy_regime_simple = spy_clean[['date', 'spy_regime', 'deviation_bin', 'trend', 'price_ma_pct']].copy()
    
    output_file = os.path.join(results_dir, "SPY_REGIME_BY_DATE_SIMPLE.csv")
    spy_regime_simple.to_csv(output_file, index=False)
    
    print(f"\n💾 Simple SPY regime file saved: {output_file}")
    print(f"📊 Records: {len(spy_regime_simple):,}")
    
    return spy_regime_simple

def analyze_with_simple_regimes():
    """
    Analyze stock performance using simple 2% deviation + trend SPY regimes
    """
    print(f"\nANALYZING STOCK PERFORMANCE WITH SIMPLE SPY REGIMES")
    print("=" * 60)
    
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    
    # Load data
    iwls_file = os.path.join(results_dir, "IWLS_WITH_FORWARD_PERFORMANCE.csv")
    spy_regime_file = os.path.join(results_dir, "SPY_REGIME_BY_DATE_SIMPLE.csv")
    
    if not os.path.exists(iwls_file):
        print(f"⛔ IWLS file not found: {iwls_file}")
        return None
    if not os.path.exists(spy_regime_file):
        print(f"⛔ SPY regime file not found: {spy_regime_file}")
        return None
    
    iwls_df = pd.read_csv(iwls_file)
    spy_regime_df = pd.read_csv(spy_regime_file)
    
    iwls_df['date'] = pd.to_datetime(iwls_df['date']).dt.date
    spy_regime_df['date'] = pd.to_datetime(spy_regime_df['date']).dt.date
    
    print(f"✅ IWLS records: {len(iwls_df):,}")
    print(f"✅ SPY regime records: {len(spy_regime_df):,}")
    
    # Filter IWLS data and assign stock deviation bins
    iwls_clean = iwls_df.dropna(subset=['price_deviation_from_trend']).copy()
    
    def assign_stock_bin(deviation):
        """Stock deviation bins - 10% increments"""
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
    print(f"📊 IWLS records with deviation: {len(iwls_clean):,}")
    
    # Merge datasets
    print(f"\n🔗 Merging datasets...")
    merged_data = pd.merge(iwls_clean, spy_regime_df, on='date', how='inner')
    print(f"📊 Merged records: {len(merged_data):,}")
    
    if len(merged_data) == 0:
        print("⛔ No merged records!")
        return None
    
    # Show regime distribution in merged data
    regime_counts = merged_data['spy_regime'].value_counts()
    print(f"\n📊 SPY REGIME DISTRIBUTION IN MERGED DATA:")
    
    # Sort for display (2% bins)
    spy_dev_order = [
        ">+12%", "+10% to +12%", "+8% to +10%", "+6% to +8%", 
        "+4% to +6%", "+2% to +4%", "0% to +2%", "-2% to 0%", 
        "-4% to -2%", "-6% to -4%", "-8% to -6%", 
        "-10% to -8%", "-12% to -10%", "<-12%"
    ]
    
    def sort_regimes_for_display(regimes):
        uptrends = [r for r in regimes if 'UPTREND' in r]
        downtrends = [r for r in regimes if 'DOWNTREND' in r]
        
        def sort_by_dev(regime_list):
            def get_dev_order(regime):
                dev_part = regime.split(' - ')[1]
                try:
                    return spy_dev_order.index(dev_part)
                except ValueError:
                    return 999
            return sorted(regime_list, key=get_dev_order)
        
        return sort_by_dev(uptrends) + sort_by_dev(downtrends)
    
    sorted_regimes = sort_regimes_for_display(list(regime_counts.index))
    
    for regime in sorted_regimes:
        count = regime_counts[regime]
        pct = (count / len(merged_data)) * 100
        print(f"   {regime:<35}: {count:>8,} ({pct:>4.1f}%)")
    
    # Define analysis parameters
    stock_dev_bins = [
        ">+50%", "+40% to +50%", "+30% to +40%", "+20% to +30%", 
        "+10% to +20%", "-10% to +10%", "-20% to -10%", "-30% to -20%", 
        "-40% to -30%", "-50% to -40%", "<-50%"
    ]
    
    periods = ['6_month', '12_month', '18_month']
    
    # Analyze each regime (simplified to just UPTREND/DOWNTREND)
    print(f"\n🔍 ANALYZING PERFORMANCE BY SPY REGIME...")
    all_results = []
    
    sorted_regimes = ['UPTREND', 'DOWNTREND']  # Simplified regimes
    
    for regime in sorted_regimes:
        regime_count = regime_counts[regime]
        
        if regime_count < 100:
            print(f"\n⚠️ Skipping {regime} - only {regime_count:,} records")
            continue
            
        print(f"\n" + "="*80)
        print(f"📊 {regime}")
        print(f"Total Records: {regime_count:,}")
        print("="*80)
        
        regime_data = merged_data[merged_data['spy_regime'] == regime]
        
        # Show 12-month performance for this regime
        period = '12_month'
        print(f"\n🔍 {regime} - 12-MONTH FORWARD PERFORMANCE:")
        print("-" * 85)
        print(f"{'Stock Dev Bin':>15} | {'Count':>6} | {'Avg Return':>10} | {'Avg Max Gain':>12} | {'Avg Max DD':>10}")
        print("-" * 85)
        
        for stock_bin in stock_dev_bins:
            bin_data = regime_data[regime_data['stock_dev_bin'] == stock_bin]
            
            if len(bin_data) > 0:
                return_data = bin_data[f'forward_return_{period}'].dropna()
                gain_data = bin_data[f'max_gain_{period}'].dropna()
                dd_data = bin_data[f'max_drawdown_{period}'].dropna()
                
                if len(return_data) > 0:
                    count = len(return_data)
                    avg_return = return_data.mean()
                    avg_gain = gain_data.mean() if len(gain_data) > 0 else np.nan
                    avg_dd = dd_data.mean() if len(dd_data) > 0 else np.nan
                    
                    print(f"{stock_bin:>15} | {count:>6,} | {avg_return:>9.1f}% | {avg_gain:>11.1f}% | {avg_dd:>9.1f}%")
                    
                    # Store results for all periods
                    for p in periods:
                        p_return_data = bin_data[f'forward_return_{p}'].dropna()
                        p_gain_data = bin_data[f'max_gain_{p}'].dropna()
                        p_dd_data = bin_data[f'max_drawdown_{p}'].dropna()
                        
                        if len(p_return_data) > 0:
                            all_results.append({
                                'spy_regime': regime,
                                'stock_deviation_bin': stock_bin,
                                'period': p,
                                'count': len(p_return_data),
                                'avg_forward_return': p_return_data.mean(),
                                'avg_max_gain': p_gain_data.mean() if len(p_gain_data) > 0 else np.nan,
                                'avg_max_drawdown': p_dd_data.mean() if len(p_dd_data) > 0 else np.nan
                            })
                else:
                    print(f"{stock_bin:>15} | {'0':>6} | {'N/A':>9} | {'N/A':>11} | {'N/A':>9}")
            else:
                print(f"{stock_bin:>15} | {'0':>6} | {'N/A':>9} | {'N/A':>11} | {'N/A':>9}")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = os.path.join(results_dir, "SPY_REGIME_STOCK_ANALYSIS_SIMPLE.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved: {output_file}")
        print(f"📊 Total result rows: {len(results_df):,}")
    
    print(f"\n✨ SIMPLE SPY REGIME ANALYSIS COMPLETE!")
    return merged_data

def main():
    """Run the simplified trend-only analysis"""
    print("STARTING SIMPLIFIED TREND ANALYSIS")
    print("Using only UPTREND/DOWNTREND for SPY (no deviation bins)")
    
    # Step 1: Create simple regime file
    spy_regime_data = create_simple_spy_regimes()
    
    if spy_regime_data is not None:
        # Step 2: Analyze with simple regimes
        merged_data = analyze_with_simple_regimes()
        
        if merged_data is not None:
            print(f"\n🎯 SIMPLIFIED ANALYSIS COMPLETE!")
            print(f"   📊 SPY regimes: UPTREND/DOWNTREND only")
            print(f"   🎯 Stock deviation bins: 10% increments")
            print(f"   📈 Clean, simple regime classification")
            print(f"   💡 Focus on trend direction impact only!")

if __name__ == "__main__":
    main()