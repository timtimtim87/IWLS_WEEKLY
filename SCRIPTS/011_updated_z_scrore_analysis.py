import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def create_simple_spy_trends():
    """
    Create SPY market regimes using only simple trend direction (up/down)
    """
    print("SPY TREND ANALYSIS - SIMPLE UP/DOWN TRENDS ONLY")
    print("=" * 60)
    
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    spy_file = os.path.join(results_dir, "SPY_DAILY_MARKET_REGIME_DATA.csv")
    
    if not os.path.exists(spy_file):
        print(f"❌ SPY file not found: {spy_file}")
        return None
    
    # Load SPY data
    spy_df = pd.read_csv(spy_file)
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    
    # Filter and sort data
    spy_clean = spy_df.dropna(subset=['spy_8week_ma']).copy()
    spy_clean = spy_clean.sort_values('date')
    
    print(f"✅ SPY records with moving average data: {len(spy_clean):,}")
    print(f"📅 Date range: {spy_clean['date'].min().strftime('%Y-%m-%d')} to {spy_clean['date'].max().strftime('%Y-%m-%d')}")
    
    # Determine trend direction (only thing we need)
    print(f"\n📈 Calculating trend direction (MA today vs MA yesterday)...")
    spy_clean['ma_previous_day'] = spy_clean['spy_8week_ma'].shift(1)
    spy_clean['spy_trend'] = np.where(
        spy_clean['spy_8week_ma'] > spy_clean['ma_previous_day'], 
        'UPTREND', 
        'DOWNTREND'
    )
    
    # Remove rows where we can't determine trend (first day only)
    spy_clean = spy_clean.dropna(subset=['spy_trend']).copy()
    
    # Convert date back to date-only for merging
    spy_clean['date'] = spy_clean['date'].dt.date
    
    # Show trend distribution
    trend_counts = spy_clean['spy_trend'].value_counts()
    print(f"\n📈 SPY TREND DISTRIBUTION:")
    for trend, count in trend_counts.items():
        pct = (count / len(spy_clean)) * 100
        print(f"   {trend:>12}: {count:>6,} ({pct:>4.1f}%)")
    
    # Save the simple trend file
    spy_trend_simple = spy_clean[['date', 'spy_trend']].copy()
    
    output_file = os.path.join(results_dir, "SPY_TREND_SIMPLE.csv")
    spy_trend_simple.to_csv(output_file, index=False)
    
    print(f"\n💾 Simple SPY trend file saved: {output_file}")
    print(f"📊 Records: {len(spy_trend_simple):,}")
    
    return spy_trend_simple

def analyze_with_simple_trends():
    """
    Analyze IWLS stock performance using simple SPY trends (up/down) only
    """
    print(f"\nANALYZING IWLS PERFORMANCE WITH SIMPLE SPY TRENDS")
    print("=" * 60)
    
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    
    # Load data
    iwls_file = os.path.join(results_dir, "IWLS_WITH_FORWARD_PERFORMANCE.csv")
    spy_trend_file = os.path.join(results_dir, "SPY_TREND_SIMPLE.csv")
    
    if not os.path.exists(iwls_file):
        print(f"❌ IWLS file not found: {iwls_file}")
        return None
    if not os.path.exists(spy_trend_file):
        print(f"❌ SPY trend file not found: {spy_trend_file}")
        return None
    
    iwls_df = pd.read_csv(iwls_file)
    spy_trend_df = pd.read_csv(spy_trend_file)
    
    iwls_df['date'] = pd.to_datetime(iwls_df['date']).dt.date
    spy_trend_df['date'] = pd.to_datetime(spy_trend_df['date']).dt.date
    
    print(f"✅ IWLS records: {len(iwls_df):,}")
    print(f"✅ SPY trend records: {len(spy_trend_df):,}")
    
    # Filter IWLS data and assign stock deviation bins (10% increments)
    iwls_clean = iwls_df.dropna(subset=['price_deviation_from_trend']).copy()
    
    def assign_stock_bin(deviation):
        """IWLS stock deviation bins - 10% increments"""
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
    
    # Show IWLS deviation distribution
    dev_counts = iwls_clean['stock_dev_bin'].value_counts()
    print(f"\n📊 IWLS DEVIATION BIN DISTRIBUTION:")
    stock_dev_order = [
        ">+50%", "+40% to +50%", "+30% to +40%", "+20% to +30%", 
        "+10% to +20%", "-10% to +10%", "-20% to -10%", "-30% to -20%", 
        "-40% to -30%", "-50% to -40%", "<-50%"
    ]
    
    for bin_name in stock_dev_order:
        if bin_name in dev_counts:
            count = dev_counts[bin_name]
            pct = (count / len(iwls_clean)) * 100
            print(f"   {bin_name:>15}: {count:>6,} ({pct:>4.1f}%)")
    
    # Merge datasets
    print(f"\n🔗 Merging datasets...")
    merged_data = pd.merge(iwls_clean, spy_trend_df, on='date', how='inner')
    print(f"📊 Merged records: {len(merged_data):,}")
    
    if len(merged_data) == 0:
        print("❌ No merged records!")
        return None
    
    # Show trend distribution in merged data
    trend_counts = merged_data['spy_trend'].value_counts()
    print(f"\n📊 SPY TREND DISTRIBUTION IN MERGED DATA:")
    
    for trend in ['UPTREND', 'DOWNTREND']:
        if trend in trend_counts:
            count = trend_counts[trend]
            pct = (count / len(merged_data)) * 100
            print(f"   {trend:>12}: {count:>8,} ({pct:>4.1f}%)")
    
    # Define analysis parameters
    stock_dev_bins = stock_dev_order
    periods = ['6_month', '12_month', '18_month']
    
    # Analyze each SPY trend
    print(f"\n🔍 ANALYZING IWLS PERFORMANCE BY SPY TREND...")
    all_results = []
    
    for spy_trend in ['UPTREND', 'DOWNTREND']:
        if spy_trend not in trend_counts:
            continue
            
        trend_count = trend_counts[spy_trend]
        print(f"\n" + "="*80)
        print(f"📊 SPY {spy_trend}")
        print(f"Total IWLS Records: {trend_count:,}")
        print("="*80)
        
        trend_data = merged_data[merged_data['spy_trend'] == spy_trend]
        
        # Show performance breakdown by IWLS deviation bins for each time period
        for period in periods:
            period_label = period.replace('_', '-').upper()
            print(f"\n🎯 {spy_trend} - {period_label} FORWARD PERFORMANCE:")
            print("-" * 90)
            print(f"{'IWLS Dev Bin':>15} | {'Count':>6} | {'Avg Return':>10} | {'Avg Max Gain':>12} | {'Avg Max DD':>10}")
            print("-" * 90)
            
            for stock_bin in stock_dev_bins:
                bin_data = trend_data[trend_data['stock_dev_bin'] == stock_bin]
                
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
                        
                        # Store results
                        all_results.append({
                            'spy_trend': spy_trend,
                            'iwls_deviation_bin': stock_bin,
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
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = os.path.join(results_dir, "SPY_TREND_IWLS_ANALYSIS_SIMPLE.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved: {output_file}")
        print(f"📊 Total result rows: {len(results_df):,}")
        
        # Show summary comparison
        print(f"\n📈 SUMMARY COMPARISON - 12 MONTH AVERAGE RETURNS:")
        print("=" * 70)
        summary_12m = results_df[results_df['period'] == '12_month'].copy()
        
        for stock_bin in stock_dev_bins:
            uptrend_data = summary_12m[(summary_12m['spy_trend'] == 'UPTREND') & 
                                    (summary_12m['iwls_deviation_bin'] == stock_bin)]
            downtrend_data = summary_12m[(summary_12m['spy_trend'] == 'DOWNTREND') & 
                                       (summary_12m['iwls_deviation_bin'] == stock_bin)]
            
            up_return = uptrend_data['avg_forward_return'].iloc[0] if len(uptrend_data) > 0 else np.nan
            down_return = downtrend_data['avg_forward_return'].iloc[0] if len(downtrend_data) > 0 else np.nan
            up_count = uptrend_data['count'].iloc[0] if len(uptrend_data) > 0 else 0
            down_count = downtrend_data['count'].iloc[0] if len(downtrend_data) > 0 else 0
            
            if not pd.isna(up_return) or not pd.isna(down_return):
                up_str = f"{up_return:6.1f}%" if not pd.isna(up_return) else "  N/A"
                down_str = f"{down_return:6.1f}%" if not pd.isna(down_return) else "  N/A"
                print(f"{stock_bin:>15} | Up: {up_str} ({up_count:>4,}) | Down: {down_str} ({down_count:>4,})")
    
    print(f"\n✨ SIMPLE SPY TREND ANALYSIS COMPLETE!")
    return merged_data

def main():
    """Run the simplified SPY trend analysis"""
    print("STARTING SIMPLIFIED SPY TREND ANALYSIS")
    print("Using only SPY UPTREND/DOWNTREND classification")
    print("IWLS deviation bins remain at 10% increments")
    
    # Step 1: Create simple trend file (no z-scores or deviation bins)
    spy_trend_data = create_simple_spy_trends()
    
    if spy_trend_data is not None:
        # Step 2: Analyze IWLS performance by SPY trends
        merged_data = analyze_with_simple_trends()
        
        if merged_data is not None:
            print(f"\n🎯 SIMPLIFIED ANALYSIS COMPLETE!")
            print(f"   📈 SPY classification: UPTREND vs DOWNTREND only")
            print(f"   📊 IWLS deviation bins: 10% increments (unchanged)")
            print(f"   🔍 Clean binary SPY trend analysis")
            print(f"   💡 Focus on core trend impact on IWLS performance!")

if __name__ == "__main__":
    main()