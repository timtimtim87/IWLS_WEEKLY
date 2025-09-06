import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def simple_stacking_test():
    """
    Simple test: For each day with D10 assets, compare group max gain vs individual max gains
    """
    print("SIMPLE STACKING TEST")
    print("=" * 30)
    print("Question: Does the group achieve higher max gain than individuals?")
    print()
    
    # Load the enhanced dataset
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    input_file = os.path.join(results_dir, "IWLS_WITH_WAVE_ANALYSIS.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Enhanced dataset not found at {input_file}")
        return None
    
    print(f"📁 Loading data...")
    
    try:
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Loaded {len(df):,} records")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Filter to complete records
    complete_data = df.dropna(subset=['rolling_range_pct_6_month', 'max_gain_12_month', 'price']).copy()
    complete_data = complete_data[complete_data['rolling_range_pct_6_month'] >= 0.1].copy()
    
    # Create D10 classification
    complete_data['range_decile'] = pd.qcut(
        complete_data['rolling_range_pct_6_month'], 
        q=10, 
        labels=range(1, 11),
        precision=1,
        duplicates='drop'
    )
    
    # Filter to only D10 assets
    d10_data = complete_data[complete_data['range_decile'] == 10].copy()
    print(f"📊 D10 records: {len(d10_data):,}")
    
    # Find days with 3+ D10 assets
    daily_d10_counts = d10_data.groupby('date').size()
    trading_days = daily_d10_counts[daily_d10_counts >= 3].index
    print(f"📅 Trading days with 3+ D10 assets: {len(trading_days):,}")
    
    # Simple comparison for each day
    results = []
    
    print(f"\n🔍 Testing each trading day...")
    
    for i, trade_date in enumerate(trading_days):
        if (i + 1) % 500 == 0:
            print(f"   Processed {i + 1:,}/{len(trading_days):,} days...")
        
        # Get D10 assets for this date
        day_assets = d10_data[d10_data['date'] == trade_date]
        
        if len(day_assets) < 3:
            continue
        
        # Individual max gains
        individual_max_gains = day_assets['max_gain_12_month'].tolist()
        best_individual_gain = max(individual_max_gains)
        avg_individual_gain = np.mean(individual_max_gains)
        
        # Group max gain (calculate portfolio peak)
        group_max_gain = calculate_simple_group_max_gain(day_assets, df, trade_date)
        
        # Simple comparison
        group_wins = group_max_gain > best_individual_gain
        group_advantage = group_max_gain - best_individual_gain
        
        results.append({
            'date': trade_date,
            'num_assets': len(day_assets),
            'individual_max_gains': individual_max_gains,
            'best_individual': best_individual_gain,
            'avg_individual': avg_individual_gain,
            'group_max_gain': group_max_gain,
            'group_wins': group_wins,
            'group_advantage': group_advantage
        })
    
    # Convert to DataFrame and analyze
    results_df = pd.DataFrame(results)
    
    print(f"\n📊 SIMPLE RESULTS:")
    print("=" * 40)
    
    total_days = len(results_df)
    group_wins = results_df['group_wins'].sum()
    group_win_rate = (group_wins / total_days) * 100
    
    print(f"Total trading days analyzed: {total_days:,}")
    print(f"Group wins: {group_wins:,} days")
    print(f"Individual wins: {total_days - group_wins:,} days")
    print(f"Group win rate: {group_win_rate:.1f}%")
    
    # Average advantages
    avg_group_gain = results_df['group_max_gain'].mean()
    avg_best_individual = results_df['best_individual'].mean()
    avg_advantage = results_df['group_advantage'].mean()
    
    print(f"\nAverage Performance:")
    print(f"Group max gain: {avg_group_gain:.1f}%")
    print(f"Best individual: {avg_best_individual:.1f}%")
    print(f"Group advantage: {avg_advantage:+.1f}%")
    
    # When group wins, by how much?
    winning_results = results_df[results_df['group_wins']]
    losing_results = results_df[~results_df['group_wins']]
    
    if len(winning_results) > 0:
        avg_win_margin = winning_results['group_advantage'].mean()
        print(f"\nWhen group wins:")
        print(f"Average win margin: +{avg_win_margin:.1f}%")
        print(f"Biggest win margin: +{winning_results['group_advantage'].max():.1f}%")
    
    if len(losing_results) > 0:
        avg_loss_margin = losing_results['group_advantage'].mean()
        print(f"\nWhen group loses:")
        print(f"Average loss margin: {avg_loss_margin:.1f}%")
        print(f"Biggest loss margin: {losing_results['group_advantage'].min():.1f}%")
    
    # Simple answer
    print(f"\n🎯 SIMPLE ANSWER:")
    if group_win_rate > 50:
        print(f"✅ YES: Group beats individuals {group_win_rate:.1f}% of the time")
        print(f"💰 Average group advantage: +{avg_advantage:.1f}%")
        print(f"📊 STACKING WORKS!")
    else:
        print(f"❌ NO: Individuals beat group {100-group_win_rate:.1f}% of the time")
        print(f"💰 Average individual advantage: +{-avg_advantage:.1f}%")
        print(f"📊 CONCENTRATION WORKS!")
    
    # Save simple results
    output_file = os.path.join(results_dir, "SIMPLE_STACKING_TEST.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved: {output_file}")
    
    return results_df

def calculate_simple_group_max_gain(day_assets, full_df, trade_date):
    """
    Calculate max gain for equal-weighted group over 12 months
    """
    asset_list = day_assets['asset'].tolist()
    entry_prices = dict(zip(day_assets['asset'], day_assets['price']))
    
    # Get 12 months of future data
    end_date = trade_date + timedelta(days=365)
    future_data = full_df[
        (full_df['asset'].isin(asset_list)) & 
        (full_df['date'] > trade_date) & 
        (full_df['date'] <= end_date) &
        (full_df['price'].notna())
    ]
    
    if len(future_data) == 0:
        return 0.0
    
    # Track portfolio value daily
    max_portfolio_gain = 0.0
    
    for date in future_data['date'].unique():
        day_data = future_data[future_data['date'] == date]
        
        # Calculate portfolio return for this day
        portfolio_return = 0.0
        valid_assets = 0
        
        for asset in asset_list:
            asset_data = day_data[day_data['asset'] == asset]
            if len(asset_data) > 0:
                current_price = asset_data['price'].iloc[0]
                entry_price = entry_prices[asset]
                asset_return = (current_price / entry_price) - 1
                portfolio_return += asset_return / len(asset_list)  # Equal weight
                valid_assets += 1
        
        # Only use if we have data for most assets
        if valid_assets >= len(asset_list) * 0.7:
            portfolio_gain_pct = portfolio_return * 100
            max_portfolio_gain = max(max_portfolio_gain, portfolio_gain_pct)
    
    return max_portfolio_gain

def main():
    """
    Run the simple stacking test
    """
    print("TESTING: Does stacking D10 assets create higher peaks?")
    print()
    
    results = simple_stacking_test()
    
    if results is not None:
        print(f"\n✨ TEST COMPLETE!")
        print("Now you know whether to stack or concentrate!")
    
    return results

if __name__ == "__main__":
    results = main()