import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RollingRangeDecileBacktester:
    """
    Test wave theory across all rolling range deciles:
    
    Theory: If rolling range predicts performance, we should see:
    - Decile 1 (highest ranges): Best returns, fastest profit targets
    - Decile 2-9: Gradually declining performance  
    - Decile 10 (lowest ranges): Worst returns, slowest profit targets
    
    Tests 9 separate backtests:
    - Group 1: Assets ranked 1-9 (highest rolling ranges)
    - Group 2: Assets ranked 10-18  
    - Group 3: Assets ranked 19-27
    - ...
    - Group 9: Assets ranked 73-81 (lowest rolling ranges)
    """
    
    def __init__(self, initial_capital=10000, num_positions=9, profit_target=15.0, max_hold_days=200):
        self.initial_capital = initial_capital
        self.num_positions = num_positions
        self.profit_target = profit_target
        self.max_hold_days = max_hold_days
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def load_data(self):
        """Load the IWLS data with wave analysis"""
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        input_file = os.path.join(results_dir, "IWLS_WITH_WAVE_ANALYSIS.csv")
        
        if not os.path.exists(input_file):
            print(f"❌ Error: Wave data file not found at {input_file}")
            return None
        
        print(f"📂 Loading wave data: {input_file}")
        
        try:
            df = pd.read_csv(input_file)
            df['date'] = pd.to_datetime(df['date'])
            print(f"✅ Successfully loaded {len(df):,} records")
            
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def load_spy_data(self):
        """Load SPY data for comparison"""
        data_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/DATA"
        spy_files = [
            os.path.join(data_dir, "BATS_SPY, 1W-3_daily.csv"),
            os.path.join(data_dir, "BATS_SPY, 1W-3.csv")
        ]
        
        for spy_file in spy_files:
            if os.path.exists(spy_file):
                try:
                    spy_df = pd.read_csv(spy_file)
                    
                    if 'time' in spy_df.columns:
                        spy_df['date'] = pd.to_datetime(spy_df['time'], unit='s')
                    else:
                        spy_df['date'] = pd.to_datetime(spy_df['date'])
                    
                    spy_df = spy_df[['date', 'close']].rename(columns={'close': 'spy_price'})
                    spy_df = spy_df.sort_values('date').dropna()
                    
                    print(f"✅ SPY data loaded: {len(spy_df):,} records")
                    return spy_df
                except Exception as e:
                    continue
        
        print("⚠️ Warning: No SPY data file found")
        return None
    
    def prepare_data(self, df):
        """Prepare data starting from 1994"""
        print("🔧 Preparing trading data...")
        
        # Keep only records with wave analysis and prices
        clean_data = df.dropna(subset=['rolling_range_pct_6_month', 'price']).copy()
        
        # Filter to start from 1994 onwards
        start_date = pd.to_datetime('1994-01-01')
        clean_data = clean_data[clean_data['date'] >= start_date].copy()
        
        # Sort chronologically
        clean_data = clean_data.sort_values(['date', 'asset']).reset_index(drop=True)
        
        print(f"📊 Clean records (1994+): {len(clean_data):,}")
        if len(clean_data) > 0:
            print(f"📅 Date range: {clean_data['date'].min().strftime('%Y-%m-%d')} to {clean_data['date'].max().strftime('%Y-%m-%d')}")
            print(f"🏢 Unique assets: {clean_data['asset'].nunique()}")
        
        return clean_data
    
    def find_rolling_range_decile_assets(self, df, current_date, decile_group, lookback_days=7):
        """
        Find assets in specific rolling range decile
        decile_group: 1 = highest ranges (ranks 1-9), 2 = next highest (ranks 10-18), etc.
        """
        start_date = current_date - timedelta(days=lookback_days)
        end_date = current_date + timedelta(days=lookback_days)
        
        recent_data = df[
            (df['date'] >= start_date) & 
            (df['date'] <= end_date)
        ]
        
        if len(recent_data) == 0:
            return []
        
        # Get latest record for each asset
        latest_by_asset = recent_data.loc[
            recent_data.groupby('asset')['date'].idxmax()
        ].reset_index(drop=True)
        
        # Sort by rolling range (highest to lowest)
        sorted_by_range = latest_by_asset.sort_values('rolling_range_pct_6_month', ascending=False)
        
        # Calculate start and end positions for this decile group
        start_rank = (decile_group - 1) * self.num_positions
        end_rank = start_rank + self.num_positions
        
        # Get assets in this decile
        decile_assets = sorted_by_range.iloc[start_rank:end_rank]
        
        opportunities = []
        for _, row in decile_assets.iterrows():
            opportunities.append({
                'asset': row['asset'],
                'entry_price': row['price'],
                'rolling_range': row['rolling_range_pct_6_month'],
                'range_rank': start_rank + len(opportunities) + 1,  # Actual rank
                'entry_date': row['date']
            })
        
        return opportunities
    
    def get_current_prices(self, df, assets, current_date, lookback_days=7):
        """Get current prices for assets"""
        start_date = current_date - timedelta(days=lookback_days)
        end_date = current_date + timedelta(days=lookback_days)
        
        prices = {}
        
        for asset in assets:
            asset_data = df[
                (df['asset'] == asset) & 
                (df['date'] >= start_date) & 
                (df['date'] <= end_date)
            ]
            
            if len(asset_data) > 0:
                latest = asset_data.loc[asset_data['date'].idxmax()]
                prices[asset] = latest['price']
            else:
                prices[asset] = None
        
        return prices
    
    def calculate_portfolio_value(self, positions, current_prices):
        """Calculate current portfolio value"""
        total_value = 0
        
        for position in positions:
            asset = position['asset']
            shares = position['shares']
            current_price = current_prices.get(asset)
            
            if current_price is not None:
                position_value = shares * current_price
                total_value += position_value
            else:
                total_value += position['investment']
        
        return total_value
    
    def interpolate_spy_daily(self, spy_df, all_dates):
        """Interpolate SPY prices to daily data"""
        if spy_df is None:
            return None
        
        daily_df = pd.DataFrame({'date': all_dates})
        merged_df = pd.merge(daily_df, spy_df, on='date', how='left')
        merged_df['spy_price'] = merged_df['spy_price'].fillna(method='ffill').fillna(method='bfill')
        merged_df = merged_df.dropna(subset=['spy_price'])
        
        return merged_df
    
    def get_spy_price_on_date(self, spy_daily, target_date, lookback_days=7):
        """Get SPY price on or near a specific date"""
        if spy_daily is None:
            return None
        
        exact_match = spy_daily[spy_daily['date'] == target_date]
        if len(exact_match) > 0:
            return exact_match.iloc[0]['spy_price']
        
        start_date = target_date - timedelta(days=lookback_days)
        end_date = target_date + timedelta(days=lookback_days)
        
        nearby_data = spy_daily[
            (spy_daily['date'] >= start_date) & 
            (spy_daily['date'] <= end_date)
        ]
        
        if len(nearby_data) > 0:
            nearby_data['date_diff'] = abs((nearby_data['date'] - target_date).dt.days)
            closest = nearby_data.loc[nearby_data['date_diff'].idxmin()]
            return closest['spy_price']
        
        return None
    
    def backtest_single_decile(self, data, spy_daily, decile_group):
        """Run backtest for a single rolling range decile group"""
        
        start_date = data['date'].min()
        all_dates = sorted(data['date'].unique())
        
        print(f"\n🔥 DECILE GROUP {decile_group} BACKTEST")
        print(f"    📊 Range ranks: {(decile_group-1)*self.num_positions + 1} to {decile_group*self.num_positions}")
        print(f"    🎯 Strategy: Buy {self.num_positions} assets ranked {(decile_group-1)*self.num_positions + 1}-{decile_group*self.num_positions} by rolling range")
        
        # SPY setup
        spy_start_price = self.get_spy_price_on_date(spy_daily, start_date)
        spy_shares = self.initial_capital / spy_start_price if spy_start_price else 0
        
        # Initialize tracking
        current_capital = self.initial_capital
        active_positions = []
        portfolio_number = 0
        completed_portfolios = []
        profit_exits = 0
        time_exits = 0
        
        # Start first portfolio
        initial_opportunities = self.find_rolling_range_decile_assets(data, start_date, decile_group)
        
        if len(initial_opportunities) == 0:
            print(f"    ❌ No opportunities found for decile {decile_group}")
            return None
        
        # Enter first portfolio
        investment_per_position = self.initial_capital / len(initial_opportunities)
        
        for opp in initial_opportunities:
            position = {
                'asset': opp['asset'],
                'entry_date': opp['entry_date'],
                'entry_price': opp['entry_price'],
                'rolling_range': opp['rolling_range'],
                'range_rank': opp['range_rank'],
                'investment': investment_per_position,
                'shares': investment_per_position / opp['entry_price']
            }
            active_positions.append(position)
        
        portfolio_number = 1
        portfolio_entry_date = start_date
        total_invested = sum(pos['investment'] for pos in active_positions)
        current_capital -= total_invested
        
        avg_range = np.mean([pos['rolling_range'] for pos in active_positions])
        print(f"    🌊 Avg rolling range: {avg_range:.1f}%")
        print(f"    📈 Assets: {[pos['asset'] for pos in active_positions]}")
        
        # Main backtest loop
        progress_counter = 0
        for current_date in all_dates:
            progress_counter += 1
            
            # Progress indicator every 1000 days
            if progress_counter % 1000 == 0:
                print(f"    📅 Progress: {progress_counter:,}/{len(all_dates):,} days")
            
            if len(active_positions) > 0:
                asset_list = [pos['asset'] for pos in active_positions]
                current_prices = self.get_current_prices(data, asset_list, current_date)
                portfolio_value = self.calculate_portfolio_value(active_positions, current_prices)
                
                total_invested = sum(pos['investment'] for pos in active_positions)
                portfolio_return = ((portfolio_value / total_invested) - 1) * 100
                hold_days = (current_date - portfolio_entry_date).days
                
                # Check exit conditions
                profit_target_hit = portfolio_return >= self.profit_target
                max_hold_hit = hold_days >= self.max_hold_days
                
                if profit_target_hit or max_hold_hit:
                    # Calculate SPY performance
                    spy_entry_price = self.get_spy_price_on_date(spy_daily, portfolio_entry_date)
                    spy_current_price = self.get_spy_price_on_date(spy_daily, current_date)
                    spy_period_return = 0.0
                    
                    if spy_entry_price and spy_current_price:
                        spy_period_return = ((spy_current_price / spy_entry_price) - 1) * 100
                    
                    # Determine exit reason
                    if profit_target_hit:
                        exit_reason = f"PROFIT_TARGET (+{portfolio_return:.1f}%)"
                        profit_exits += 1
                    else:
                        exit_reason = f"MAX_HOLD ({hold_days} days, {portfolio_return:+.1f}%)"
                        time_exits += 1
                    
                    # EXIT CURRENT PORTFOLIO
                    current_capital += portfolio_value
                    
                    # Record completed portfolio
                    completed_portfolio = {
                        'decile_group': decile_group,
                        'portfolio_number': portfolio_number,
                        'entry_date': portfolio_entry_date,
                        'exit_date': current_date,
                        'hold_days': hold_days,
                        'invested': total_invested,
                        'exit_value': portfolio_value,
                        'return_pct': portfolio_return,
                        'exit_reason': exit_reason,
                        'avg_rolling_range': np.mean([pos['rolling_range'] for pos in active_positions]),
                        'avg_range_rank': np.mean([pos['range_rank'] for pos in active_positions]),
                        'assets': [pos['asset'] for pos in active_positions],
                        'spy_entry_price': spy_entry_price,
                        'spy_exit_price': spy_current_price,
                        'spy_period_return': spy_period_return,
                        'excess_return': portfolio_return - spy_period_return,
                        'outperformed_spy': portfolio_return > spy_period_return
                    }
                    
                    completed_portfolios.append(completed_portfolio)
                    
                    # ENTER NEW PORTFOLIO
                    new_opportunities = self.find_rolling_range_decile_assets(data, current_date, decile_group)
                    
                    if len(new_opportunities) > 0 and current_capital > 1000:
                        investment_per_position = current_capital / len(new_opportunities)
                        
                        new_positions = []
                        for opp in new_opportunities:
                            position = {
                                'asset': opp['asset'],
                                'entry_date': current_date,
                                'entry_price': opp['entry_price'],
                                'rolling_range': opp['rolling_range'],
                                'range_rank': opp['range_rank'],
                                'investment': investment_per_position,
                                'shares': investment_per_position / opp['entry_price']
                            }
                            new_positions.append(position)
                        
                        active_positions = new_positions
                        portfolio_number += 1
                        portfolio_entry_date = current_date
                        total_new_invested = sum(pos['investment'] for pos in active_positions)
                        current_capital -= total_new_invested
                        
                        asset_list = [pos['asset'] for pos in active_positions]
                        current_prices = self.get_current_prices(data, asset_list, current_date)
                        portfolio_value = self.calculate_portfolio_value(active_positions, current_prices)
                    else:
                        active_positions = []
                        portfolio_value = 0
            else:
                portfolio_value = 0
        
        # Final calculations
        final_value = current_capital
        if len(active_positions) > 0:
            asset_list = [pos['asset'] for pos in active_positions]
            final_prices = self.get_current_prices(data, asset_list, all_dates[-1])
            final_portfolio_value = self.calculate_portfolio_value(active_positions, final_prices)
            final_value += final_portfolio_value
        
        final_spy_value = self.initial_capital
        final_spy_price = self.get_spy_price_on_date(spy_daily, all_dates[-1])
        
        if spy_start_price and final_spy_price:
            final_spy_value = spy_shares * final_spy_price
        
        # Calculate summary metrics
        total_return = ((final_value / self.initial_capital) - 1) * 100
        spy_total_return = ((final_spy_value / self.initial_capital) - 1) * 100
        excess_return = total_return - spy_total_return
        
        avg_portfolio_return = np.mean([p['return_pct'] for p in completed_portfolios]) if completed_portfolios else 0
        avg_hold_days = np.mean([p['hold_days'] for p in completed_portfolios]) if completed_portfolios else 0
        avg_rolling_range = np.mean([p['avg_rolling_range'] for p in completed_portfolios]) if completed_portfolios else 0
        
        print(f"    ✅ Decile {decile_group} Complete:")
        print(f"       Final Value: ${final_value:,.2f} ({total_return:+.1f}%)")
        print(f"       SPY: ${final_spy_value:,.2f} ({spy_total_return:+.1f}%)")
        print(f"       Excess: {excess_return:+.1f}%")
        print(f"       Portfolios: {len(completed_portfolios)} | Profit Exits: {profit_exits} | Time Exits: {time_exits}")
        print(f"       Avg Hold: {avg_hold_days:.0f} days | Avg Range: {avg_rolling_range:.1f}%")
        
        return {
            'decile_group': decile_group,
            'completed_portfolios': completed_portfolios,
            'final_value': final_value,
            'total_return': total_return,
            'spy_total_return': spy_total_return,
            'excess_return': excess_return,
            'profit_exits': profit_exits,
            'time_exits': time_exits,
            'avg_portfolio_return': avg_portfolio_return,
            'avg_hold_days': avg_hold_days,
            'avg_rolling_range': avg_rolling_range,
            'total_portfolios': len(completed_portfolios)
        }
    
    def run_all_decile_backtests(self):
        """Run backtests for all rolling range deciles"""
        
        # Load and prepare data
        df = self.load_data()
        if df is None:
            return None
        
        data = self.prepare_data(df)
        spy_df = self.load_spy_data()
        
        start_date = data['date'].min()
        end_date = data['date'].max()
        all_dates = sorted(data['date'].unique())
        
        print(f"\n🚀 ROLLING RANGE DECILE BACKTEST ANALYSIS")
        print("="*80)
        print(f"💡 THEORY TEST: Higher rolling ranges should produce better returns")
        print(f"📊 Testing {self.num_positions} asset groups across rolling range spectrum")
        print(f"💰 Initial capital per group: ${self.initial_capital:,.2f}")
        print(f"🎯 Exit: +{self.profit_target}% profit OR {self.max_hold_days} days max hold")
        print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"📁 Run ID: {self.run_timestamp}")
        
        # Prepare SPY data
        spy_daily = self.interpolate_spy_daily(spy_df, all_dates)
        
        # Determine number of decile groups based on asset count
        sample_date = all_dates[len(all_dates)//2]  # Use middle date for asset count
        sample_data = data[data['date'] == sample_date]
        total_assets = len(sample_data)
        num_decile_groups = min(9, total_assets // self.num_positions)  # Max 9 groups or until we run out of assets
        
        print(f"🏢 Total assets available: ~{total_assets}")
        print(f"📊 Testing {num_decile_groups} decile groups of {self.num_positions} assets each")
        
        # Run backtests for each decile group
        all_results = []
        
        for decile_group in range(1, num_decile_groups + 1):
            print(f"\n" + "="*60)
            print(f"🔍 TESTING DECILE GROUP {decile_group}/{num_decile_groups}")
            print(f"📊 Rolling range ranks {(decile_group-1)*self.num_positions + 1} to {decile_group*self.num_positions}")
            
            result = self.backtest_single_decile(data, spy_daily, decile_group)
            if result:
                all_results.append(result)
        
        return {
            'decile_results': all_results,
            'run_timestamp': self.run_timestamp,
            'num_decile_groups': num_decile_groups,
            'total_assets': total_assets
        }
    
    def analyze_decile_results(self, results):
        """Analyze and compare results across all deciles"""
        if not results or not results['decile_results']:
            return
        
        decile_results = results['decile_results']
        
        print(f"\n📊 ROLLING RANGE DECILE ANALYSIS RESULTS")
        print("="*80)
        print(f"🕐 Run ID: {results['run_timestamp']}")
        print(f"📈 THEORY TEST: Do higher rolling ranges produce better returns?")
        
        # Create summary table
        print(f"\n📋 DECILE PERFORMANCE SUMMARY:")
        print(f"{'Group':>5} | {'Ranks':>10} | {'Avg Range':>10} | {'Final Value':>12} | {'Total Return':>12} | {'Excess vs SPY':>12} | {'Portfolios':>10} | {'Profit Exits':>12} | {'Avg Hold Days':>12}")
        print("-" * 140)
        
        # Sort results by decile group
        sorted_results = sorted(decile_results, key=lambda x: x['decile_group'])
        
        for result in sorted_results:
            group = result['decile_group']
            start_rank = (group - 1) * self.num_positions + 1
            end_rank = group * self.num_positions
            avg_range = result['avg_rolling_range']
            final_value = result['final_value']
            total_return = result['total_return']
            excess_return = result['excess_return']
            total_portfolios = result['total_portfolios']
            profit_exits = result['profit_exits']
            avg_hold_days = result['avg_hold_days']
            
            print(f"{group:>5} | {start_rank:>3}-{end_rank:>2} | {avg_range:>9.1f}% | ${final_value:>10,.0f} | {total_return:>+10.1f}% | {excess_return:>+10.1f}% | {total_portfolios:>10} | {profit_exits:>12} | {avg_hold_days:>11.0f}")
        
        # Statistical analysis
        print(f"\n🔍 STATISTICAL ANALYSIS:")
        
        # Test correlation between decile rank and performance
        decile_ranks = [r['decile_group'] for r in sorted_results]
        total_returns = [r['total_return'] for r in sorted_results]
        excess_returns = [r['excess_return'] for r in sorted_results]
        profit_exit_rates = [r['profit_exits'] / r['total_portfolios'] if r['total_portfolios'] > 0 else 0 for r in sorted_results]
        avg_ranges = [r['avg_rolling_range'] for r in sorted_results]
        
        # Calculate correlations
        if len(sorted_results) > 2:
            range_return_corr = np.corrcoef(avg_ranges, total_returns)[0, 1] if len(avg_ranges) > 1 else 0
            range_excess_corr = np.corrcoef(avg_ranges, excess_returns)[0, 1] if len(avg_ranges) > 1 else 0
            range_profit_corr = np.corrcoef(avg_ranges, profit_exit_rates)[0, 1] if len(avg_ranges) > 1 else 0
            
            print(f"   Rolling Range vs Total Return correlation: {range_return_corr:+.3f}")
            print(f"   Rolling Range vs Excess Return correlation: {range_excess_corr:+.3f}")
            print(f"   Rolling Range vs Profit Exit Rate correlation: {range_profit_corr:+.3f}")
            
            # Theory validation
            print(f"\n🎯 THEORY VALIDATION:")
            if range_return_corr > 0.3:
                print(f"   ✅ STRONG SUPPORT: Higher rolling ranges correlate with better returns ({range_return_corr:+.3f})")
            elif range_return_corr > 0.1:
                print(f"   ⚠️  MODERATE SUPPORT: Some correlation between rolling ranges and returns ({range_return_corr:+.3f})")
            else:
                print(f"   ❌ WEAK SUPPORT: Little correlation between rolling ranges and returns ({range_return_corr:+.3f})")
            
            # Compare best vs worst deciles
            best_decile = sorted_results[0]  # Group 1 (highest ranges)
            worst_decile = sorted_results[-1]  # Last group (lowest ranges)
            
            print(f"\n🏆 BEST vs WORST DECILE COMPARISON:")
            print(f"   Best Decile (Group {best_decile['decile_group']}):")
            print(f"     Avg Range: {best_decile['avg_rolling_range']:.1f}%")
            print(f"     Total Return: {best_decile['total_return']:+.1f}%")
            print(f"     Excess Return: {best_decile['excess_return']:+.1f}%")
            print(f"     Profit Exit Rate: {best_decile['profit_exits']}/{best_decile['total_portfolios']} ({best_decile['profit_exits']/best_decile['total_portfolios']*100:.1f}%)")
            
            print(f"   Worst Decile (Group {worst_decile['decile_group']}):")
            print(f"     Avg Range: {worst_decile['avg_rolling_range']:.1f}%")
            print(f"     Total Return: {worst_decile['total_return']:+.1f}%")
            print(f"     Excess Return: {worst_decile['excess_return']:+.1f}%")
            print(f"     Profit Exit Rate: {worst_decile['profit_exits']}/{worst_decile['total_portfolios']} ({worst_decile['profit_exits']/worst_decile['total_portfolios']*100:.1f}%)")
            
            performance_gap = best_decile['total_return'] - worst_decile['total_return']
            print(f"   Performance Gap: {performance_gap:+.1f}% (Best - Worst)")
        
        return results
    
    def save_decile_results(self, results):
        """Save all decile results with unique timestamp"""
        if not results:
            return
        
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        timestamp = results['run_timestamp']
        
        # Save summary of all deciles
        decile_summary = []
        for result in results['decile_results']:
            summary_row = {
                'decile_group': result['decile_group'],
                'range_ranks_start': (result['decile_group'] - 1) * self.num_positions + 1,
                'range_ranks_end': result['decile_group'] * self.num_positions,
                'avg_rolling_range': result['avg_rolling_range'],
                'final_value': result['final_value'],
                'total_return_pct': result['total_return'],
                'spy_total_return_pct': result['spy_total_return'],
                'excess_return_pct': result['excess_return'],
                'total_portfolios': result['total_portfolios'],
                'profit_exits': result['profit_exits'],
                'time_exits': result['time_exits'],
                'profit_exit_rate_pct': (result['profit_exits'] / result['total_portfolios'] * 100) if result['total_portfolios'] > 0 else 0,
                'avg_portfolio_return_pct': result['avg_portfolio_return'],
                'avg_hold_days': result['avg_hold_days']
            }
            decile_summary.append(summary_row)
        
        summary_df = pd.DataFrame(decile_summary)
        summary_file = os.path.join(results_dir, f"ROLLING_RANGE_DECILE_SUMMARY_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Decile summary: {summary_file}")
        
        # Save detailed portfolios for all deciles