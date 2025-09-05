import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UnderperformanceBacktester:
    """
    Enhanced backtest strategy:
    1. Find 5 most negatively deviated stocks
    2. Split $10,000 equally amongst them ($2,000 each)
    3. Track portfolio value daily
    4. Exit when portfolio hits +50% total return
    5. Immediately re-enter with new worst 5 stocks
    6. REBALANCING: If sum of top 5 worst deviations > 250% AND it's been 180+ days since last rebalance,
       exit current positions and enter the new worst 5
    """
    
    def __init__(self, initial_capital=10000, num_positions=5, profit_target=50.0, 
                 rebalance_threshold=250.0, rebalance_wait_days=180):
        self.initial_capital = initial_capital
        self.num_positions = num_positions
        self.profit_target = profit_target  # 50% profit target
        self.rebalance_threshold = rebalance_threshold  # 250% total deviation threshold
        self.rebalance_wait_days = rebalance_wait_days  # 180 day waiting period
        
    def load_data(self):
        """Load the IWLS data"""
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        input_file = os.path.join(results_dir, "IWLS_WITH_FORWARD_PERFORMANCE.csv")
        
        if not os.path.exists(input_file):
            print(f"❌ Error: Data file not found at {input_file}")
            return None
        
        print(f"📂 Loading data: {input_file}")
        
        try:
            df = pd.read_csv(input_file)
            df['date'] = pd.to_datetime(df['date'])
            print(f"✅ Successfully loaded {len(df):,} records")
            
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def prepare_data(self, df):
        """Prepare and clean data for backtesting"""
        print("🔧 Preparing trading data...")
        
        # Keep only records with IWLS deviation values and prices
        clean_data = df.dropna(subset=['price_deviation_from_trend', 'price']).copy()
        
        # Sort chronologically
        clean_data = clean_data.sort_values(['date', 'asset']).reset_index(drop=True)
        
        print(f"📊 Clean records: {len(clean_data):,}")
        print(f"📅 Date range: {clean_data['date'].min().strftime('%Y-%m-%d')} to {clean_data['date'].max().strftime('%Y-%m-%d')}")
        print(f"🏢 Unique assets: {clean_data['asset'].nunique()}")
        
        return clean_data
    
    def find_worst_performers(self, df, current_date, lookback_days=7):
        """Find the 5 most negatively deviated stocks around current date"""
        start_date = current_date - timedelta(days=lookback_days)
        end_date = current_date + timedelta(days=lookback_days)
        
        # Get recent data around this date
        recent_data = df[
            (df['date'] >= start_date) & 
            (df['date'] <= end_date)
        ]
        
        if len(recent_data) == 0:
            return []
        
        # Get the most recent record for each asset
        latest_by_asset = recent_data.loc[
            recent_data.groupby('asset')['date'].idxmax()
        ].reset_index(drop=True)
        
        # Sort by worst deviation (most negative)
        worst_performers = latest_by_asset.sort_values('price_deviation_from_trend')
        
        # Return top N worst performers
        opportunities = []
        for _, row in worst_performers.head(self.num_positions).iterrows():
            opportunities.append({
                'asset': row['asset'],
                'entry_price': row['price'],
                'deviation': row['price_deviation_from_trend'],
                'entry_date': row['date']
            })
        
        return opportunities
    
    def check_rebalance_trigger(self, df, current_date, last_rebalance_date):
        """Check if rebalancing conditions are met"""
        # Check if enough time has passed since last rebalance
        if last_rebalance_date is not None:
            days_since_rebalance = (current_date - last_rebalance_date).days
            if days_since_rebalance < self.rebalance_wait_days:
                return False, 0.0, []
        
        # Find current worst performers
        worst_performers = self.find_worst_performers(df, current_date)
        
        if len(worst_performers) < self.num_positions:
            return False, 0.0, []
        
        # Calculate sum of absolute deviations (since they're negative, we take absolute)
        total_deviation = sum(abs(opp['deviation']) for opp in worst_performers)
        
        # Check if threshold is exceeded
        trigger_rebalance = total_deviation >= self.rebalance_threshold
        
        return trigger_rebalance, total_deviation, worst_performers
    
    def get_current_prices(self, df, assets, current_date, lookback_days=7):
        """Get current prices for a list of assets on or near current date"""
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
                # Get most recent price
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
                # If no current price, use entry value
                total_value += position['investment']
        
        return total_value
    
    def create_positions(self, opportunities, available_capital):
        """Create position list from opportunities"""
        if len(opportunities) == 0:
            return []
        
        investment_per_position = available_capital / len(opportunities)
        positions = []
        
        for opp in opportunities:
            position = {
                'asset': opp['asset'],
                'entry_date': opp['entry_date'],
                'entry_price': opp['entry_price'],
                'entry_deviation': opp['deviation'],
                'investment': investment_per_position,
                'shares': investment_per_position / opp['entry_price']
            }
            positions.append(position)
        
        return positions
    
    def backtest_strategy(self):
        """Run the backtest"""
        
        # Load and prepare data
        df = self.load_data()
        if df is None:
            return None
        
        data = self.prepare_data(df)
        
        # Find start of data with IWLS values
        start_date = data['date'].min()
        end_date = data['date'].max()
        
        print(f"\n🚀 STARTING BACKTEST")
        print("="*70)
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Strategy: Buy {self.num_positions} worst performers, exit at +{self.profit_target}%")
        print(f"🔄 Rebalance: If top 5 deviations sum > {self.rebalance_threshold}% and {self.rebalance_wait_days}+ days passed")
        print(f"📅 Backtest period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get all unique trading dates
        all_dates = sorted(data['date'].unique())
        print(f"📊 Trading days: {len(all_dates):,}")
        
        # Initialize tracking
        current_capital = self.initial_capital
        active_positions = []
        portfolio_number = 0
        completed_portfolios = []
        daily_tracking = []
        rebalance_events = []
        
        # Track rebalancing
        last_rebalance_date = None
        
        # Start with first portfolio
        initial_opportunities = self.find_worst_performers(data, start_date)
        
        if len(initial_opportunities) == 0:
            print("❌ No initial opportunities found")
            return None
        
        # Enter first portfolio
        active_positions = self.create_positions(initial_opportunities, self.initial_capital)
        portfolio_number = 1
        portfolio_entry_date = start_date
        last_rebalance_date = start_date
        total_invested = sum(pos['investment'] for pos in active_positions)
        current_capital -= total_invested
        
        print(f"\n🔥 Portfolio #1 - {start_date.strftime('%Y-%m-%d')}")
        print(f"    💰 Invested: ${total_invested:,.2f}")
        print(f"    🎯 Target Exit: ${total_invested * (1 + self.profit_target/100):,.2f} (+{self.profit_target}%)")
        print(f"    📊 Assets: {[pos['asset'] for pos in active_positions]}")
        print(f"    📉 Deviations: {[f'{pos['entry_deviation']:.1f}%' for pos in active_positions]}")
        
        # Main backtest loop
        for i, current_date in enumerate(all_dates):
            
            # Get current prices for active positions
            if len(active_positions) > 0:
                asset_list = [pos['asset'] for pos in active_positions]
                current_prices = self.get_current_prices(data, asset_list, current_date)
                
                # Calculate portfolio value
                portfolio_value = self.calculate_portfolio_value(active_positions, current_prices)
                
                # Calculate return
                total_invested = sum(pos['investment'] for pos in active_positions)
                portfolio_return = ((portfolio_value / total_invested) - 1) * 100
                hold_days = (current_date - portfolio_entry_date).days
                
                # Check for profit target exit
                profit_target_hit = portfolio_return >= self.profit_target
                
                # Check for rebalance trigger
                should_rebalance, total_deviation, rebalance_opportunities = self.check_rebalance_trigger(
                    data, current_date, last_rebalance_date)
                
                # Decide on exit strategy
                exit_reason = None
                if profit_target_hit:
                    exit_reason = f"PROFIT_TARGET (+{portfolio_return:.1f}%)"
                elif should_rebalance:
                    exit_reason = f"REBALANCE (deviation sum: {total_deviation:.1f}%)"
                
                if exit_reason:
                    # EXIT CURRENT PORTFOLIO
                    current_capital += portfolio_value
                    
                    # Record completed portfolio
                    completed_portfolio = {
                        'portfolio_number': portfolio_number,
                        'entry_date': portfolio_entry_date,
                        'exit_date': current_date,
                        'hold_days': hold_days,
                        'invested': total_invested,
                        'exit_value': portfolio_value,
                        'return_pct': portfolio_return,
                        'exit_reason': exit_reason,
                        'assets': [pos['asset'] for pos in active_positions],
                        'entry_deviations': [pos['entry_deviation'] for pos in active_positions]
                    }
                    
                    completed_portfolios.append(completed_portfolio)
                    
                    print(f"\n📤 Portfolio #{portfolio_number} EXIT - {current_date.strftime('%Y-%m-%d')}")
                    print(f"    📊 Reason: {exit_reason}")
                    if should_rebalance:
                        days_since_rebalance = (current_date - last_rebalance_date).days if last_rebalance_date else 0
                        print(f"    🔄 Days since last rebalance: {days_since_rebalance}")
                        print(f"    📉 Current worst 5 deviation sum: {total_deviation:.1f}%")
                    print(f"    📈 Return: {portfolio_return:+.1f}% in {hold_days} days")
                    print(f"    💰 Value: ${portfolio_value:,.2f} -> Total: ${current_capital:,.2f}")
                    
                    # Track rebalance event if applicable
                    if should_rebalance:
                        rebalance_events.append({
                            'date': current_date,
                            'total_deviation': total_deviation,
                            'days_since_last': (current_date - last_rebalance_date).days if last_rebalance_date else 0,
                            'old_assets': [pos['asset'] for pos in active_positions],
                            'new_assets': [opp['asset'] for opp in rebalance_opportunities]
                        })
                        last_rebalance_date = current_date
                    
                    # IMMEDIATELY ENTER NEW PORTFOLIO
                    if should_rebalance:
                        new_opportunities = rebalance_opportunities
                    else:
                        new_opportunities = self.find_worst_performers(data, current_date)
                    
                    if len(new_opportunities) > 0 and current_capital > 1000:
                        # Enter new portfolio
                        active_positions = self.create_positions(new_opportunities, current_capital)
                        portfolio_number += 1
                        portfolio_entry_date = current_date
                        total_new_invested = sum(pos['investment'] for pos in active_positions)
                        current_capital -= total_new_invested
                        
                        print(f"\n🔥 Portfolio #{portfolio_number} - {current_date.strftime('%Y-%m-%d')}")
                        print(f"    💰 Invested: ${total_new_invested:,.2f}")
                        print(f"    🎯 Assets: {[pos['asset'] for pos in active_positions]}")
                        print(f"    📊 Deviations: {[f'{pos['entry_deviation']:.1f}%' for pos in active_positions]}")
                        
                        # Recalculate portfolio value for tracking
                        asset_list = [pos['asset'] for pos in active_positions]
                        current_prices = self.get_current_prices(data, asset_list, current_date)
                        portfolio_value = self.calculate_portfolio_value(active_positions, current_prices)
                    else:
                        # No new opportunities or insufficient capital
                        active_positions = []
                        portfolio_value = 0
            else:
                portfolio_value = 0
            
            # Daily tracking
            total_account_value = current_capital + portfolio_value
            daily_tracking.append({
                'date': current_date,
                'cash': current_capital,
                'portfolio_value': portfolio_value,
                'total_value': total_account_value,
                'total_return_pct': ((total_account_value / self.initial_capital) - 1) * 100,
                'active_positions': len(active_positions),
                'completed_portfolios': len(completed_portfolios),
                'portfolio_number': portfolio_number if len(active_positions) > 0 else 0
            })
            
            # Progress updates (monthly)
            if current_date.day == 1:  # First day of month
                total_return = ((total_account_value / self.initial_capital) - 1) * 100
                print(f"📅 {current_date.strftime('%Y-%m')}: ${total_account_value:,.0f} ({total_return:+.1f}%) | "
                      f"Completed: {len(completed_portfolios)} | Active: {len(active_positions)} | Rebalances: {len(rebalance_events)}")
        
        # Final calculations
        final_value = current_capital
        if len(active_positions) > 0:
            # Calculate final portfolio value
            asset_list = [pos['asset'] for pos in active_positions]
            final_prices = self.get_current_prices(data, asset_list, all_dates[-1])
            final_portfolio_value = self.calculate_portfolio_value(active_positions, final_prices)
            final_value += final_portfolio_value
        
        return {
            'completed_portfolios': completed_portfolios,
            'daily_tracking': daily_tracking,
            'rebalance_events': rebalance_events,
            'final_value': final_value,
            'total_return': ((final_value / self.initial_capital) - 1) * 100,
            'active_positions': active_positions
        }
    
    def analyze_results(self, results):
        """Analyze backtest results"""
        if results is None:
            return
        
        print(f"\n📊 BACKTEST RESULTS")
        print("="*60)
        
        # Overall performance
        print(f"\n💰 PERFORMANCE:")
        print(f"   Initial: ${self.initial_capital:,.2f}")
        print(f"   Final: ${results['final_value']:,.2f}")
        print(f"   Total Return: {results['total_return']:+.2f}%")
        
        if len(results['daily_tracking']) > 0:
            days = len(results['daily_tracking'])
            years = days / 365.25
            if years > 0:
                annualized = ((results['final_value'] / self.initial_capital) ** (1/years) - 1) * 100
                print(f"   Period: {years:.2f} years")
                print(f"   Annualized: {annualized:+.2f}%")
        
        # Portfolio statistics
        portfolios = results['completed_portfolios']
        if len(portfolios) > 0:
            profit_exits = [p for p in portfolios if 'PROFIT_TARGET' in p.get('exit_reason', '')]
            rebalance_exits = [p for p in portfolios if 'REBALANCE' in p.get('exit_reason', '')]
            
            returns = [p['return_pct'] for p in portfolios]
            hold_times = [p['hold_days'] for p in portfolios]
            
            print(f"\n📈 PORTFOLIO STATS:")
            print(f"   Completed: {len(portfolios)}")
            print(f"   Profit Target Exits: {len(profit_exits)}")
            print(f"   Rebalance Exits: {len(rebalance_exits)}")
            print(f"   Avg Return: {np.mean(returns):.1f}%")
            print(f"   Avg Hold Time: {np.mean(hold_times):.0f} days")
            print(f"   Min Hold Time: {min(hold_times):.0f} days")
            print(f"   Max Hold Time: {max(hold_times):.0f} days")
        
        # Rebalance statistics
        rebalances = results['rebalance_events']
        if len(rebalances) > 0:
            print(f"\n🔄 REBALANCE STATS:")
            print(f"   Total Rebalances: {len(rebalances)}")
            avg_deviation = np.mean([r['total_deviation'] for r in rebalances])
            print(f"   Avg Trigger Deviation: {avg_deviation:.1f}%")
            
            print(f"\n🎯 FIRST 3 REBALANCES:")
            for i, r in enumerate(rebalances[:3]):
                print(f"   #{i+1}: {r['date'].strftime('%Y-%m-%d')} | "
                      f"Deviation: {r['total_deviation']:.1f}% | "
                      f"Days since last: {r['days_since_last']}")
        
        return results
    
    def save_results(self, results):
        """Save results to CSV"""
        if results is None:
            return
        
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        
        # Save daily tracking
        if len(results['daily_tracking']) > 0:
            daily_df = pd.DataFrame(results['daily_tracking'])
            daily_file = os.path.join(results_dir, "UNDERPERFORMANCE_BACKTEST_DAILY.csv")
            daily_df.to_csv(daily_file, index=False)
            print(f"✅ Daily values: {daily_file}")
        
        # Save completed portfolios
        if len(results['completed_portfolios']) > 0:
            portfolios_df = pd.DataFrame(results['completed_portfolios'])
            # Convert lists to strings for CSV
            portfolios_df['assets'] = portfolios_df['assets'].astype(str)
            portfolios_df['entry_deviations'] = portfolios_df['entry_deviations'].astype(str)
            portfolio_file = os.path.join(results_dir, "UNDERPERFORMANCE_BACKTEST_PORTFOLIOS.csv")
            portfolios_df.to_csv(portfolio_file, index=False)
            print(f"✅ Portfolios: {portfolio_file}")
        
        # Save rebalance events
        if len(results['rebalance_events']) > 0:
            rebalance_df = pd.DataFrame(results['rebalance_events'])
            # Convert lists to strings for CSV
            rebalance_df['old_assets'] = rebalance_df['old_assets'].astype(str)
            rebalance_df['new_assets'] = rebalance_df['new_assets'].astype(str)
            rebalance_file = os.path.join(results_dir, "UNDERPERFORMANCE_BACKTEST_REBALANCES.csv")
            rebalance_df.to_csv(rebalance_file, index=False)
            print(f"✅ Rebalances: {rebalance_file}")


def run_backtest():
    """Run the underperformance backtest"""
    print("IWLS UNDERPERFORMANCE BACKTEST")
    print("="*50)
    
    backtester = UnderperformanceBacktester(
        initial_capital=10000,
        num_positions=3,
        profit_target=100.0,  # 50% profit target
        rebalance_threshold=200.0,  # Rebalance if total deviation > 250%
        rebalance_wait_days=200  # Wait 180 days between rebalances
    )
    
    # Run backtest
    results = backtester.backtest_strategy()
    
    # Analyze and save
    backtester.analyze_results(results)
    backtester.save_results(results)
    
    print(f"\n✨ BACKTEST COMPLETE!")
    
    return results


if __name__ == "__main__":
    backtest_results = run_backtest()