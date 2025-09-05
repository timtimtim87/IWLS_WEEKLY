import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RollingWindowBacktester:
    """
    Rolling Window Strategy:
    1. Find 3 most negatively deviated stocks each day
    2. Calculate sum of their absolute deviations
    3. Compare to 365-day rolling window maximum
    4. If current sum > all previous 365 days, enter new positions
    5. Exit old positions when entering new ones
    6. No profit targets - only rebalance based on rolling window maximum
    """
    
    def __init__(self, initial_capital=10000, num_positions=3, rolling_window_days=365):
        self.initial_capital = initial_capital
        self.num_positions = num_positions
        self.rolling_window_days = rolling_window_days
        
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
        """Find the N most negatively deviated stocks around current date"""
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
    
    def calculate_deviation_sum(self, opportunities):
        """Calculate sum of absolute deviations"""
        if len(opportunities) == 0:
            return 0.0
        return sum(abs(opp['deviation']) for opp in opportunities)
    
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
        """Run the rolling window backtest"""
        
        # Load and prepare data
        df = self.load_data()
        if df is None:
            return None
        
        data = self.prepare_data(df)
        
        # Find start of data with IWLS values
        start_date = data['date'].min()
        end_date = data['date'].max()
        
        print(f"\n🚀 STARTING ROLLING WINDOW BACKTEST")
        print("="*70)
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Strategy: Buy {self.num_positions} worst performers when rolling window maximum exceeded")
        print(f"🔄 Rolling Window: {self.rolling_window_days} days")
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
        entry_events = []
        
        # Track daily deviation sums for rolling window
        daily_deviation_sums = {}
        
        print(f"\n🔍 BUILDING ROLLING WINDOW DATA...")
        
        # First pass: Calculate daily deviation sums for all dates
        for current_date in all_dates:
            worst_performers = self.find_worst_performers(data, current_date)
            deviation_sum = self.calculate_deviation_sum(worst_performers)
            daily_deviation_sums[current_date] = {
                'sum': deviation_sum,
                'opportunities': worst_performers
            }
        
        print(f"✅ Built deviation sums for {len(daily_deviation_sums)} trading days")
        
        # Second pass: Run backtest with rolling window logic
        for i, current_date in enumerate(all_dates):
            
            # Skip first year to build rolling window
            if i < self.rolling_window_days:
                continue
            
            current_data = daily_deviation_sums[current_date]
            current_sum = current_data['sum']
            current_opportunities = current_data['opportunities']
            
            # Get rolling window (previous N days)
            window_start_idx = max(0, i - self.rolling_window_days)
            window_dates = all_dates[window_start_idx:i]  # Exclude current day
            
            # Find maximum in rolling window
            window_max = 0.0
            window_max_date = None
            for window_date in window_dates:
                if window_date in daily_deviation_sums:
                    window_sum = daily_deviation_sums[window_date]['sum']
                    if window_sum > window_max:
                        window_max = window_sum
                        window_max_date = window_date
            
            # Check if current sum exceeds rolling window maximum
            new_entry_signal = current_sum > window_max and len(current_opportunities) == self.num_positions
            
            # Calculate current portfolio value if we have positions
            portfolio_value = 0
            if len(active_positions) > 0:
                asset_list = [pos['asset'] for pos in active_positions]
                current_prices = self.get_current_prices(data, asset_list, current_date)
                portfolio_value = self.calculate_portfolio_value(active_positions, current_prices)
            
            # Handle new entry signal
            if new_entry_signal:
                
                # Exit current positions if any
                if len(active_positions) > 0:
                    current_capital += portfolio_value
                    
                    # Record completed portfolio
                    portfolio_entry_date = active_positions[0]['entry_date']  # They should all be the same
                    total_invested = sum(pos['investment'] for pos in active_positions)
                    portfolio_return = ((portfolio_value / total_invested) - 1) * 100
                    hold_days = (current_date - portfolio_entry_date).days
                    
                    completed_portfolio = {
                        'portfolio_number': portfolio_number,
                        'entry_date': portfolio_entry_date,
                        'exit_date': current_date,
                        'hold_days': hold_days,
                        'invested': total_invested,
                        'exit_value': portfolio_value,
                        'return_pct': portfolio_return,
                        'exit_reason': 'ROLLING_WINDOW_SIGNAL',
                        'assets': [pos['asset'] for pos in active_positions],
                        'entry_deviations': [pos['entry_deviation'] for pos in active_positions]
                    }
                    
                    completed_portfolios.append(completed_portfolio)
                    
                    print(f"\n📤 Portfolio #{portfolio_number} EXIT - {current_date.strftime('%Y-%m-%d')}")
                    print(f"    📊 Reason: New rolling window maximum detected")
                    print(f"    📈 Return: {portfolio_return:+.1f}% in {hold_days} days")
                    print(f"    💰 Value: ${portfolio_value:,.2f}")
                
                # Enter new portfolio
                if current_capital > 1000:
                    active_positions = self.create_positions(current_opportunities, current_capital)
                    portfolio_number += 1
                    total_invested = sum(pos['investment'] for pos in active_positions)
                    current_capital -= total_invested
                    
                    # Record entry event
                    entry_events.append({
                        'date': current_date,
                        'portfolio_number': portfolio_number,
                        'deviation_sum': current_sum,
                        'rolling_window_max': window_max,
                        'window_max_date': window_max_date,
                        'assets': [opp['asset'] for opp in current_opportunities],
                        'deviations': [opp['deviation'] for opp in current_opportunities],
                        'invested': total_invested
                    })
                    
                    print(f"\n🔥 Portfolio #{portfolio_number} ENTRY - {current_date.strftime('%Y-%m-%d')}")
                    print(f"    📊 Current deviation sum: {current_sum:.1f}%")
                    print(f"    📈 Rolling window max: {window_max:.1f}% ({window_max_date.strftime('%Y-%m-%d') if window_max_date else 'N/A'})")
                    print(f"    💰 Invested: ${total_invested:,.2f}")
                    print(f"    🎯 Assets: {[pos['asset'] for pos in active_positions]}")
                    print(f"    📉 Deviations: {[f'{pos['entry_deviation']:.1f}%' for pos in active_positions]}")
                    
                    # Recalculate portfolio value for tracking
                    asset_list = [pos['asset'] for pos in active_positions]
                    current_prices = self.get_current_prices(data, asset_list, current_date)
                    portfolio_value = self.calculate_portfolio_value(active_positions, current_prices)
                else:
                    active_positions = []
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
                'portfolio_number': portfolio_number if len(active_positions) > 0 else 0,
                'current_deviation_sum': current_sum,
                'rolling_window_max': window_max,
                'new_entry_signal': new_entry_signal
            })
            
            # Progress updates (monthly)
            if current_date.day == 1:  # First day of month
                total_return = ((total_account_value / self.initial_capital) - 1) * 100
                print(f"📅 {current_date.strftime('%Y-%m')}: ${total_account_value:,.0f} ({total_return:+.1f}%) | "
                      f"Portfolio: {portfolio_number if len(active_positions) > 0 else 0} | "
                      f"Entries: {len(entry_events)} | Current Sum: {current_sum:.1f}%")
        
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
            'entry_events': entry_events,
            'daily_deviation_sums': daily_deviation_sums,
            'final_value': final_value,
            'total_return': ((final_value / self.initial_capital) - 1) * 100,
            'active_positions': active_positions
        }
    
    def analyze_results(self, results):
        """Analyze backtest results"""
        if results is None:
            return
        
        print(f"\n📊 ROLLING WINDOW BACKTEST RESULTS")
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
        entry_events = results['entry_events']
        
        if len(portfolios) > 0:
            returns = [p['return_pct'] for p in portfolios]
            hold_times = [p['hold_days'] for p in portfolios]
            
            print(f"\n📈 PORTFOLIO STATS:")
            print(f"   Completed: {len(portfolios)}")
            print(f"   Total Entries: {len(entry_events)}")
            print(f"   Avg Return: {np.mean(returns):.1f}%")
            print(f"   Best Return: {max(returns):.1f}%")
            print(f"   Worst Return: {min(returns):.1f}%")
            print(f"   Avg Hold Time: {np.mean(hold_times):.0f} days")
            print(f"   Min Hold Time: {min(hold_times):.0f} days")
            print(f"   Max Hold Time: {max(hold_times):.0f} days")
        
        # Entry signal statistics
        if len(entry_events) > 0:
            deviation_sums = [e['deviation_sum'] for e in entry_events]
            print(f"\n🔍 ENTRY SIGNAL STATS:")
            print(f"   Total Entries: {len(entry_events)}")
            print(f"   Avg Entry Deviation Sum: {np.mean(deviation_sums):.1f}%")
            print(f"   Min Entry Deviation Sum: {min(deviation_sums):.1f}%")
            print(f"   Max Entry Deviation Sum: {max(deviation_sums):.1f}%")
            
            print(f"\n🎯 FIRST 5 ENTRIES:")
            for i, e in enumerate(entry_events[:5]):
                print(f"   #{i+1}: {e['date'].strftime('%Y-%m-%d')} | "
                      f"Sum: {e['deviation_sum']:.1f}% | "
                      f"Assets: {e['assets']} | "
                      f"Window Max: {e['rolling_window_max']:.1f}%")
        
        return results
    
    def save_results(self, results):
        """Save results to CSV"""
        if results is None:
            return
        
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        
        # Save daily tracking
        if len(results['daily_tracking']) > 0:
            daily_df = pd.DataFrame(results['daily_tracking'])
            daily_file = os.path.join(results_dir, "ROLLING_WINDOW_BACKTEST_DAILY.csv")
            daily_df.to_csv(daily_file, index=False)
            print(f"✅ Daily values: {daily_file}")
        
        # Save completed portfolios
        if len(results['completed_portfolios']) > 0:
            portfolios_df = pd.DataFrame(results['completed_portfolios'])
            # Convert lists to strings for CSV
            portfolios_df['assets'] = portfolios_df['assets'].astype(str)
            portfolios_df['entry_deviations'] = portfolios_df['entry_deviations'].astype(str)
            portfolio_file = os.path.join(results_dir, "ROLLING_WINDOW_BACKTEST_PORTFOLIOS.csv")
            portfolios_df.to_csv(portfolio_file, index=False)
            print(f"✅ Portfolios: {portfolio_file}")
        
        # Save entry events
        if len(results['entry_events']) > 0:
            entries_df = pd.DataFrame(results['entry_events'])
            # Convert lists to strings for CSV
            entries_df['assets'] = entries_df['assets'].astype(str)
            entries_df['deviations'] = entries_df['deviations'].astype(str)
            entries_file = os.path.join(results_dir, "ROLLING_WINDOW_BACKTEST_ENTRIES.csv")
            entries_df.to_csv(entries_file, index=False)
            print(f"✅ Entry events: {entries_file}")
        
        # Save daily deviation sums
        deviation_data = []
        for date, info in results['daily_deviation_sums'].items():
            assets = [opp['asset'] for opp in info['opportunities']]
            deviations = [opp['deviation'] for opp in info['opportunities']]
            deviation_data.append({
                'date': date,
                'deviation_sum': info['sum'],
                'assets': str(assets),
                'deviations': str(deviations)
            })
        
        if len(deviation_data) > 0:
            deviation_df = pd.DataFrame(deviation_data)
            deviation_file = os.path.join(results_dir, "ROLLING_WINDOW_DAILY_DEVIATIONS.csv")
            deviation_df.to_csv(deviation_file, index=False)
            print(f"✅ Daily deviations: {deviation_file}")


def run_rolling_window_backtest():
    """Run the rolling window backtest"""
    print("IWLS ROLLING WINDOW BACKTEST")
    print("="*50)
    
    backtester = RollingWindowBacktester(
        initial_capital=10000,
        num_positions=3,  # 3 most underperforming stocks
        rolling_window_days=365  # 365 day rolling window
    )
    
    # Run backtest
    results = backtester.backtest_strategy()
    
    # Analyze and save
    backtester.analyze_results(results)
    backtester.save_results(results)
    
    print(f"\n✨ ROLLING WINDOW BACKTEST COMPLETE!")
    
    return results


if __name__ == "__main__":
    backtest_results = run_rolling_window_backtest()