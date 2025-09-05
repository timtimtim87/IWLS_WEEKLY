import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UnderperformanceBacktester:
    """
    Simple backtest strategy:
    1. Find 5 most negatively deviated stocks
    2. Split $10,000 equally amongst them ($2,000 each)
    3. Track portfolio value daily
    4. Exit when portfolio hits +50% total return
    5. Immediately re-enter with new worst 5 stocks
    """
    
    def __init__(self, initial_capital=10000, num_positions=5, profit_target=50.0):
        self.initial_capital = initial_capital
        self.num_positions = num_positions
        self.profit_target = profit_target  # 50% profit target
        
    def load_data(self):
        """Load the IWLS data"""
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        input_file = os.path.join(results_dir, "IWLS_WITH_FORWARD_PERFORMANCE.csv")
        
        if not os.path.exists(input_file):
            print(f"❌ Error: Data file not found at {input_file}")
            return None
        
        print(f"📁 Loading data: {input_file}")
        
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
        print("="*60)
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Strategy: Buy {self.num_positions} worst performers, exit at +{self.profit_target}%")
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
        
        # Start with first portfolio
        initial_opportunities = self.find_worst_performers(data, start_date)
        
        if len(initial_opportunities) == 0:
            print("❌ No initial opportunities found")
            return None
        
        # Enter first portfolio
        investment_per_position = self.initial_capital / len(initial_opportunities)
        
        for opp in initial_opportunities:
            position = {
                'asset': opp['asset'],
                'entry_date': opp['entry_date'],
                'entry_price': opp['entry_price'],
                'entry_deviation': opp['deviation'],
                'investment': investment_per_position,
                'shares': investment_per_position / opp['entry_price']
            }
            active_positions.append(position)
        
        portfolio_number = 1
        portfolio_entry_date = start_date
        total_invested = sum(pos['investment'] for pos in active_positions)
        current_capital -= total_invested
        
        print(f"\n📥 Portfolio #1 - {start_date.strftime('%Y-%m-%d')}")
        print(f"    💰 Invested: ${total_invested:,.2f}")
        print(f"    🎯 Assets: {[pos['asset'] for pos in active_positions]}")
        print(f"    📊 Deviations: {[f'{pos['entry_deviation']:.1f}%' for pos in active_positions]}")
        
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
                
                # Check for exit condition
                if portfolio_return >= self.profit_target:
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
                        'assets': [pos['asset'] for pos in active_positions],
                        'entry_deviations': [pos['entry_deviation'] for pos in active_positions]
                    }
                    
                    completed_portfolios.append(completed_portfolio)
                    
                    print(f"\n📤 Portfolio #{portfolio_number} EXIT - {current_date.strftime('%Y-%m-%d')}")
                    print(f"    📊 Return: {portfolio_return:+.1f}% in {hold_days} days")
                    print(f"    💰 Value: ${portfolio_value:,.2f} -> Total: ${current_capital:,.2f}")
                    
                    # IMMEDIATELY ENTER NEW PORTFOLIO
                    new_opportunities = self.find_worst_performers(data, current_date)
                    
                    if len(new_opportunities) > 0 and current_capital > 1000:
                        # Enter new portfolio
                        investment_per_position = current_capital / len(new_opportunities)
                        
                        new_positions = []
                        for opp in new_opportunities:
                            position = {
                                'asset': opp['asset'],
                                'entry_date': current_date,
                                'entry_price': opp['entry_price'],
                                'entry_deviation': opp['deviation'],
                                'investment': investment_per_position,
                                'shares': investment_per_position / opp['entry_price']
                            }
                            new_positions.append(position)
                        
                        active_positions = new_positions
                        portfolio_number += 1
                        portfolio_entry_date = current_date
                        total_new_invested = sum(pos['investment'] for pos in active_positions)
                        current_capital -= total_new_invested
                        
                        print(f"\n📥 Portfolio #{portfolio_number} - {current_date.strftime('%Y-%m-%d')}")
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
                'completed_portfolios': len(completed_portfolios)
            })
            
            # Progress updates (monthly)
            if current_date.day == 1:  # First day of month
                total_return = ((total_account_value / self.initial_capital) - 1) * 100
                print(f"📅 {current_date.strftime('%Y-%m')}: ${total_account_value:,.0f} ({total_return:+.1f}%) | "
                      f"Completed: {len(completed_portfolios)} | Active: {len(active_positions)}")
        
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
            'final_value': final_value,
            'total_return': ((final_value / self.initial_capital) - 1) * 100,
            'active_positions': active_positions
        }
    
    def analyze_results(self, results):
        """Analyze backtest results"""
        if results is None:
            return
        
        print(f"\n📊 BACKTEST RESULTS")
        print("="*50)
        
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
            returns = [p['return_pct'] for p in portfolios]
            hold_times = [p['hold_days'] for p in portfolios]
            
            print(f"\n📈 PORTFOLIO STATS:")
            print(f"   Completed: {len(portfolios)}")
            print(f"   All hit {self.profit_target}% target")
            print(f"   Avg Hold Time: {np.mean(hold_times):.0f} days")
            print(f"   Min Hold Time: {min(hold_times):.0f} days")
            print(f"   Max Hold Time: {max(hold_times):.0f} days")
            
            # Show first few portfolios
            print(f"\n🎯 FIRST 5 PORTFOLIOS:")
            for i, p in enumerate(portfolios[:5]):
                print(f"   #{i+1}: {p['entry_date'].strftime('%Y-%m-%d')} -> {p['exit_date'].strftime('%Y-%m-%d')} "
                      f"({p['hold_days']} days) | {p['return_pct']:.1f}% | {p['assets']}")
        
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
            portfolio_file = os.path.join(results_dir, "UNDERPERFORMANCE_BACKTEST_PORTFOLIOS.csv")
            portfolios_df.to_csv(portfolio_file, index=False)
            print(f"✅ Portfolios: {portfolio_file}")


def run_backtest():
    """Run the underperformance backtest"""
    print("IWLS UNDERPERFORMANCE BACKTEST")
    print("="*50)
    
    backtester = UnderperformanceBacktester(
        initial_capital=10000,
        num_positions=3,
        profit_target=20.0  # 50% profit target
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