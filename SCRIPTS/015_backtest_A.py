import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FixedUnderperformanceBacktester:
    """
    Fixed backtest strategy with proper chronological logic:
    1. Enter positions when opportunities arise
    2. Track daily portfolio value 
    3. Exit when 50% target hit OR 12 months elapsed
    4. Only one portfolio active at a time
    """
    
    def __init__(self, initial_capital=10000, max_positions=5):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        
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
            
            start_date = df['date'].min()
            end_date = df['date'].max()
            print(f"📅 Full data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def prepare_data(self, df):
        """Prepare and clean data for backtesting"""
        print("🔧 Preparing trading data...")
        
        # Keep records with price and deviation data
        clean_data = df.dropna(subset=['price_deviation_from_trend', 'price']).copy()
        
        # Sort chronologically
        clean_data = clean_data.sort_values(['date', 'asset']).reset_index(drop=True)
        
        print(f"📊 Clean records: {len(clean_data):,}")
        print(f"📅 Date range: {clean_data['date'].min().strftime('%Y-%m-%d')} to {clean_data['date'].max().strftime('%Y-%m-%d')}")
        print(f"🏢 Unique assets: {clean_data['asset'].nunique()}")
        
        return clean_data
    
    def get_price_for_asset(self, df, asset, date, lookback_days=7):
        """Get price for specific asset on or near specific date"""
        start_date = date - timedelta(days=lookback_days)
        
        asset_data = df[
            (df['asset'] == asset) & 
            (df['date'] >= start_date) & 
            (df['date'] <= date)
        ]
        
        if len(asset_data) == 0:
            return None, None
            
        latest = asset_data.loc[asset_data['date'].idxmax()]
        return latest['price'], latest['date']
    
    def find_underperformers(self, df, current_date, lookback_days=14):
        """Find most underperforming stocks around current date"""
        start_date = current_date - timedelta(days=lookback_days)
        
        # Get recent data
        recent_data = df[
            (df['date'] >= start_date) & 
            (df['date'] <= current_date)
        ]
        
        if len(recent_data) == 0:
            return []
        
        # Get most recent record for each asset
        latest_by_asset = recent_data.groupby('asset').apply(
            lambda x: x.loc[x['date'].idxmax()]
        ).reset_index(drop=True)
        
        # Filter severe underperformers
        underperformers = latest_by_asset[
            latest_by_asset['price_deviation_from_trend'] < -20
        ].copy()
        
        if len(underperformers) == 0:
            return []
        
        # Sort by worst performance first
        underperformers = underperformers.sort_values('price_deviation_from_trend')
        
        # Return top candidates
        opportunities = []
        for _, row in underperformers.head(self.max_positions).iterrows():
            opportunities.append({
                'asset': row['asset'],
                'price': row['price'],
                'deviation': row['price_deviation_from_trend'],
                'data_date': row['date']
            })
        
        return opportunities
    
    def calculate_portfolio_value(self, positions, df, current_date):
        """Calculate current total portfolio value"""
        total_value = 0
        position_details = {}
        
        for pos in positions:
            current_price, price_date = self.get_price_for_asset(
                df, pos['asset'], current_date
            )
            
            if current_price is not None:
                position_value = pos['shares'] * current_price
                total_value += position_value
                
                position_details[pos['asset']] = {
                    'current_price': current_price,
                    'position_value': position_value,
                    'return_pct': ((current_price / pos['entry_price']) - 1) * 100,
                    'price_date': price_date
                }
            else:
                # No current price available, use entry value
                total_value += pos['investment']
                position_details[pos['asset']] = {
                    'current_price': pos['entry_price'],
                    'position_value': pos['investment'],
                    'return_pct': 0.0,
                    'price_date': pos['entry_date']
                }
        
        return total_value, position_details
    
    def backtest_strategy(self, start_year=None, end_year=None):
        """Run backtest with proper chronological logic"""
        
        # Load data
        df = self.load_data()
        if df is None:
            return None
        
        data = self.prepare_data(df)
        
        # Set date range - use full range if not specified
        if start_year is None:
            start_year = data['date'].min().year + 1  # Skip first year for lookback
        if end_year is None:
            end_year = data['date'].max().year - 1  # Leave last year for forward tracking
        
        backtest_start = datetime(start_year, 1, 1)
        backtest_end = datetime(end_year, 12, 31)
        
        print(f"\n🚀 STARTING FIXED BACKTEST: {start_year} to {end_year}")
        print("="*60)
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Strategy: Buy 5 worst performers, exit at +50% or 12 months")
        print(f"📈 Track portfolio value daily")
        
        # Get all unique dates in backtest period
        backtest_data = data[
            (data['date'] >= backtest_start) & 
            (data['date'] <= backtest_end)
        ]
        
        all_dates = sorted(backtest_data['date'].unique())
        print(f"📅 Trading days in period: {len(all_dates):,}")
        
        # Initialize tracking
        current_capital = self.initial_capital
        active_positions = []
        portfolio_entry_date = None
        completed_portfolios = []
        daily_tracking = []
        
        portfolio_count = 0
        
        # Main backtest loop - go through each date chronologically
        for i, current_date in enumerate(all_dates):
            
            # If no active portfolio, look for entry opportunity
            if len(active_positions) == 0:
                
                # Only look for entries on first trading day of each month
                if current_date.day <= 7 or (i > 0 and all_dates[i-1].month != current_date.month):
                    
                    opportunities = self.find_underperformers(backtest_data, current_date)
                    
                    if len(opportunities) > 0 and current_capital > 1000:
                        # Enter new portfolio
                        investment_per_position = current_capital / len(opportunities)
                        
                        new_positions = []
                        for opp in opportunities:
                            position = {
                                'asset': opp['asset'],
                                'entry_date': current_date,
                                'entry_price': opp['price'],
                                'entry_deviation': opp['deviation'],
                                'investment': investment_per_position,
                                'shares': investment_per_position / opp['price']
                            }
                            new_positions.append(position)
                        
                        active_positions = new_positions
                        portfolio_entry_date = current_date
                        total_invested = sum(pos['investment'] for pos in active_positions)
                        current_capital -= total_invested
                        portfolio_count += 1
                        
                        print(f"\n📥 Portfolio #{portfolio_count} - {current_date.strftime('%Y-%m-%d')}")
                        print(f"    💰 Invested: ${total_invested:,.2f}")
                        print(f"    🎯 Assets: {', '.join([pos['asset'] for pos in active_positions])}")
                        print(f"    📊 Deviations: {[f\"{pos['entry_deviation']:.1f}%\" for pos in active_positions]}")
            
            # If we have active positions, check for exit conditions
            if len(active_positions) > 0:
                
                # Calculate current portfolio value
                portfolio_value, position_details = self.calculate_portfolio_value(
                    active_positions, backtest_data, current_date
                )
                
                total_invested = sum(pos['investment'] for pos in active_positions)
                portfolio_return = ((portfolio_value / total_invested) - 1) * 100
                hold_days = (current_date - portfolio_entry_date).days
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                if portfolio_return >= 50:
                    should_exit = True
                    exit_reason = "50% Target Hit"
                elif hold_days >= 365:
                    should_exit = True
                    exit_reason = "12 Month Limit"
                
                if should_exit:
                    # Exit portfolio
                    current_capital += portfolio_value
                    
                    # Record completed portfolio
                    individual_returns = {}
                    for asset, details in position_details.items():
                        individual_returns[asset] = details['return_pct']
                    
                    completed_portfolio = {
                        'portfolio_number': portfolio_count,
                        'entry_date': portfolio_entry_date,
                        'exit_date': current_date,
                        'hold_days': hold_days,
                        'invested': total_invested,
                        'exit_value': portfolio_value,
                        'return_pct': portfolio_return,
                        'exit_reason': exit_reason,
                        'assets': [pos['asset'] for pos in active_positions],
                        'individual_returns': individual_returns
                    }
                    
                    completed_portfolios.append(completed_portfolio)
                    
                    print(f"📤 Portfolio #{portfolio_count} EXIT - {current_date.strftime('%Y-%m-%d')}")
                    print(f"    📊 Return: {portfolio_return:+.1f}% in {hold_days} days")
                    print(f"    💰 Value: ${portfolio_value:,.2f} -> Total: ${current_capital:,.2f}")
                    print(f"    🚪 Reason: {exit_reason}")
                    
                    # Clear active positions
                    active_positions = []
                    portfolio_entry_date = None
                
                # Record daily tracking regardless of exit
                total_account_value = current_capital + portfolio_value
            else:
                # No active positions
                portfolio_value = 0
                total_account_value = current_capital
            
            # Daily tracking
            daily_tracking.append({
                'date': current_date,
                'cash': current_capital,
                'portfolio_value': portfolio_value,
                'total_value': total_account_value,
                'total_return_pct': ((total_account_value / self.initial_capital) - 1) * 100,
                'active_positions': len(active_positions),
                'completed_portfolios': len(completed_portfolios)
            })
            
            # Progress updates
            if current_date.month == 1 and current_date.day <= 7:  # New year
                total_return = ((total_account_value / self.initial_capital) - 1) * 100
                print(f"📅 {current_date.year}: ${total_account_value:,.2f} ({total_return:+.1f}%) | "
                      f"Completed: {len(completed_portfolios)} | Active: {len(active_positions)}")
        
        # Final calculations
        if len(active_positions) > 0:
            final_portfolio_value, _ = self.calculate_portfolio_value(
                active_positions, backtest_data, all_dates[-1]
            )
            final_value = current_capital + final_portfolio_value
        else:
            final_value = current_capital
        
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
        print("="*40)
        
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
            print(f"   Avg Return: {np.mean(returns):.2f}%")
            print(f"   Median Return: {np.median(returns):.2f}%")
            print(f"   Best: {max(returns):.2f}%")
            print(f"   Worst: {min(returns):.2f}%")
            print(f"   Win Rate: {len([r for r in returns if r > 0])/len(returns)*100:.1f}%")
            print(f"   Avg Hold: {np.mean(hold_times):.0f} days")
            
            # Exit reasons
            exit_reasons = {}
            for p in portfolios:
                reason = p['exit_reason']
                if reason not in exit_reasons:
                    exit_reasons[reason] = []
                exit_reasons[reason].append(p['return_pct'])
            
            print(f"\n🚪 EXIT REASONS:")
            for reason, rets in exit_reasons.items():
                avg_return = np.mean(rets)
                print(f"   {reason}: {len(rets)} portfolios, {avg_return:+.1f}% avg")
        
        return results
    
    def save_results(self, results):
        """Save results to CSV"""
        if results is None:
            return
        
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        
        # Save daily tracking
        if len(results['daily_tracking']) > 0:
            daily_df = pd.DataFrame(results['daily_tracking'])
            daily_file = os.path.join(results_dir, "FIXED_BACKTEST_DAILY_VALUES.csv")
            daily_df.to_csv(daily_file, index=False)
            print(f"✅ Daily values: {daily_file}")
        
        # Save completed portfolios
        if len(results['completed_portfolios']) > 0:
            portfolios_data = []
            trades_data = []
            
            for portfolio in results['completed_portfolios']:
                # Portfolio summary
                portfolios_data.append({
                    'portfolio_number': portfolio['portfolio_number'],
                    'entry_date': portfolio['entry_date'],
                    'exit_date': portfolio['exit_date'],
                    'hold_days': portfolio['hold_days'],
                    'invested': portfolio['invested'],
                    'exit_value': portfolio['exit_value'],
                    'return_pct': portfolio['return_pct'],
                    'exit_reason': portfolio['exit_reason'],
                    'num_assets': len(portfolio['assets'])
                })
                
                # Individual trades
                for asset in portfolio['assets']:
                    individual_return = portfolio['individual_returns'].get(asset, 0)
                    trades_data.append({
                        'portfolio_number': portfolio['portfolio_number'],
                        'asset': asset,
                        'entry_date': portfolio['entry_date'],
                        'exit_date': portfolio['exit_date'],
                        'individual_return_pct': individual_return,
                        'hold_days': portfolio['hold_days'],
                        'exit_reason': portfolio['exit_reason']
                    })
            
            # Save files
            portfolio_file = os.path.join(results_dir, "FIXED_BACKTEST_PORTFOLIOS.csv")
            pd.DataFrame(portfolios_data).to_csv(portfolio_file, index=False)
            print(f"✅ Portfolios: {portfolio_file}")
            
            trades_file = os.path.join(results_dir, "FIXED_BACKTEST_TRADES.csv")
            pd.DataFrame(trades_data).to_csv(trades_file, index=False)
            print(f"✅ Trades: {trades_file}")


def run_fixed_backtest():
    """Run the fixed backtest"""
    print("FIXED IWLS UNDERPERFORMANCE BACKTEST")
    print("="*50)
    
    backtester = FixedUnderperformanceBacktester(
        initial_capital=10000,
        max_positions=5
    )
    
    # Run with full data range (auto-detect)
    results = backtester.backtest_strategy()
    
    # Analyze and save
    backtester.analyze_results(results)
    backtester.save_results(results)
    
    print(f"\n✨ FIXED BACKTEST COMPLETE!")
    
    return results


if __name__ == "__main__":
    backtest_results = run_fixed_backtest()