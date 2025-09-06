import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UnderperformanceBacktester:
    """
    Simple backtest strategy with SPY comparison:
    1. Find 5 most negatively deviated stocks
    2. Split $10,000 equally amongst them ($2,000 each)
    3. Track portfolio value daily
    4. Exit when portfolio hits +50% total return
    5. Immediately re-enter with new worst 5 stocks
    6. Compare performance against SPY buy-and-hold for each period
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
        
        print(f"📂 Loading data: {input_file}")
        
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
        # Try both possible filenames
        spy_files = [
            os.path.join(data_dir, "BATS_SPY, 1W-3_daily.csv"),
            os.path.join(data_dir, "BATS_SPY, 1W-3.csv")
        ]
        
        spy_df = None
        for spy_file in spy_files:
            if os.path.exists(spy_file):
                print(f"📂 Loading SPY data: {spy_file}")
                
                try:
                    spy_df = pd.read_csv(spy_file)
                    
                    # Handle your data format - time column with Unix timestamp
                    if 'time' in spy_df.columns:
                        spy_df['date'] = pd.to_datetime(spy_df['time'], unit='s')
                    else:
                        spy_df['date'] = pd.to_datetime(spy_df['date'])
                    
                    spy_df = spy_df[['date', 'close']].rename(columns={'close': 'spy_price'})
                    spy_df = spy_df.sort_values('date')
                    spy_df = spy_df.dropna()  # Remove any NaN values
                    
                    print(f"✅ SPY data loaded: {len(spy_df):,} records")
                    print(f"📅 SPY date range: {spy_df['date'].min().strftime('%Y-%m-%d')} to {spy_df['date'].max().strftime('%Y-%m-%d')}")
                    
                    return spy_df
                except Exception as e:
                    print(f"⚠️ Error loading SPY data from {spy_file}: {e}")
                    continue
        
        print("⚠️ Warning: No SPY data file found")
        return None
    
    def interpolate_spy_daily(self, spy_df, all_dates):
        """Interpolate SPY prices to daily data for comparison"""
        if spy_df is None:
            return None
        
        print("📈 Creating daily SPY data...")
        
        # Create daily date range matching our backtest dates
        daily_df = pd.DataFrame({'date': all_dates})
        
        # Merge with SPY data
        merged_df = pd.merge(daily_df, spy_df, on='date', how='left')
        
        # Forward fill and backward fill to handle missing values
        merged_df['spy_price'] = merged_df['spy_price'].fillna(method='ffill').fillna(method='bfill')
        
        # Drop any remaining NaN values
        merged_df = merged_df.dropna(subset=['spy_price'])
        
        print(f"✅ SPY daily data created: {len(merged_df):,} trading days")
        
        return merged_df
    
    def get_spy_price_on_date(self, spy_daily, target_date, lookback_days=7):
        """Get SPY price on or near a specific date"""
        if spy_daily is None:
            return None
        
        # First try exact date
        exact_match = spy_daily[spy_daily['date'] == target_date]
        if len(exact_match) > 0:
            return exact_match.iloc[0]['spy_price']
        
        # If no exact match, look for closest date within lookback period
        start_date = target_date - timedelta(days=lookback_days)
        end_date = target_date + timedelta(days=lookback_days)
        
        nearby_data = spy_daily[
            (spy_daily['date'] >= start_date) & 
            (spy_daily['date'] <= end_date)
        ]
        
        if len(nearby_data) > 0:
            # Get closest date
            nearby_data['date_diff'] = abs((nearby_data['date'] - target_date).dt.days)
            closest = nearby_data.loc[nearby_data['date_diff'].idxmin()]
            return closest['spy_price']
        
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
        """Run the backtest with SPY comparison"""
        
        # Load and prepare data
        df = self.load_data()
        if df is None:
            return None
        
        data = self.prepare_data(df)
        
        # Load SPY data
        spy_df = self.load_spy_data()
        
        # Find start of data with IWLS values
        start_date = data['date'].min()
        end_date = data['date'].max()
        
        print(f"\n🚀 STARTING BACKTEST WITH SPY COMPARISON")
        print("="*70)
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Strategy: Buy {self.num_positions} worst performers, exit at +{self.profit_target}%")
        print(f"📅 Backtest period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get all unique trading dates
        all_dates = sorted(data['date'].unique())
        print(f"📊 Trading days: {len(all_dates):,}")
        
        # Prepare SPY comparison data
        spy_daily = self.interpolate_spy_daily(spy_df, all_dates)
        spy_start_price = self.get_spy_price_on_date(spy_daily, start_date)
        spy_shares = 0
        
        if spy_start_price is not None:
            spy_shares = self.initial_capital / spy_start_price
            print(f"📈 SPY comparison: ${spy_start_price:.2f}/share, {spy_shares:.2f} shares")
        else:
            print("⚠️ Warning: Could not find SPY start price")
        
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
        
        print(f"\n🔥 Portfolio #1 - {start_date.strftime('%Y-%m-%d')}")
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
                    # Calculate SPY performance for this period
                    spy_entry_price = self.get_spy_price_on_date(spy_daily, portfolio_entry_date)
                    spy_current_price = self.get_spy_price_on_date(spy_daily, current_date)
                    spy_period_return = 0.0
                    
                    if spy_entry_price is not None and spy_current_price is not None:
                        spy_period_return = ((spy_current_price / spy_entry_price) - 1) * 100
                    
                    # EXIT CURRENT PORTFOLIO
                    current_capital += portfolio_value
                    
                    # Record completed portfolio with SPY comparison
                    completed_portfolio = {
                        'portfolio_number': portfolio_number,
                        'entry_date': portfolio_entry_date,
                        'exit_date': current_date,
                        'hold_days': hold_days,
                        'invested': total_invested,
                        'exit_value': portfolio_value,
                        'return_pct': portfolio_return,
                        'assets': [pos['asset'] for pos in active_positions],
                        'entry_deviations': [pos['entry_deviation'] for pos in active_positions],
                        'spy_entry_price': spy_entry_price,
                        'spy_exit_price': spy_current_price,
                        'spy_period_return': spy_period_return,
                        'excess_return': portfolio_return - spy_period_return,
                        'outperformed_spy': portfolio_return > spy_period_return
                    }
                    
                    completed_portfolios.append(completed_portfolio)
                    
                    print(f"\n📤 Portfolio #{portfolio_number} EXIT - {current_date.strftime('%Y-%m-%d')}")
                    print(f"    📊 Strategy Return: {portfolio_return:+.1f}% in {hold_days} days")
                    print(f"    📈 SPY Return: {spy_period_return:+.1f}% in {hold_days} days")
                    print(f"    🎯 Excess Return: {portfolio_return - spy_period_return:+.1f}%")
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
            
            # Calculate SPY value for daily tracking
            spy_current_value = self.initial_capital
            spy_current_price = self.get_spy_price_on_date(spy_daily, current_date)
            
            if spy_start_price is not None and spy_current_price is not None:
                spy_current_value = spy_shares * spy_current_price
            
            # Daily tracking
            total_account_value = current_capital + portfolio_value
            spy_return_pct = ((spy_current_value / self.initial_capital) - 1) * 100 if spy_current_value != self.initial_capital else 0
            
            daily_tracking.append({
                'date': current_date,
                'cash': current_capital,
                'portfolio_value': portfolio_value,
                'total_value': total_account_value,
                'total_return_pct': ((total_account_value / self.initial_capital) - 1) * 100,
                'spy_value': spy_current_value,
                'spy_return_pct': spy_return_pct,
                'excess_return_pct': ((total_account_value / self.initial_capital) - 1) * 100 - spy_return_pct,
                'active_positions': len(active_positions),
                'completed_portfolios': len(completed_portfolios)
            })
            
            # Progress updates (monthly) with SPY comparison
            if current_date.day == 1:  # First day of month
                total_return = ((total_account_value / self.initial_capital) - 1) * 100
                excess_monthly = total_return - spy_return_pct
                print(f"📅 {current_date.strftime('%Y-%m')}: Strategy: ${total_account_value:,.0f} ({total_return:+.1f}%) | "
                      f"SPY: ${spy_current_value:,.0f} ({spy_return_pct:+.1f}%) | "
                      f"Excess: {excess_monthly:+.1f}% | Completed: {len(completed_portfolios)}")
        
        # Final calculations
        final_value = current_capital
        if len(active_positions) > 0:
            # Calculate final portfolio value
            asset_list = [pos['asset'] for pos in active_positions]
            final_prices = self.get_current_prices(data, asset_list, all_dates[-1])
            final_portfolio_value = self.calculate_portfolio_value(active_positions, final_prices)
            final_value += final_portfolio_value
        
        # Final SPY value
        final_spy_value = self.initial_capital
        final_spy_price = self.get_spy_price_on_date(spy_daily, all_dates[-1])
        
        if spy_start_price is not None and final_spy_price is not None:
            final_spy_value = spy_shares * final_spy_price
        
        return {
            'completed_portfolios': completed_portfolios,
            'daily_tracking': daily_tracking,
            'final_value': final_value,
            'total_return': ((final_value / self.initial_capital) - 1) * 100,
            'final_spy_value': final_spy_value,
            'spy_total_return': ((final_spy_value / self.initial_capital) - 1) * 100,
            'active_positions': active_positions,
            'spy_start_price': spy_start_price,
            'final_spy_price': final_spy_price
        }
    
    def analyze_results(self, results):
        """Analyze backtest results with SPY comparison"""
        if results is None:
            return
        
        print(f"\n📊 BACKTEST RESULTS WITH SPY COMPARISON")
        print("="*60)
        
        # Overall performance comparison
        print(f"\n💰 OVERALL PERFORMANCE:")
        print(f"   Strategy Initial: ${self.initial_capital:,.2f}")
        print(f"   Strategy Final:   ${results['final_value']:,.2f}")
        print(f"   Strategy Return:  {results['total_return']:+.2f}%")
        print(f"   ")
        print(f"   SPY Initial:      ${self.initial_capital:,.2f}")
        print(f"   SPY Final:        ${results['final_spy_value']:,.2f}")
        print(f"   SPY Return:       {results['spy_total_return']:+.2f}%")
        print(f"   ")
        excess_return = results['total_return'] - results['spy_total_return']
        print(f"   Excess Return:    {excess_return:+.2f}%")
        
        # Annualized returns
        if len(results['daily_tracking']) > 0:
            days = len(results['daily_tracking'])
            years = days / 365.25
            if years > 0:
                strategy_annualized = ((results['final_value'] / self.initial_capital) ** (1/years) - 1) * 100
                spy_annualized = ((results['final_spy_value'] / self.initial_capital) ** (1/years) - 1) * 100
                print(f"   Period: {years:.2f} years")
                print(f"   Strategy Annualized: {strategy_annualized:+.2f}%")
                print(f"   SPY Annualized:      {spy_annualized:+.2f}%")
                print(f"   Excess Annualized:   {strategy_annualized - spy_annualized:+.2f}%")
        
        # Portfolio statistics with SPY comparison
        portfolios = results['completed_portfolios']
        if len(portfolios) > 0:
            returns = [p['return_pct'] for p in portfolios]
            spy_returns = [p['spy_period_return'] for p in portfolios]
            excess_returns = [p['excess_return'] for p in portfolios]
            hold_times = [p['hold_days'] for p in portfolios]
            
            print(f"\n📈 PORTFOLIO STATISTICS:")
            print(f"   Completed: {len(portfolios)}")
            print(f"   All hit {self.profit_target}% target")
            print(f"   ")
            print(f"   Strategy Returns:")
            print(f"     Average: {np.mean(returns):.1f}%")
            print(f"     Min: {min(returns):.1f}%")
            print(f"     Max: {max(returns):.1f}%")
            print(f"   ")
            print(f"   SPY Period Returns:")
            print(f"     Average: {np.mean(spy_returns):.1f}%")
            print(f"     Min: {min(spy_returns):.1f}%")
            print(f"     Max: {max(spy_returns):.1f}%")
            print(f"   ")
            print(f"   Excess Returns:")
            print(f"     Average: {np.mean(excess_returns):.1f}%")
            print(f"     Min: {min(excess_returns):.1f}%")
            print(f"     Max: {max(excess_returns):.1f}%")
            print(f"   ")
            print(f"   Hold Times:")
            print(f"     Average: {np.mean(hold_times):.0f} days")
            print(f"     Min: {min(hold_times):.0f} days")
            print(f"     Max: {max(hold_times):.0f} days")
            
            # Win rate vs SPY
            outperformed_spy = sum(1 for p in portfolios if p.get('outperformed_spy', False))
            win_rate_vs_spy = (outperformed_spy / len(portfolios)) * 100
            print(f"   ")
            print(f"   Performance vs SPY:")
            print(f"     Outperformed: {outperformed_spy}/{len(portfolios)} ({win_rate_vs_spy:.1f}%)")
            
            # Portfolio comparison table
            print(f"\n📋 PORTFOLIO vs SPY COMPARISON:")
            print("="*90)
            print(f"{'Port#':>5} | {'Entry Date':>12} | {'Exit Date':>12} | {'Days':>5} | "
                  f"{'Strategy':>9} | {'SPY':>9} | {'Excess':>8} | {'Winner':>8}")
            print("-"*90)
            
            for p in portfolios:
                excess = p.get('excess_return', p['return_pct'] - p['spy_period_return'])
                winner = "Strategy" if excess > 0 else "SPY" if excess < 0 else "Tie"
                
                print(f"#{p['portfolio_number']:>4} | {p['entry_date'].strftime('%Y-%m-%d'):>12} | "
                      f"{p['exit_date'].strftime('%Y-%m-%d'):>12} | {p['hold_days']:>5} | "
                      f"{p['return_pct']:>8.1f}% | {p['spy_period_return']:>8.1f}% | "
                      f"{excess:>+7.1f}% | {winner:>8}")
            
            # Summary row
            print("-"*90)
            avg_strategy = np.mean([p['return_pct'] for p in portfolios])
            avg_spy = np.mean([p['spy_period_return'] for p in portfolios])
            avg_excess = avg_strategy - avg_spy
            
            print(f"{'AVG':>5} | {'':>12} | {'':>12} | {np.mean([p['hold_days'] for p in portfolios]):>5.0f} | "
                  f"{avg_strategy:>8.1f}% | {avg_spy:>8.1f}% | {avg_excess:>+7.1f}% | {'':>8}")
            
            # Show first few portfolios
            print(f"\n🎯 FIRST 5 PORTFOLIOS WITH SPY COMPARISON:")
            for i, p in enumerate(portfolios[:5]):
                excess = p.get('excess_return', p['return_pct'] - p['spy_period_return'])
                print(f"   #{i+1}: {p['entry_date'].strftime('%Y-%m-%d')} -> {p['exit_date'].strftime('%Y-%m-%d')} "
                      f"({p['hold_days']} days)")
                print(f"        Strategy: {p['return_pct']:.1f}%, SPY: {p['spy_period_return']:.1f}%, "
                      f"Excess: {excess:+.1f}%")
                print(f"        Assets: {p['assets']}")
        
        return results
    
    def save_results(self, results):
        """Save results to CSV with SPY comparison"""
        if results is None:
            return
        
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        
        # Save daily tracking with SPY comparison
        if len(results['daily_tracking']) > 0:
            daily_df = pd.DataFrame(results['daily_tracking'])
            daily_file = os.path.join(results_dir, "SIMPLE_BACKTEST_DAILY_WITH_SPY.csv")
            daily_df.to_csv(daily_file, index=False)
            print(f"✅ Daily values with SPY: {daily_file}")
        
        # Save completed portfolios with SPY comparison
        if len(results['completed_portfolios']) > 0:
            portfolios_df = pd.DataFrame(results['completed_portfolios'])
            # Convert lists to strings for CSV
            portfolios_df['assets'] = portfolios_df['assets'].astype(str)
            portfolios_df['entry_deviations'] = portfolios_df['entry_deviations'].astype(str)
            portfolio_file = os.path.join(results_dir, "SIMPLE_BACKTEST_PORTFOLIOS_WITH_SPY.csv")
            portfolios_df.to_csv(portfolio_file, index=False)
            print(f"✅ Portfolios with SPY: {portfolio_file}")
        
        # Create summary statistics
        summary_stats = {
            'metric': [
                'Initial Capital',
                'Final Strategy Value',
                'Final SPY Value',
                'Strategy Total Return %',
                'SPY Total Return %',
                'Excess Return %',
                'Total Portfolios',
                'Portfolios Outperforming SPY',
                'Win Rate vs SPY %',
                'Average Portfolio Return %',
                'Average SPY Period Return %',
                'Average Excess Return %'
            ],
            'value': [
                self.initial_capital,
                results['final_value'],
                results['final_spy_value'],
                results['total_return'],
                results['spy_total_return'],
                results['total_return'] - results['spy_total_return'],
                len(results['completed_portfolios']),
                sum(1 for p in results['completed_portfolios'] if p.get('outperformed_spy', False)),
                (sum(1 for p in results['completed_portfolios'] if p.get('outperformed_spy', False)) / 
                 len(results['completed_portfolios']) * 100) if results['completed_portfolios'] else 0,
                np.mean([p['return_pct'] for p in results['completed_portfolios']]) if results['completed_portfolios'] else 0,
                np.mean([p['spy_period_return'] for p in results['completed_portfolios']]) if results['completed_portfolios'] else 0,
                np.mean([p.get('excess_return', 0) for p in results['completed_portfolios']]) if results['completed_portfolios'] else 0
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        summary_file = os.path.join(results_dir, "SIMPLE_BACKTEST_SUMMARY_WITH_SPY.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Summary statistics: {summary_file}")


def run_backtest():
    """Run the underperformance backtest with SPY comparison"""
    print("IWLS UNDERPERFORMANCE BACKTEST WITH SPY COMPARISON")
    print("="*60)
    
    backtester = UnderperformanceBacktester(
        initial_capital=10000,
        num_positions=9,
        profit_target=15.0  # 20% profit target
    )
    
    # Run backtest
    results = backtester.backtest_strategy()
    
    if results is None:
        print("❌ Backtest failed - no results to analyze")
        return None
    
    # Analyze and save
    backtester.analyze_results(results)
    backtester.save_results(results)
    
    print(f"\n✨ BACKTEST WITH SPY COMPARISON COMPLETE!")
    print("="*60)
    print(f"📊 Key Features Added:")
    print(f"   • SPY performance tracking for each portfolio period")
    print(f"   • Excess return calculation (Strategy - SPY)")
    print(f"   • Win rate analysis vs SPY")
    print(f"   • Portfolio-by-portfolio comparison table")
    print(f"   • Enhanced monthly progress with SPY tracking")
    print(f"   • Summary statistics with SPY metrics")
    print(f"\n📁 Output Files:")
    print(f"   • SIMPLE_BACKTEST_DAILY_WITH_SPY.csv")
    print(f"   • SIMPLE_BACKTEST_PORTFOLIOS_WITH_SPY.csv") 
    print(f"   • SIMPLE_BACKTEST_SUMMARY_WITH_SPY.csv")
    
    return results


if __name__ == "__main__":
    backtest_results = run_backtest()