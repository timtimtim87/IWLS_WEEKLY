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
    7. ENHANCED SPY COMPARISON: Track SPY buy-and-hold performance for comparison with detailed period analysis
    """
    
    def __init__(self, initial_capital=10000, num_positions=3, profit_target=500.0, 
                 rebalance_threshold=500.0, rebalance_wait_days=365):
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
    
    def load_spy_data(self):
        """Load SPY data for comparison - Updated to handle your actual data format"""
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
                    print(f"💰 SPY price range: ${spy_df['spy_price'].min():.2f} to ${spy_df['spy_price'].max():.2f}")
                    
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
        
        print("📈 Interpolating SPY data to daily...")
        
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
    
    def backtest_strategy(self):
        """Run the backtest with enhanced SPY comparison"""
        
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
        
        print(f"\n🚀 STARTING ENHANCED BACKTEST WITH SPY COMPARISON")
        print("="*80)
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Strategy: Buy {self.num_positions} worst performers, exit at +{self.profit_target}%")
        print(f"🔄 Rebalance: If top {self.num_positions} deviations sum > {self.rebalance_threshold}% and {self.rebalance_wait_days}+ days passed")
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
            print(f"📈 SPY comparison setup: ${spy_start_price:.2f}/share, {spy_shares:.2f} shares")
        else:
            print("⚠️ Warning: Could not find SPY start price - SPY comparison will be limited")
        
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
                
                # Calculate SPY performance for this period
                spy_entry_price = self.get_spy_price_on_date(spy_daily, portfolio_entry_date)
                spy_current_price = self.get_spy_price_on_date(spy_daily, current_date)
                spy_period_return = 0.0
                
                if spy_entry_price is not None and spy_current_price is not None:
                    spy_period_return = ((spy_current_price / spy_entry_price) - 1) * 100
                
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
                    
                    # Record completed portfolio with enhanced SPY comparison
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
                        'entry_deviations': [pos['entry_deviation'] for pos in active_positions],
                        'spy_entry_price': spy_entry_price,
                        'spy_exit_price': spy_current_price,
                        'spy_period_return': spy_period_return,
                        'excess_return': portfolio_return - spy_period_return,
                        'outperformed_spy': portfolio_return > spy_period_return
                    }
                    
                    completed_portfolios.append(completed_portfolio)
                    
                    print(f"\n📤 Portfolio #{portfolio_number} EXIT - {current_date.strftime('%Y-%m-%d')}")
                    print(f"    📊 Reason: {exit_reason}")
                    if should_rebalance:
                        days_since_rebalance = (current_date - last_rebalance_date).days if last_rebalance_date else 0
                        print(f"    🔄 Days since last rebalance: {days_since_rebalance}")
                        print(f"    📉 Current worst {self.num_positions} deviation sum: {total_deviation:.1f}%")
                    print(f"    📈 Strategy Return: {portfolio_return:+.1f}% in {hold_days} days")
                    print(f"    📈 SPY Return: {spy_period_return:+.1f}% in {hold_days} days")
                    print(f"    🎯 Excess Return: {portfolio_return - spy_period_return:+.1f}%")
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
                        
                        print(f"\n🔥 Portfolio #{portfolio_number} ENTRY - {current_date.strftime('%Y-%m-%d')}")
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
                'active_positions': len(active_positions),
                'completed_portfolios': len(completed_portfolios),
                'portfolio_number': portfolio_number if len(active_positions) > 0 else 0,
                'spy_price': spy_current_price
            })
            
            # Progress updates (monthly) - now with enhanced SPY comparison
            if current_date.day == 1:  # First day of month
                total_return = ((total_account_value / self.initial_capital) - 1) * 100
                excess_monthly = total_return - spy_return_pct
                print(f"📅 {current_date.strftime('%Y-%m')}: Strategy: ${total_account_value:,.0f} ({total_return:+.1f}%) | "
                      f"SPY: ${spy_current_value:,.0f} ({spy_return_pct:+.1f}%) | "
                      f"Excess: {excess_monthly:+.1f}% | "
                      f"Completed: {len(completed_portfolios)} | Rebalances: {len(rebalance_events)}")
        
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
            'rebalance_events': rebalance_events,
            'final_value': final_value,
            'total_return': ((final_value / self.initial_capital) - 1) * 100,
            'final_spy_value': final_spy_value,
            'spy_total_return': ((final_spy_value / self.initial_capital) - 1) * 100,
            'active_positions': active_positions,
            'spy_start_price': spy_start_price,
            'final_spy_price': final_spy_price
        }
    
    def analyze_results(self, results):
        """Enhanced analysis with detailed SPY comparison"""
        if results is None:
            return
        
        print(f"\n📊 ENHANCED BACKTEST RESULTS WITH SPY COMPARISON")
        print("="*80)
        
        # Overall performance comparison
        print(f"\n💰 OVERALL PERFORMANCE COMPARISON:")
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
        
        if results.get('spy_start_price') and results.get('final_spy_price'):
            print(f"   SPY Price: ${results['spy_start_price']:.2f} -> ${results['final_spy_price']:.2f}")
        
        # Annualized returns
        if len(results['daily_tracking']) > 0:
            days = len(results['daily_tracking'])
            years = days / 365.25
            if years > 0:
                strategy_annualized = ((results['final_value'] / self.initial_capital) ** (1/years) - 1) * 100
                spy_annualized = ((results['final_spy_value'] / self.initial_capital) ** (1/years) - 1) * 100
                print(f"   Period: {years:.2f} years ({days} days)")
                print(f"   Strategy Annualized: {strategy_annualized:+.2f}%")
                print(f"   SPY Annualized:      {spy_annualized:+.2f}%")
                print(f"   Excess Annualized:   {strategy_annualized - spy_annualized:+.2f}%")
        
        # Enhanced portfolio statistics
        portfolios = results['completed_portfolios']
        if len(portfolios) > 0:
            profit_exits = [p for p in portfolios if 'PROFIT_TARGET' in p.get('exit_reason', '')]
            rebalance_exits = [p for p in portfolios if 'REBALANCE' in p.get('exit_reason', '')]
            
            returns = [p['return_pct'] for p in portfolios]
            spy_returns = [p['spy_period_return'] for p in portfolios]
            excess_returns = [p['excess_return'] for p in portfolios]
            hold_times = [p['hold_days'] for p in portfolios]
            
            print(f"\n📈 DETAILED PORTFOLIO STATISTICS:")
            print(f"   Total Portfolios: {len(portfolios)}")
            print(f"   Profit Target Exits: {len(profit_exits)} ({len(profit_exits)/len(portfolios)*100:.1f}%)")
            print(f"   Rebalance Exits: {len(rebalance_exits)} ({len(rebalance_exits)/len(portfolios)*100:.1f}%)")
            print(f"   ")
            print(f"   Strategy Returns:")
            print(f"     Average: {np.mean(returns):.1f}%")
            print(f"     Median:  {np.median(returns):.1f}%")
            print(f"     Min:     {min(returns):.1f}%")
            print(f"     Max:     {max(returns):.1f}%")
            print(f"   ")
            print(f"   SPY Period Returns:")
            print(f"     Average: {np.mean(spy_returns):.1f}%")
            print(f"     Median:  {np.median(spy_returns):.1f}%")
            print(f"     Min:     {min(spy_returns):.1f}%")
            print(f"     Max:     {max(spy_returns):.1f}%")
            print(f"   ")
            print(f"   Excess Returns:")
            print(f"     Average: {np.mean(excess_returns):.1f}%")
            print(f"     Median:  {np.median(excess_returns):.1f}%")
            print(f"     Min:     {min(excess_returns):.1f}%")
            print(f"     Max:     {max(excess_returns):.1f}%")
            print(f"   ")
            print(f"   Hold Times:")
            print(f"     Average: {np.mean(hold_times):.0f} days")
            print(f"     Median:  {np.median(hold_times):.0f} days")
            print(f"     Min:     {min(hold_times):.0f} days")
            print(f"     Max:     {max(hold_times):.0f} days")
            
            # Win rate analysis
            outperformed_spy = sum(1 for p in portfolios if p.get('outperformed_spy', False))
            win_rate_vs_spy = (outperformed_spy / len(portfolios)) * 100
            print(f"   ")
            print(f"   Performance vs SPY:")
            print(f"     Outperformed: {outperformed_spy}/{len(portfolios)} ({win_rate_vs_spy:.1f}%)")
            print(f"     Underperformed: {len(portfolios) - outperformed_spy}/{len(portfolios)} ({100 - win_rate_vs_spy:.1f}%)")
            
            # Best and worst performing portfolios
            best_portfolio = max(portfolios, key=lambda x: x['excess_return'])
            worst_portfolio = min(portfolios, key=lambda x: x['excess_return'])
            
            print(f"   ")
            print(f"   Best vs SPY: Portfolio #{best_portfolio['portfolio_number']} "
                  f"({best_portfolio['entry_date'].strftime('%Y-%m-%d')} to {best_portfolio['exit_date'].strftime('%Y-%m-%d')})")
            print(f"     Strategy: {best_portfolio['return_pct']:+.1f}%, SPY: {best_portfolio['spy_period_return']:+.1f}%, "
                  f"Excess: {best_portfolio['excess_return']:+.1f}%")
            
            print(f"   Worst vs SPY: Portfolio #{worst_portfolio['portfolio_number']} "
                  f"({worst_portfolio['entry_date'].strftime('%Y-%m-%d')} to {worst_portfolio['exit_date'].strftime('%Y-%m-%d')})")
            print(f"     Strategy: {worst_portfolio['return_pct']:+.1f}%, SPY: {worst_portfolio['spy_period_return']:+.1f}%, "
                  f"Excess: {worst_portfolio['excess_return']:+.1f}%")
        
        # Rebalance statistics
        rebalances = results['rebalance_events']
        if len(rebalances) > 0:
            print(f"\n🔄 REBALANCE STATISTICS:")
            print(f"   Total Rebalances: {len(rebalances)}")
            avg_deviation = np.mean([r['total_deviation'] for r in rebalances])
            print(f"   Average Trigger Deviation: {avg_deviation:.1f}%")
            
            if len(rebalances) >= 3:
                print(f"\n   📋 RECENT REBALANCES:")
                for i, r in enumerate(rebalances[:3]):
                    print(f"   #{i+1}: {r['date'].strftime('%Y-%m-%d')} | "
                          f"Deviation: {r['total_deviation']:.1f}% | "
                          f"Days since last: {r['days_since_last']}")
        
        # Enhanced portfolio comparison table
        if len(portfolios) > 0:
            print(f"\n📋 DETAILED PORTFOLIO vs SPY COMPARISON TABLE:")
            print("="*110)
            print(f"{'Port#':>5} | {'Entry Date':>12} | {'Exit Date':>12} | {'Days':>5} | {'Reason':>15} | "
                  f"{'Strategy':>9} | {'SPY':>9} | {'Excess':>8} | {'Winner':>8}")
            print("-"*110)
            
            for p in portfolios:
                reason_short = p['exit_reason'].split('(')[0].strip()
                excess = p.get('excess_return', p['return_pct'] - p['spy_period_return'])
                winner = "Strategy" if excess > 0 else "SPY" if excess < 0 else "Tie"
                
                print(f"#{p['portfolio_number']:>4} | {p['entry_date'].strftime('%Y-%m-%d'):>12} | "
                      f"{p['exit_date'].strftime('%Y-%m-%d'):>12} | {p['hold_days']:>5} | "
                      f"{reason_short:>15} | {p['return_pct']:>8.1f}% | "
                      f"{p['spy_period_return']:>8.1f}% | {excess:>+7.1f}% | {winner:>8}")
            
            # Summary row
            print("-"*110)
            avg_strategy = np.mean([p['return_pct'] for p in portfolios])
            avg_spy = np.mean([p['spy_period_return'] for p in portfolios])
            avg_excess = avg_strategy - avg_spy
            
            print(f"{'AVG':>5} | {'':>12} | {'':>12} | {np.mean([p['hold_days'] for p in portfolios]):>5.0f} | "
                  f"{'':>15} | {avg_strategy:>8.1f}% | {avg_spy:>8.1f}% | {avg_excess:>+7.1f}% | {'':>8}")
        
        # Monthly performance analysis
        if len(results['daily_tracking']) > 0:
            daily_df = pd.DataFrame(results['daily_tracking'])
            daily_df['year_month'] = daily_df['date'].dt.to_period('M')
            
            # Get monthly end values
            monthly_data = daily_df.groupby('year_month').last().reset_index()
            
            if len(monthly_data) > 1:
                print(f"\n📅 MONTHLY PERFORMANCE COMPARISON:")
                print("="*80)
                print(f"{'Month':>10} | {'Strategy':>12} | {'SPY':>12} | {'Strategy %':>12} | "
                      f"{'SPY %':>12} | {'Excess %':>10}")
                print("-"*80)
                
                for i, row in monthly_data.iterrows():
                    if i == 0:  # Skip first month as we need previous month for comparison
                        continue
                    
                    prev_row = monthly_data.iloc[i-1]
                    
                    strategy_monthly = ((row['total_value'] / prev_row['total_value']) - 1) * 100
                    spy_monthly = ((row['spy_value'] / prev_row['spy_value']) - 1) * 100
                    excess_monthly = strategy_monthly - spy_monthly
                    
                    print(f"{str(row['year_month']):>10} | ${row['total_value']:>10,.0f} | "
                          f"${row['spy_value']:>10,.0f} | {strategy_monthly:>+10.1f}% | "
                          f"{spy_monthly:>+10.1f}% | {excess_monthly:>+8.1f}%")
        
        return results
    
    def save_results(self, results):
        """Save enhanced results to CSV files"""
        if results is None:
            return
        
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        
        # Save enhanced daily tracking
        if len(results['daily_tracking']) > 0:
            daily_df = pd.DataFrame(results['daily_tracking'])
            # Add additional calculated columns
            daily_df['excess_return_pct'] = daily_df['total_return_pct'] - daily_df['spy_return_pct']
            daily_df['strategy_outperforming'] = daily_df['excess_return_pct'] > 0
            
            daily_file = os.path.join(results_dir, "ENHANCED_BACKTEST_DAILY_WITH_SPY.csv")
            daily_df.to_csv(daily_file, index=False)
            print(f"✅ Enhanced daily tracking: {daily_file}")
        
        # Save enhanced portfolio results
        if len(results['completed_portfolios']) > 0:
            portfolios_df = pd.DataFrame(results['completed_portfolios'])
            # Convert lists to strings for CSV
            portfolios_df['assets'] = portfolios_df['assets'].astype(str)
            portfolios_df['entry_deviations'] = portfolios_df['entry_deviations'].astype(str)
            
            portfolio_file = os.path.join(results_dir, "ENHANCED_BACKTEST_PORTFOLIOS_WITH_SPY.csv")
            portfolios_df.to_csv(portfolio_file, index=False)
            print(f"✅ Enhanced portfolios: {portfolio_file}")
        
        # Save rebalance events
        if len(results['rebalance_events']) > 0:
            rebalance_df = pd.DataFrame(results['rebalance_events'])
            # Convert lists to strings for CSV
            rebalance_df['old_assets'] = rebalance_df['old_assets'].astype(str)
            rebalance_df['new_assets'] = rebalance_df['new_assets'].astype(str)
            rebalance_file = os.path.join(results_dir, "ENHANCED_BACKTEST_REBALANCES.csv")
            rebalance_df.to_csv(rebalance_file, index=False)
            print(f"✅ Enhanced rebalances: {rebalance_file}")
        
        # Create summary statistics file
        summary_stats = {
            'metric': [
                'Initial Capital',
                'Final Strategy Value',
                'Final SPY Value',
                'Strategy Total Return %',
                'SPY Total Return %',
                'Excess Return %',
                'Total Portfolios',
                'Profit Target Exits',
                'Rebalance Exits',
                'Portfolios Outperforming SPY',
                'Win Rate vs SPY %',
                'Average Portfolio Return %',
                'Average SPY Period Return %',
                'Average Excess Return %',
                'Total Rebalances'
            ],
            'value': [
                self.initial_capital,
                results['final_value'],
                results['final_spy_value'],
                results['total_return'],
                results['spy_total_return'],
                results['total_return'] - results['spy_total_return'],
                len(results['completed_portfolios']),
                len([p for p in results['completed_portfolios'] if 'PROFIT_TARGET' in p.get('exit_reason', '')]),
                len([p for p in results['completed_portfolios'] if 'REBALANCE' in p.get('exit_reason', '')]),
                sum(1 for p in results['completed_portfolios'] if p.get('outperformed_spy', False)),
                (sum(1 for p in results['completed_portfolios'] if p.get('outperformed_spy', False)) / 
                 len(results['completed_portfolios']) * 100) if results['completed_portfolios'] else 0,
                np.mean([p['return_pct'] for p in results['completed_portfolios']]) if results['completed_portfolios'] else 0,
                np.mean([p['spy_period_return'] for p in results['completed_portfolios']]) if results['completed_portfolios'] else 0,
                np.mean([p.get('excess_return', 0) for p in results['completed_portfolios']]) if results['completed_portfolios'] else 0,
                len(results['rebalance_events'])
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        summary_file = os.path.join(results_dir, "ENHANCED_BACKTEST_SUMMARY_STATS.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Summary statistics: {summary_file}")


def run_enhanced_backtest():
    """Run the enhanced underperformance backtest with comprehensive SPY comparison"""
    print("🚀 IWLS ENHANCED UNDERPERFORMANCE BACKTEST WITH COMPREHENSIVE SPY COMPARISON")
    print("="*90)
    
    backtester = UnderperformanceBacktester(
        initial_capital=10000,
        num_positions=1,  # Number of worst performing stocks to buy
        profit_target=100.0,  # 100% profit target
        rebalance_threshold=65.0,  # Rebalance if total deviation > 65%
        rebalance_wait_days=300  # Wait 300 days between rebalances
    )
    
    # Run enhanced backtest
    results = backtester.backtest_strategy()
    
    if results is None:
        print("❌ Backtest failed - no results to analyze")
        return None
    
    # Analyze and save with enhanced features
    backtester.analyze_results(results)
    backtester.save_results(results)
    
    print(f"\n✨ ENHANCED BACKTEST WITH COMPREHENSIVE SPY COMPARISON COMPLETE!")
    print("="*80)
    print(f"📊 Enhanced Features Added:")
    print(f"   • Detailed SPY performance tracking for each portfolio period")
    print(f"   • Excess return calculation (Strategy - SPY) for every portfolio")
    print(f"   • Win rate analysis showing how often strategy beats SPY")
    print(f"   • Monthly performance breakdown comparing both strategies")
    print(f"   • Enhanced summary statistics with SPY comparison metrics")
    print(f"   • Best/worst performing portfolios vs SPY identification")
    print(f"   • Comprehensive CSV output files for further analysis")
    print(f"\n📁 Output Files:")
    print(f"   • ENHANCED_BACKTEST_DAILY_WITH_SPY.csv - Daily performance tracking")
    print(f"   • ENHANCED_BACKTEST_PORTFOLIOS_WITH_SPY.csv - Portfolio-by-portfolio comparison")
    print(f"   • ENHANCED_BACKTEST_SUMMARY_STATS.csv - Key performance metrics")
    print(f"   • ENHANCED_BACKTEST_REBALANCES.csv - Rebalancing events")
    
    return results


if __name__ == "__main__":
    enhanced_results = run_enhanced_backtest()