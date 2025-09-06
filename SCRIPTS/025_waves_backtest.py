import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WaveRangeBacktester:
    """
    Wave Range Strategy:
    1. Find N assets with highest 6-month rolling ranges (biggest waves)
    2. Split capital equally amongst them
    3. Track portfolio value daily
    4. Exit when portfolio hits +X% total return
    5. Immediately re-enter with new highest wave assets
    6. Compare performance against SPY
    """
    
    def __init__(self, initial_capital=10000, num_positions=3, profit_target=50.0, 
                 min_range_threshold=10.0):
        self.initial_capital = initial_capital
        self.num_positions = num_positions
        self.profit_target = profit_target
        self.min_range_threshold = min_range_threshold  # Minimum 6m range % to consider
        
    def load_data(self):
        """Load the enhanced IWLS data with wave analysis"""
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        input_file = os.path.join(results_dir, "IWLS_WITH_WAVE_ANALYSIS.csv")
        
        if not os.path.exists(input_file):
            print(f"❌ Error: Enhanced wave data not found at {input_file}")
            print("   Please run the wave enhancement script first.")
            return None
        
        print(f"📂 Loading enhanced wave data: {input_file}")
        
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
        
        spy_df = None
        for spy_file in spy_files:
            if os.path.exists(spy_file):
                print(f"📂 Loading SPY data: {spy_file}")
                
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
                    print(f"⚠️ Error loading SPY data from {spy_file}: {e}")
                    continue
        
        print("⚠️ Warning: No SPY data file found")
        return None
    
    def interpolate_spy_daily(self, spy_df, all_dates):
        """Interpolate SPY prices to daily data for comparison"""
        if spy_df is None:
            return None
        
        print("📈 Interpolating SPY data to daily...")
        
        daily_df = pd.DataFrame({'date': all_dates})
        merged_df = pd.merge(daily_df, spy_df, on='date', how='left')
        merged_df['spy_price'] = merged_df['spy_price'].fillna(method='ffill').fillna(method='bfill')
        merged_df = merged_df.dropna(subset=['spy_price'])
        
        print(f"✅ SPY daily data created: {len(merged_df):,} trading days")
        return merged_df
    
    def prepare_data(self, df):
        """Prepare and clean data for backtesting"""
        print("🔧 Preparing wave trading data...")
        
        # Keep only records with wave analysis and price data
        required_columns = ['rolling_range_pct_6_month', 'price', 'position_pct_6_month']
        clean_data = df.dropna(subset=required_columns).copy()
        
        # Filter out tiny ranges
        clean_data = clean_data[clean_data['rolling_range_pct_6_month'] >= self.min_range_threshold].copy()
        
        # Sort chronologically
        clean_data = clean_data.sort_values(['date', 'asset']).reset_index(drop=True)
        
        print(f"📊 Clean wave records: {len(clean_data):,}")
        print(f"📅 Date range: {clean_data['date'].min().strftime('%Y-%m-%d')} to {clean_data['date'].max().strftime('%Y-%m-%d')}")
        print(f"🏢 Unique assets: {clean_data['asset'].nunique()}")
        print(f"🌊 Range threshold: ≥{self.min_range_threshold}%")
        
        return clean_data
    
    def find_highest_wave_assets(self, df, current_date, lookback_days=7):
        """Find the N assets with highest 6-month rolling ranges"""
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
        
        # Sort by highest rolling range (biggest waves)
        highest_waves = latest_by_asset.sort_values('rolling_range_pct_6_month', ascending=False)
        
        # Return top N highest wave assets
        opportunities = []
        for _, row in highest_waves.head(self.num_positions).iterrows():
            opportunities.append({
                'asset': row['asset'],
                'entry_price': row['price'],
                'rolling_range_pct': row['rolling_range_pct_6_month'],
                'position_pct': row['position_pct_6_month'],
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
        """Create position list from wave opportunities"""
        if len(opportunities) == 0:
            return []
        
        investment_per_position = available_capital / len(opportunities)
        positions = []
        
        for opp in opportunities:
            position = {
                'asset': opp['asset'],
                'entry_date': opp['entry_date'],
                'entry_price': opp['entry_price'],
                'rolling_range_pct': opp['rolling_range_pct'],
                'position_pct': opp['position_pct'],
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
            nearby_data['date_diff'] = abs((nearby_data['date'] - target_date).dt.days)
            closest = nearby_data.loc[nearby_data['date_diff'].idxmin()]
            return closest['spy_price']
        
        return None
    
    def backtest_strategy(self):
        """Run the wave range backtest with SPY comparison"""
        
        # Load and prepare data
        df = self.load_data()
        if df is None:
            return None
        
        data = self.prepare_data(df)
        
        # Load SPY data
        spy_df = self.load_spy_data()
        
        # Find start of data with wave values
        start_date = data['date'].min()
        end_date = data['date'].max()
        
        print(f"\n🚀 STARTING WAVE RANGE BACKTEST")
        print("="*70)
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"🌊 Strategy: Buy {self.num_positions} highest rolling range assets, exit at +{self.profit_target}%")
        print(f"📊 Range threshold: ≥{self.min_range_threshold}% (6-month)")
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
            print("⚠️ Warning: Could not find SPY start price")
        
        # Initialize tracking
        current_capital = self.initial_capital
        active_positions = []
        portfolio_number = 0
        completed_portfolios = []
        daily_tracking = []
        
        # Start with first portfolio
        initial_opportunities = self.find_highest_wave_assets(data, start_date)
        
        if len(initial_opportunities) == 0:
            print("❌ No initial wave opportunities found")
            return None
        
        # Enter first portfolio
        active_positions = self.create_positions(initial_opportunities, self.initial_capital)
        portfolio_number = 1
        portfolio_entry_date = start_date
        total_invested = sum(pos['investment'] for pos in active_positions)
        current_capital -= total_invested
        
        print(f"\n🔥 Portfolio #1 - {start_date.strftime('%Y-%m-%d')}")
        print(f"    💰 Invested: ${total_invested:,.2f}")
        print(f"    🎯 Target Exit: ${total_invested * (1 + self.profit_target/100):,.2f} (+{self.profit_target}%)")
        print(f"    🌊 Assets: {[pos['asset'] for pos in active_positions]}")
        print(f"    📊 Wave Ranges: {[f'{pos['rolling_range_pct']:.1f}%' for pos in active_positions]}")
        print(f"    📍 Positions in Range: {[f'{pos['position_pct']:.1f}%' for pos in active_positions]}")
        
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
                        'exit_reason': f"PROFIT_TARGET (+{portfolio_return:.1f}%)",
                        'assets': [pos['asset'] for pos in active_positions],
                        'wave_ranges': [pos['rolling_range_pct'] for pos in active_positions],
                        'wave_positions': [pos['position_pct'] for pos in active_positions],
                        'spy_entry_price': spy_entry_price,
                        'spy_exit_price': spy_current_price,
                        'spy_period_return': spy_period_return,
                        'excess_return': portfolio_return - spy_period_return,
                        'outperformed_spy': portfolio_return > spy_period_return
                    }
                    
                    completed_portfolios.append(completed_portfolio)
                    
                    print(f"\n📤 Portfolio #{portfolio_number} EXIT - {current_date.strftime('%Y-%m-%d')}")
                    print(f"    📊 Wave Strategy Return: {portfolio_return:+.1f}% in {hold_days} days")
                    print(f"    📈 SPY Return: {spy_period_return:+.1f}% in {hold_days} days")
                    print(f"    🎯 Excess Return: {portfolio_return - spy_period_return:+.1f}%")
                    print(f"    💰 Value: ${portfolio_value:,.2f} -> Total: ${current_capital:,.2f}")
                    
                    # IMMEDIATELY ENTER NEW PORTFOLIO
                    new_opportunities = self.find_highest_wave_assets(data, current_date)
                    
                    if len(new_opportunities) > 0 and current_capital > 1000:
                        # Enter new portfolio
                        active_positions = self.create_positions(new_opportunities, current_capital)
                        portfolio_number += 1
                        portfolio_entry_date = current_date
                        total_new_invested = sum(pos['investment'] for pos in active_positions)
                        current_capital -= total_new_invested
                        
                        print(f"\n🔥 Portfolio #{portfolio_number} ENTRY - {current_date.strftime('%Y-%m-%d')}")
                        print(f"    💰 Invested: ${total_new_invested:,.2f}")
                        print(f"    🌊 Assets: {[pos['asset'] for pos in active_positions]}")
                        print(f"    📊 Wave Ranges: {[f'{pos['rolling_range_pct']:.1f}%' for pos in active_positions]}")
                        
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
                'completed_portfolios': len(completed_portfolios),
                'portfolio_number': portfolio_number if len(active_positions) > 0 else 0
            })
            
            # Progress updates (monthly)
            if current_date.day == 1:
                total_return = ((total_account_value / self.initial_capital) - 1) * 100
                excess_monthly = total_return - spy_return_pct
                print(f"📅 {current_date.strftime('%Y-%m')}: Wave: ${total_account_value:,.0f} ({total_return:+.1f}%) | "
                      f"SPY: ${spy_current_value:,.0f} ({spy_return_pct:+.1f}%) | "
                      f"Excess: {excess_monthly:+.1f}% | Completed: {len(completed_portfolios)}")
        
        # Final calculations
        final_value = current_capital
        if len(active_positions) > 0:
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
        """Analyze wave range backtest results with SPY comparison"""
        if results is None:
            return
        
        print(f"\n📊 WAVE RANGE BACKTEST RESULTS")
        print("="*70)
        
        # Overall performance comparison
        print(f"\n💰 OVERALL PERFORMANCE COMPARISON:")
        print(f"   Wave Strategy Initial: ${self.initial_capital:,.2f}")
        print(f"   Wave Strategy Final:   ${results['final_value']:,.2f}")
        print(f"   Wave Strategy Return:  {results['total_return']:+.2f}%")
        print(f"   ")
        print(f"   SPY Initial:           ${self.initial_capital:,.2f}")
        print(f"   SPY Final:             ${results['final_spy_value']:,.2f}")
        print(f"   SPY Return:            {results['spy_total_return']:+.2f}%")
        print(f"   ")
        excess_return = results['total_return'] - results['spy_total_return']
        print(f"   Excess Return:         {excess_return:+.2f}%")
        
        # Annualized returns
        if len(results['daily_tracking']) > 0:
            days = len(results['daily_tracking'])
            years = days / 365.25
            if years > 0:
                wave_annualized = ((results['final_value'] / self.initial_capital) ** (1/years) - 1) * 100
                spy_annualized = ((results['final_spy_value'] / self.initial_capital) ** (1/years) - 1) * 100
                print(f"   Period: {years:.2f} years ({days} days)")
                print(f"   Wave Annualized:       {wave_annualized:+.2f}%")
                print(f"   SPY Annualized:        {spy_annualized:+.2f}%")
                print(f"   Excess Annualized:     {wave_annualized - spy_annualized:+.2f}%")
        
        # Portfolio statistics
        portfolios = results['completed_portfolios']
        if len(portfolios) > 0:
            returns = [p['return_pct'] for p in portfolios]
            spy_returns = [p['spy_period_return'] for p in portfolios]
            excess_returns = [p['excess_return'] for p in portfolios]
            hold_times = [p['hold_days'] for p in portfolios]
            wave_ranges = []
            for p in portfolios:
                if p['wave_ranges']:
                    wave_ranges.extend(p['wave_ranges'])
            
            print(f"\n📈 WAVE PORTFOLIO STATISTICS:")
            print(f"   Completed Portfolios: {len(portfolios)}")
            print(f"   All hit {self.profit_target}% target")
            print(f"   ")
            print(f"   Wave Strategy Returns:")
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
            
            # Wave-specific statistics
            if wave_ranges:
                print(f"   ")
                print(f"   Wave Range Statistics:")
                print(f"     Average entry range: {np.mean(wave_ranges):.1f}%")
                print(f"     Median entry range:  {np.median(wave_ranges):.1f}%")
                print(f"     Min entry range:     {min(wave_ranges):.1f}%")
                print(f"     Max entry range:     {max(wave_ranges):.1f}%")
            
            # Win rate vs SPY
            outperformed_spy = sum(1 for p in portfolios if p.get('outperformed_spy', False))
            win_rate_vs_spy = (outperformed_spy / len(portfolios)) * 100
            print(f"   ")
            print(f"   Performance vs SPY:")
            print(f"     Outperformed: {outperformed_spy}/{len(portfolios)} ({win_rate_vs_spy:.1f}%)")
            
            # Show first few portfolios
            print(f"\n🎯 FIRST 5 WAVE PORTFOLIOS:")
            for i, p in enumerate(portfolios[:5]):
                assets_str = ', '.join(p['assets'])
                ranges_str = ', '.join([f"{r:.1f}%" for r in p['wave_ranges']])
                print(f"   #{i+1}: {p['entry_date'].strftime('%Y-%m-%d')} -> {p['exit_date'].strftime('%Y-%m-%d')} "
                      f"({p['hold_days']} days) | {p['return_pct']:.1f}% | {assets_str}")
                print(f"        Wave Ranges: {ranges_str} | SPY: {p['spy_period_return']:+.1f}% | "
                      f"Excess: {p['excess_return']:+.1f}%")
        
        return results
    
    def save_results(self, results):
        """Save wave backtest results to CSV"""
        if results is None:
            return
        
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        
        # Save daily tracking
        if len(results['daily_tracking']) > 0:
            daily_df = pd.DataFrame(results['daily_tracking'])
            daily_file = os.path.join(results_dir, "WAVE_RANGE_BACKTEST_DAILY.csv")
            daily_df.to_csv(daily_file, index=False)
            print(f"✅ Daily tracking: {daily_file}")
        
        # Save completed portfolios
        if len(results['completed_portfolios']) > 0:
            portfolios_df = pd.DataFrame(results['completed_portfolios'])
            # Convert lists to strings for CSV
            portfolios_df['assets'] = portfolios_df['assets'].astype(str)
            portfolios_df['wave_ranges'] = portfolios_df['wave_ranges'].astype(str)
            portfolios_df['wave_positions'] = portfolios_df['wave_positions'].astype(str)
            
            portfolio_file = os.path.join(results_dir, "WAVE_RANGE_BACKTEST_PORTFOLIOS.csv")
            portfolios_df.to_csv(portfolio_file, index=False)
            print(f"✅ Portfolios: {portfolio_file}")


def run_wave_range_backtest():
    """Run the wave range backtest"""
    print("🌊 WAVE RANGE BACKTEST STRATEGY")
    print("="*50)
    print("Strategy: Buy assets with highest 6-month rolling ranges")
    print("Theory: Biggest waves = highest volatility = biggest profit potential")
    print()
    
    backtester = WaveRangeBacktester(
        initial_capital=10000,
        num_positions=5,          # Number of highest wave assets to buy
        profit_target=20.0,       # 50% profit target
        min_range_threshold=15.0  # Only consider assets with ≥15% rolling range
    )
    
    # Run backtest
    results = backtester.backtest_strategy()
    
    if results is None:
        print("❌ Backtest failed - no results to analyze")
        return None
    
    # Analyze and save
    backtester.analyze_results(results)
    backtester.save_results(results)