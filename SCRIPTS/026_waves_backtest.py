import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedWaveRangeBacktester:
    """
    Enhanced Wave Range backtest with trend analysis and range position:
    1. Find 5 highest rolling range assets (biggest waves)
    2. Calculate 21-day SMA trend direction for each asset
    3. Determine position within 6-month range (top/middle/bottom)
    4. Split $10,000 equally amongst them
    5. Track individual asset performance within portfolios
    6. Exit when portfolio hits profit target OR max hold days
    7. Analyze which asset types (trend/position) perform best
    8. Generate unique timestamped output files
    """
    
    def __init__(self, initial_capital=10000, num_positions=5, profit_target=50.0, max_hold_days=250):
        self.initial_capital = initial_capital
        self.num_positions = num_positions
        self.profit_target = profit_target
        self.max_hold_days = max_hold_days
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def calculate_trend_and_position_metrics(self, df):
        """Calculate 21-day SMA trend and range position for all assets"""
        print("🔍 Calculating trend and position metrics...")
        
        enhanced_data = []
        assets = df['asset'].unique()
        
        for i, asset in enumerate(assets, 1):
            if i % 100 == 0:
                print(f"   Processing asset {i}/{len(assets)}: {asset}")
            
            # Get asset data sorted by date
            asset_data = df[df['asset'] == asset].sort_values('date').copy()
            
            if len(asset_data) < 21:  # Need at least 21 days for SMA
                continue
            
            # Calculate 21-day SMA
            asset_data['sma_21'] = asset_data['price'].rolling(window=21, min_periods=21).mean()
            
            # Calculate SMA trend (1 = upward, 0 = sideways, -1 = downward)
            asset_data['sma_trend'] = 0  # Default sideways
            asset_data['sma_trend_direction'] = 'Sideways'
            
            # Calculate trend based on SMA change
            for j in range(1, len(asset_data)):
                current_sma = asset_data.iloc[j]['sma_21']
                prev_sma = asset_data.iloc[j-1]['sma_21']
                
                if pd.notna(current_sma) and pd.notna(prev_sma):
                    if current_sma > prev_sma:
                        asset_data.iloc[j, asset_data.columns.get_loc('sma_trend')] = 1
                        asset_data.iloc[j, asset_data.columns.get_loc('sma_trend_direction')] = 'Upward'
                    elif current_sma < prev_sma:
                        asset_data.iloc[j, asset_data.columns.get_loc('sma_trend')] = -1
                        asset_data.iloc[j, asset_data.columns.get_loc('sma_trend_direction')] = 'Downward'
            
            # Calculate range position (if wave analysis available)
            if 'rolling_range_pct_6_month' in asset_data.columns:
                # Get high/low from wave analysis if available
                if 'rolling_high_6_month' in asset_data.columns and 'rolling_low_6_month' in asset_data.columns:
                    asset_data['range_position_pct'] = (
                        (asset_data['price'] - asset_data['rolling_low_6_month']) / 
                        (asset_data['rolling_high_6_month'] - asset_data['rolling_low_6_month'])
                    ) * 100
                else:
                    # Calculate on the fly if not available
                    asset_data['rolling_high_6_month'] = asset_data['price'].rolling(window=126, min_periods=63).max()
                    asset_data['rolling_low_6_month'] = asset_data['price'].rolling(window=126, min_periods=63).min()
                    
                    asset_data['range_position_pct'] = (
                        (asset_data['price'] - asset_data['rolling_low_6_month']) / 
                        (asset_data['rolling_high_6_month'] - asset_data['rolling_low_6_month'])
                    ) * 100
                
                # Handle division by zero
                asset_data['range_position_pct'] = asset_data['range_position_pct'].fillna(50.0)
                
                # Categorize range position
                asset_data['range_position_category'] = 'Middle'
                asset_data.loc[asset_data['range_position_pct'] <= 33, 'range_position_category'] = 'Bottom'
                asset_data.loc[asset_data['range_position_pct'] >= 67, 'range_position_category'] = 'Top'
            else:
                asset_data['range_position_pct'] = 50.0
                asset_data['range_position_category'] = 'Middle'
            
            enhanced_data.append(asset_data)
        
        # Combine all enhanced data
        enhanced_df = pd.concat(enhanced_data, ignore_index=True)
        
        print(f"✅ Enhanced {len(enhanced_df):,} records with trend and position metrics")
        
        # Show distribution
        if len(enhanced_df) > 0:
            trend_dist = enhanced_df['sma_trend_direction'].value_counts()
            position_dist = enhanced_df['range_position_category'].value_counts()
            
            print(f"📊 Trend Distribution:")
            for trend, count in trend_dist.items():
                print(f"   {trend}: {count:,} ({count/len(enhanced_df)*100:.1f}%)")
            
            print(f"📊 Range Position Distribution:")
            for pos, count in position_dist.items():
                print(f"   {pos}: {count:,} ({count/len(enhanced_df)*100:.1f}%)")
        
        return enhanced_df
    
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
    
    def prepare_data(self, df):
        """Prepare and clean data for backtesting, starting from 1994"""
        print("🔧 Preparing enhanced trading data...")
        
        # Keep only records with wave analysis and prices
        clean_data = df.dropna(subset=['rolling_range_pct_6_month', 'price']).copy()
        
        # Filter to start from 1994 onwards
        start_date = pd.to_datetime('1994-01-01')
        clean_data = clean_data[clean_data['date'] >= start_date].copy()
        
        # Calculate trend and position metrics
        enhanced_data = self.calculate_trend_and_position_metrics(clean_data)
        
        # Sort chronologically
        enhanced_data = enhanced_data.sort_values(['date', 'asset']).reset_index(drop=True)
        
        print(f"📊 Enhanced records (1994+): {len(enhanced_data):,}")
        if len(enhanced_data) > 0:
            print(f"📅 Date range: {enhanced_data['date'].min().strftime('%Y-%m-%d')} to {enhanced_data['date'].max().strftime('%Y-%m-%d')}")
            print(f"🏢 Unique assets: {enhanced_data['asset'].nunique()}")
        
        return enhanced_data
    
    def find_highest_wave_performers(self, df, current_date, lookback_days=7):
        """Find the 5 highest rolling range assets with enhanced metrics"""
        start_date = current_date - timedelta(days=lookback_days)
        end_date = current_date + timedelta(days=lookback_days)
        
        recent_data = df[
            (df['date'] >= start_date) & 
            (df['date'] <= end_date)
        ]
        
        if len(recent_data) == 0:
            return []
        
        latest_by_asset = recent_data.loc[
            recent_data.groupby('asset')['date'].idxmax()
        ].reset_index(drop=True)
        
        highest_waves = latest_by_asset.sort_values('rolling_range_pct_6_month', ascending=False)
        
        opportunities = []
        for _, row in highest_waves.head(self.num_positions).iterrows():
            opportunities.append({
                'asset': row['asset'],
                'entry_price': row['price'],
                'rolling_range': row['rolling_range_pct_6_month'],
                'sma_trend': row.get('sma_trend', 0),
                'sma_trend_direction': row.get('sma_trend_direction', 'Unknown'),
                'range_position_pct': row.get('range_position_pct', 50.0),
                'range_position_category': row.get('range_position_category', 'Middle'),
                'sma_21': row.get('sma_21', row['price']),
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
    
    def calculate_individual_asset_performance(self, positions, current_prices, entry_date, exit_date):
        """Calculate performance for each individual asset in the portfolio"""
        individual_performance = []
        
        for position in positions:
            asset = position['asset']
            entry_price = position['entry_price']
            shares = position['shares']
            investment = position['investment']
            
            # Get exit price
            exit_price = current_prices.get(asset, entry_price)
            
            # Calculate individual return
            individual_return = ((exit_price / entry_price) - 1) * 100
            exit_value = shares * exit_price
            
            individual_performance.append({
                'asset': asset,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'investment': investment,
                'exit_value': exit_value,
                'individual_return_pct': individual_return,
                'sma_trend_direction': position.get('sma_trend_direction', 'Unknown'),
                'range_position_category': position.get('range_position_category', 'Middle'),
                'rolling_range': position.get('rolling_range', 0),
                'range_position_pct': position.get('range_position_pct', 50.0),
                'entry_sma_21': position.get('sma_21', entry_price),
                'hold_days': (exit_date - entry_date).days
            })
        
        return individual_performance
    
    def interpolate_spy_daily(self, spy_df, all_dates):
        """Interpolate SPY prices to daily data"""
        if spy_df is None:
            return None
        
        print("📈 Creating daily SPY data...")
        daily_df = pd.DataFrame({'date': all_dates})
        merged_df = pd.merge(daily_df, spy_df, on='date', how='left')
        merged_df['spy_price'] = merged_df['spy_price'].fillna(method='ffill').fillna(method='bfill')
        merged_df = merged_df.dropna(subset=['spy_price'])
        
        print(f"✅ SPY daily data created: {len(merged_df):,} trading days")
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
    
    def backtest_strategy(self):
        """Run enhanced backtest with trend and position analysis"""
        
        # Load and prepare data
        df = self.load_data()
        if df is None:
            return None
        
        data = self.prepare_data(df)
        spy_df = self.load_spy_data()
        
        start_date = data['date'].min()
        end_date = data['date'].max()
        
        print(f"\n🚀 STARTING ENHANCED WAVE RANGE BACKTEST")
        print("="*75)
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"🌊 Strategy: {self.num_positions} highest rolling range assets with trend analysis")
        print(f"🎯 Exit: +{self.profit_target}% profit OR {self.max_hold_days} days max hold")
        print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"🔍 Analysis: SMA trend direction + range position tracking")
        print(f"📁 Run ID: {self.run_timestamp}")
        
        all_dates = sorted(data['date'].unique())
        print(f"📊 Trading days: {len(all_dates):,}")
        
        # SPY setup
        spy_daily = self.interpolate_spy_daily(spy_df, all_dates)
        spy_start_price = self.get_spy_price_on_date(spy_daily, start_date)
        spy_shares = self.initial_capital / spy_start_price if spy_start_price else 0
        
        # Initialize tracking
        current_capital = self.initial_capital
        active_positions = []
        portfolio_number = 0
        completed_portfolios = []
        individual_trades = []  # Track every individual asset trade
        daily_tracking = []
        profit_exits = 0
        time_exits = 0
        
        # Start first portfolio
        initial_opportunities = self.find_highest_wave_performers(data, start_date)
        
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
                'rolling_range': opp['rolling_range'],
                'sma_trend_direction': opp['sma_trend_direction'],
                'range_position_category': opp['range_position_category'],
                'range_position_pct': opp['range_position_pct'],
                'sma_21': opp['sma_21'],
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
        print(f"    🌊 Assets: {[pos['asset'] for pos in active_positions]}")
        print(f"    📊 Ranges: {[f'{pos['rolling_range']:.1f}%' for pos in active_positions]}")
        print(f"    📈 Trends: {[pos['sma_trend_direction'] for pos in active_positions]}")
        print(f"    📍 Positions: {[pos['range_position_category'] for pos in active_positions]}")
        
        # Main backtest loop
        for i, current_date in enumerate(all_dates):
            
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
                    
                    # Calculate individual asset performance
                    individual_performance = self.calculate_individual_asset_performance(
                        active_positions, current_prices, portfolio_entry_date, current_date
                    )
                    
                    # Record individual trades
                    for trade in individual_performance:
                        trade['portfolio_number'] = portfolio_number
                        trade['portfolio_entry_date'] = portfolio_entry_date
                        trade['portfolio_exit_date'] = current_date
                        trade['portfolio_return_pct'] = portfolio_return
                        trade['portfolio_exit_reason'] = exit_reason
                        individual_trades.append(trade)
                    
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
                        'rolling_ranges': [pos['rolling_range'] for pos in active_positions],
                        'sma_trends': [pos['sma_trend_direction'] for pos in active_positions],
                        'range_positions': [pos['range_position_category'] for pos in active_positions],
                        'spy_entry_price': spy_entry_price,
                        'spy_exit_price': spy_current_price,
                        'spy_period_return': spy_period_return,
                        'excess_return': portfolio_return - spy_period_return,
                        'outperformed_spy': portfolio_return > spy_period_return
                    }
                    
                    completed_portfolios.append(completed_portfolio)
                    
                    print(f"\n📤 Portfolio #{portfolio_number} EXIT - {current_date.strftime('%Y-%m-%d')}")
                    print(f"    📊 {exit_reason}")
                    print(f"    📈 Wave: {portfolio_return:+.1f}% | SPY: {spy_period_return:+.1f}% | Excess: {portfolio_return - spy_period_return:+.1f}%")
                    print(f"    💰 ${portfolio_value:,.2f} -> Total: ${current_capital:,.2f}")
                    
                    # ENTER NEW PORTFOLIO
                    new_opportunities = self.find_highest_wave_performers(data, current_date)
                    
                    if len(new_opportunities) > 0 and current_capital > 1000:
                        investment_per_position = current_capital / len(new_opportunities)
                        
                        new_positions = []
                        for opp in new_opportunities:
                            position = {
                                'asset': opp['asset'],
                                'entry_date': current_date,
                                'entry_price': opp['entry_price'],
                                'rolling_range': opp['rolling_range'],
                                'sma_trend_direction': opp['sma_trend_direction'],
                                'range_position_category': opp['range_position_category'],
                                'range_position_pct': opp['range_position_pct'],
                                'sma_21': opp['sma_21'],
                                'investment': investment_per_position,
                                'shares': investment_per_position / opp['entry_price']
                            }
                            new_positions.append(position)
                        
                        active_positions = new_positions
                        portfolio_number += 1
                        portfolio_entry_date = current_date
                        total_new_invested = sum(pos['investment'] for pos in active_positions)
                        current_capital -= total_new_invested
                        
                        print(f"\n🔥 Portfolio #{portfolio_number} ENTRY - {current_date.strftime('%Y-%m-%d')}")
                        print(f"    💰 Invested: ${total_new_invested:,.2f}")
                        print(f"    🌊 Assets: {[pos['asset'] for pos in active_positions]}")
                        print(f"    📈 Trends: {[pos['sma_trend_direction'] for pos in active_positions]}")
                        print(f"    📍 Positions: {[pos['range_position_category'] for pos in active_positions]}")
                        
                        asset_list = [pos['asset'] for pos in active_positions]
                        current_prices = self.get_current_prices(data, asset_list, current_date)
                        portfolio_value = self.calculate_portfolio_value(active_positions, current_prices)
                    else:
                        active_positions = []
                        portfolio_value = 0
            else:
                portfolio_value = 0
            
            # Daily tracking
            spy_current_value = self.initial_capital
            spy_current_price = self.get_spy_price_on_date(spy_daily, current_date)
            
            if spy_start_price and spy_current_price:
                spy_current_value = spy_shares * spy_current_price
            
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
            
            # Monthly progress
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
        
        final_spy_value = self.initial_capital
        final_spy_price = self.get_spy_price_on_date(spy_daily, all_dates[-1])
        
        if spy_start_price and final_spy_price:
            final_spy_value = spy_shares * final_spy_price
        
        return {
            'completed_portfolios': completed_portfolios,
            'individual_trades': individual_trades,
            'daily_tracking': daily_tracking,
            'final_value': final_value,
            'total_return': ((final_value / self.initial_capital) - 1) * 100,
            'final_spy_value': final_spy_value,
            'spy_total_return': ((final_spy_value / self.initial_capital) - 1) * 100,
            'active_positions': active_positions,
            'spy_start_price': spy_start_price,
            'final_spy_price': final_spy_price,
            'profit_exits': profit_exits,
            'time_exits': time_exits,
            'run_timestamp': self.run_timestamp
        }
    
    def analyze_individual_trade_performance(self, individual_trades):
        """Analyze performance by trend direction and range position"""
        if not individual_trades:
            return
        
        print(f"\n🔍 INDIVIDUAL TRADE ANALYSIS")
        print("="*50)
        
        trades_df = pd.DataFrame(individual_trades)
        
        print(f"📊 Total individual trades: {len(trades_df):,}")
        
        # Analysis by SMA trend direction
        print(f"\n📈 PERFORMANCE BY SMA TREND DIRECTION:")
        trend_analysis = trades_df.groupby('sma_trend_direction').agg({
            'individual_return_pct': ['count', 'mean', 'median', 'std'],
            'hold_days': 'mean'
        }).round(2)
        
        print(f"{'Trend':>10} | {'Count':>8} | {'Avg Return':>11} | {'Median':>8} | {'Std Dev':>8} | {'Avg Days':>9}")
        print("-" * 70)
        
        for trend in trend_analysis.index:
            count = int(trend_analysis.loc[trend, ('individual_return_pct', 'count')])
            avg_return = trend_analysis.loc[trend, ('individual_return_pct', 'mean')]
            median_return = trend_analysis.loc[trend, ('individual_return_pct', 'median')]
            std_return = trend_analysis.loc[trend, ('individual_return_pct', 'std')]
            avg_days = trend_analysis.loc[trend, ('hold_days', 'mean')]
            
            print(f"{trend:>10} | {count:>8,} | {avg_return:>+10.1f}% | {median_return:>+7.1f}% | {std_return:>7.1f}% | {avg_days:>8.0f}")
        
        # Analysis by range position
        print(f"\n📍 PERFORMANCE BY RANGE POSITION:")
        position_analysis = trades_df.groupby('range_position_category').agg({
            'individual_return_pct': ['count', 'mean', 'median', 'std'],
            'hold_days': 'mean'
        }).round(2)
        
        print(f"{'Position':>10} | {'Count':>8} | {'Avg Return':>11} | {'Median':>8} | {'Std Dev':>8} | {'Avg Days':>9}")
        print("-" * 70)
        
        for position in position_analysis.index:
            count = int(position_analysis.loc[position, ('individual_return_pct', 'count')])
            avg_return = position_analysis.loc[position, ('individual_return_pct', 'mean')]
            median_return = position_analysis.loc[position, ('individual_return_pct', 'median')]
            std_return = position_analysis.loc[position, ('individual_return_pct', 'std')]
            avg_days = position_analysis.loc[position, ('hold_days', 'mean')]
            
            print(f"{position:>10} | {count:>8,} | {avg_return:>+10.1f}% | {median_return:>+7.1f}% | {std_return:>7.1f}% | {avg_days:>8.0f}")
        
        # Combined analysis (trend + position)
        print(f"\n🎯 COMBINED TREND + POSITION ANALYSIS:")
        combined_analysis = trades_df.groupby(['sma_trend_direction', 'range_position_category']).agg({
            'individual_return_pct': ['count', 'mean'],
            'hold_days': 'mean'
        }).round(2)
        
        print(f"{'Trend':>10} | {'Position':>10} | {'Count':>8} | {'Avg Return':>11} | {'Avg Days':>9}")
        print("-" * 65)
        
        for (trend, position) in combined_analysis.index:
            count = int(combined_analysis.loc[(trend, position), ('individual_return_pct', 'count')])
            avg_return = combined_analysis.loc[(trend, position), ('individual_return_pct', 'mean')]
            avg_days = combined_analysis.loc[(trend, position), ('hold_days', 'mean')]
            
            print(f"{trend:>10} | {position:>10} | {count:>8,} | {avg_return:>+10.1f}% | {avg_days:>8.0f}")
        
        # Best and worst performers
        print(f"\n🏆 BEST PERFORMING COMBINATIONS:")
        best_combos = trades_df.groupby(['sma_trend_direction', 'range_position_category'])['individual_return_pct'].mean().sort_values(ascending=False)
        
        for i, ((trend, position), avg_return) in enumerate(best_combos.head(5).items(), 1):
            count = len(trades_df[(trades_df['sma_trend_direction'] == trend) & 
                                (trades_df['range_position_category'] == position)])
            print(f"   {i}. {trend} + {position}: {avg_return:+.1f}% avg ({count:,} trades)")
        
        print(f"\n📉 WORST PERFORMING COMBINATIONS:")
        for i, ((trend, position), avg_return) in enumerate(best_combos.tail(5).items(), 1):
            count = len(trades_df[(trades_df['sma_trend_direction'] == trend) & 
                                (trades_df['range_position_category'] == position)])
            print(f"   {i}. {trend} + {position}: {avg_return:+.1f}% avg ({count:,} trades)")
        
        # Rolling range analysis
        print(f"\n🌊 PERFORMANCE BY ROLLING RANGE QUINTILES:")
        trades_df['range_quintile'] = pd.qcut(trades_df['rolling_range'], q=5, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'])
        
        range_quintile_analysis = trades_df.groupby('range_quintile').agg({
            'individual_return_pct': ['count', 'mean', 'median'],
            'rolling_range': 'mean'
        }).round(2)
        
        print(f"{'Quintile':>12} | {'Count':>8} | {'Avg Range':>10} | {'Avg Return':>11} | {'Median':>8}")
        print("-" * 65)
        
        for quintile in range_quintile_analysis.index:
            count = int(range_quintile_analysis.loc[quintile, ('individual_return_pct', 'count')])
            avg_range = range_quintile_analysis.loc[quintile, ('rolling_range', 'mean')]
            avg_return = range_quintile_analysis.loc[quintile, ('individual_return_pct', 'mean')]
            median_return = range_quintile_analysis.loc[quintile, ('individual_return_pct', 'median')]
            
            print(f"{quintile:>12} | {count:>8,} | {avg_range:>9.1f}% | {avg_return:>+10.1f}% | {median_return:>+7.1f}%")
    
    def analyze_results(self, results):
        """Enhanced analysis with trend and position insights"""
        if results is None:
            return
        
        print(f"\n📊 ENHANCED WAVE RANGE BACKTEST RESULTS")
        print("="*70)
        print(f"🕐 Run ID: {results['run_timestamp']}")
        
        # Overall performance
        print(f"\n💰 OVERALL PERFORMANCE:")
        print(f"   Wave Strategy Final:   ${results['final_value']:,.2f}")
        print(f"   Wave Strategy Return:  {results['total_return']:+.2f}%")
        print(f"   SPY Final:             ${results['final_spy_value']:,.2f}")
        print(f"   SPY Return:            {results['spy_total_return']:+.2f}%")
        excess_return = results['total_return'] - results['spy_total_return']
        print(f"   Excess Return:         {excess_return:+.2f}%")
        
        # Exit analysis
        profit_exits = results.get('profit_exits', 0)
        time_exits = results.get('time_exits', 0)
        total_exits = profit_exits + time_exits
        
        print(f"\n🎯 EXIT ANALYSIS:")
        print(f"   Total Portfolios: {total_exits}")
        print(f"   Profit Exits: {profit_exits} ({(profit_exits/total_exits*100):.1f}%)")
        print(f"   Time Exits: {time_exits} ({(time_exits/total_exits*100):.1f}%)")
        
        # Individual trade analysis
        if 'individual_trades' in results and results['individual_trades']:
            self.analyze_individual_trade_performance(results['individual_trades'])
        
        return results
    
    def save_results(self, results):
        """Save results with unique timestamps"""
        if results is None:
            return
        
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        timestamp = results['run_timestamp']
        
        # Save daily tracking
        if len(results['daily_tracking']) > 0:
            daily_df = pd.DataFrame(results['daily_tracking'])
            daily_file = os.path.join(results_dir, f"ENHANCED_WAVE_BACKTEST_DAILY_{timestamp}.csv")
            daily_df.to_csv(daily_file, index=False)
            print(f"✅ Daily tracking: {daily_file}")
        
        # Save portfolios
        if len(results['completed_portfolios']) > 0:
            portfolios_df = pd.DataFrame(results['completed_portfolios'])
            portfolios_df['assets'] = portfolios_df['assets'].astype(str)
            portfolios_df['rolling_ranges'] = portfolios_df['rolling_ranges'].astype(str)
            portfolios_df['sma_trends'] = portfolios_df['sma_trends'].astype(str)
            portfolios_df['range_positions'] = portfolios_df['range_positions'].astype(str)
            
            portfolio_file = os.path.join(results_dir, f"ENHANCED_WAVE_BACKTEST_PORTFOLIOS_{timestamp}.csv")
            portfolios_df.to_csv(portfolio_file, index=False)
            print(f"✅ Portfolios: {portfolio_file}")
        
        # Save individual trades (NEW - most important for analysis)
        if 'individual_trades' in results and len(results['individual_trades']) > 0:
            trades_df = pd.DataFrame(results['individual_trades'])
            trades_file = os.path.join(results_dir, f"ENHANCED_WAVE_BACKTEST_INDIVIDUAL_TRADES_{timestamp}.csv")
            trades_df.to_csv(trades_file, index=False)
            print(f"✅ Individual trades: {trades_file}")
        
        # Save summary
        summary_stats = {
            'metric': [
                'Run Timestamp',
                'Initial Capital',
                'Final Wave Value',
                'Final SPY Value',
                'Wave Total Return %',
                'SPY Total Return %',
                'Excess Return %',
                'Total Portfolios',
                'Profit Target Exits',
                'Max Hold Exits',
                'Total Individual Trades',
                'Avg Trade Return %',
                'Win Rate vs SPY %'
            ],
            'value': [
                timestamp,
                self.initial_capital,
                results['final_value'],
                results['final_spy_value'],
                results['total_return'],
                results['spy_total_return'],
                results['total_return'] - results['spy_total_return'],
                len(results['completed_portfolios']),
                results.get('profit_exits', 0),
                results.get('time_exits', 0),
                len(results.get('individual_trades', [])),
                np.mean([t['individual_return_pct'] for t in results.get('individual_trades', [])]) if results.get('individual_trades') else 0,
                (sum(1 for p in results['completed_portfolios'] if p.get('outperformed_spy', False)) / 
                 len(results['completed_portfolios']) * 100) if results['completed_portfolios'] else 0
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        summary_file = os.path.join(results_dir, f"ENHANCED_WAVE_BACKTEST_SUMMARY_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Summary: {summary_file}")


def run_enhanced_wave_backtest():
    """Run enhanced wave backtest with trend and position analysis"""
    print("🌊 ENHANCED WAVE RANGE BACKTEST WITH TREND ANALYSIS")
    print("="*65)
    print("Features:")
    print("• 21-day SMA trend direction (Upward/Downward/Sideways)")
    print("• Range position analysis (Top/Middle/Bottom of 6-month range)")
    print("• Individual asset performance tracking")
    print("• Unique timestamped outputs (no overwriting)")
    print("• Deep dive into what types of assets drive performance")
    print()
    
    backtester = EnhancedWaveRangeBacktester(
        initial_capital=10000,
        num_positions=9,
        profit_target=15.0,
        max_hold_days=200
    )
    
    results = backtester.backtest_strategy()
    
    if results is None:
        print("❌ Backtest failed")
        return None
    
    backtester.analyze_results(results)
    backtester.save_results(results)
    
    timestamp = results['run_timestamp']
    print(f"\n✨ ENHANCED BACKTEST COMPLETE!")
    print(f"🕐 Run ID: {timestamp}")
    print(f"📁 All files saved with unique timestamp")
    print(f"\n🔍 KEY INSIGHTS GENERATED:")
    print(f"   • Performance by SMA trend direction")
    print(f"   • Performance by range position (top/middle/bottom)")
    print(f"   • Combined trend + position analysis")
    print(f"   • Rolling range quintile performance")
    print(f"   • Individual trade-level data for deep analysis")
    
    return results


if __name__ == "__main__":
    enhanced_results = run_enhanced_wave_backtest()