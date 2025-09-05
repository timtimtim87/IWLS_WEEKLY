import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TrendFilteredUnderperformanceBacktester:
    """
    Enhanced backtest strategy with trend filter:
    1. Find stocks with severe IWLS underperformance (negative deviation)
    2. Filter for those with RISING 20-day SMA (price trending up)
    3. Enter 5 most underperforming stocks that meet trend criteria
    4. Exit when portfolio hits +50% total return OR 12 months elapsed
    5. Track portfolio value daily
    """
    
    def __init__(self, initial_capital=10000, num_positions=5, profit_target=50.0, max_hold_months=12):
        self.initial_capital = initial_capital
        self.num_positions = num_positions
        self.profit_target = profit_target
        self.max_hold_months = max_hold_months
        
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
    
    def calculate_sma_trends(self, df):
        """Calculate 20-day SMA and trend direction for each asset"""
        print("📈 Calculating 20-day SMA trends for each asset...")
        
        # Sort data by asset and date
        df_sorted = df.sort_values(['asset', 'date']).copy()
        
        # Calculate 20-day SMA for each asset
        df_sorted['sma_20'] = df_sorted.groupby('asset')['price'].transform(
            lambda x: x.rolling(window=20, min_periods=10).mean()
        )
        
        # Calculate SMA trend (is current SMA > SMA from 5 days ago?)
        df_sorted['sma_20_lag5'] = df_sorted.groupby('asset')['sma_20'].shift(5)
        df_sorted['sma_trend_up'] = df_sorted['sma_20'] > df_sorted['sma_20_lag5']
        
        # Also calculate short-term trend (current SMA > yesterday's SMA)
        df_sorted['sma_20_lag1'] = df_sorted.groupby('asset')['sma_20'].shift(1)
        df_sorted['sma_trend_up_short'] = df_sorted['sma_20'] > df_sorted['sma_20_lag1']
        
        # Combine both trend conditions (both short and medium term uptrend)
        df_sorted['price_trending_up'] = (df_sorted['sma_trend_up'] & df_sorted['sma_trend_up_short']).fillna(False)
        
        print(f"✅ SMA trends calculated for all assets")
        
        # Show trend statistics
        trend_data = df_sorted.dropna(subset=['sma_20', 'sma_trend_up'])
        if len(trend_data) > 0:
            uptrend_pct = (trend_data['price_trending_up'].sum() / len(trend_data)) * 100
            print(f"📊 Price trending up: {trend_data['price_trending_up'].sum():,} / {len(trend_data):,} records ({uptrend_pct:.1f}%)")
        
        return df_sorted
    
    def prepare_data(self, df):
        """Prepare and clean data for backtesting"""
        print("🔧 Preparing trading data...")
        
        # Calculate SMA trends first
        df_with_trends = self.calculate_sma_trends(df)
        
        # Keep only records with all required data
        required_columns = ['price_deviation_from_trend', 'price', 'sma_20', 'price_trending_up']
        clean_data = df_with_trends.dropna(subset=required_columns).copy()
        
        # Sort chronologically
        clean_data = clean_data.sort_values(['date', 'asset']).reset_index(drop=True)
        
        print(f"📊 Clean records with trend data: {len(clean_data):,}")
        print(f"📅 Date range: {clean_data['date'].min().strftime('%Y-%m-%d')} to {clean_data['date'].max().strftime('%Y-%m-%d')}")
        print(f"🏢 Unique assets: {clean_data['asset'].nunique()}")
        
        return clean_data
    
    def find_trending_underperformers(self, df, current_date, lookback_days=7):
        """Find most underperforming stocks that are also trending up"""
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
        
        # Filter 1: Must have severe underperformance (< -30% deviation)
        underperformers = latest_by_asset[
            latest_by_asset['price_deviation_from_trend'] < -30
        ].copy()
        
        if len(underperformers) == 0:
            return []
        
        # Filter 2: Must have upward trending price (SMA rising)
        trending_underperformers = underperformers[
            underperformers['price_trending_up'] == True
        ].copy()
        
        if len(trending_underperformers) == 0:
            return []
        
        # Sort by worst deviation (most negative) and take top candidates
        trending_underperformers = trending_underperformers.sort_values('price_deviation_from_trend')
        
        # Return top opportunities
        opportunities = []
        for _, row in trending_underperformers.head(self.num_positions).iterrows():
            opportunities.append({
                'asset': row['asset'],
                'entry_price': row['price'],
                'deviation': row['price_deviation_from_trend'],
                'entry_date': row['date'],
                'sma_20': row['sma_20'],
                'price_above_sma': row['price'] > row['sma_20']
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
    
    def backtest_strategy(self, start_year=None, end_year=None):
        """Run the backtest with trend filtering"""
        
        # Load and prepare data
        df = self.load_data()
        if df is None:
            return None
        
        data = self.prepare_data(df)
        
        # Set date range
        if start_year is None:
            start_year = data['date'].min().year + 1  # Skip first year for SMA calculation
        if end_year is None:
            end_year = data['date'].max().year - 1  # Leave last year for forward tracking
        
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        # Filter data to backtest period
        backtest_data = data[
            (data['date'] >= start_date) & 
            (data['date'] <= end_date)
        ].copy()
        
        print(f"\n🚀 STARTING TREND-FILTERED BACKTEST")
        print("="*70)
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Strategy: Buy {self.num_positions} worst performers WITH upward price trend")
        print(f"📈 Trend filter: 20-day SMA must be rising")
        print(f"🚪 Exit: +{self.profit_target}% profit OR {self.max_hold_months} months")
        print(f"📅 Backtest period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get all unique trading dates
        all_dates = sorted(backtest_data['date'].unique())
        print(f"📊 Trading days: {len(all_dates):,}")
        
        # Initialize tracking
        current_capital = self.initial_capital
        active_positions = []
        portfolio_number = 0
        completed_portfolios = []
        daily_tracking = []
        
        # Track opportunities found vs. opportunities taken
        total_opportunities_found = 0
        total_opportunities_taken = 0
        opportunities_rejected_no_trend = 0
        
        # Main backtest loop
        for i, current_date in enumerate(all_dates):
            
            # If no active portfolio, look for entry opportunity (monthly)
            if len(active_positions) == 0:
                # Only look for entries on first trading day of each month
                if current_date.day <= 7 or (i > 0 and all_dates[i-1].month != current_date.month):
                    
                    # Find trending underperformers
                    opportunities = self.find_trending_underperformers(backtest_data, current_date)
                    
                    # Track statistics
                    if len(opportunities) > 0:
                        total_opportunities_found += len(opportunities)
                        
                        if current_capital > 1000:
                            # Enter new portfolio
                            investment_per_position = current_capital / len(opportunities)
                            
                            for opp in opportunities:
                                position = {
                                    'asset': opp['asset'],
                                    'entry_date': current_date,
                                    'entry_price': opp['entry_price'],
                                    'entry_deviation': opp['deviation'],
                                    'investment': investment_per_position,
                                    'shares': investment_per_position / opp['entry_price'],
                                    'sma_20_at_entry': opp['sma_20'],
                                    'price_above_sma_at_entry': opp['price_above_sma']
                                }
                                active_positions.append(position)
                            
                            portfolio_number += 1
                            portfolio_entry_date = current_date
                            total_invested = sum(pos['investment'] for pos in active_positions)
                            current_capital -= total_invested
                            total_opportunities_taken += len(opportunities)
                            
                            print(f"\n📥 Portfolio #{portfolio_number} - {current_date.strftime('%Y-%m-%d')}")
                            print(f"    💰 Invested: ${total_invested:,.2f}")
                            print(f"    🎯 Assets: {[pos['asset'] for pos in active_positions]}")
                            print(f"    📊 Deviations: {[f'{pos['entry_deviation']:.1f}%' for pos in active_positions]}")
                            print(f"    📈 SMA Status: {[f'${pos['sma_20_at_entry']:.2f}' for pos in active_positions]}")
            
            # If we have active positions, track performance and check for exit
            if len(active_positions) > 0:
                # Get current prices for active positions
                asset_list = [pos['asset'] for pos in active_positions]
                current_prices = self.get_current_prices(backtest_data, asset_list, current_date)
                
                # Calculate portfolio value
                portfolio_value = self.calculate_portfolio_value(active_positions, current_prices)
                
                # Calculate return and hold time
                total_invested = sum(pos['investment'] for pos in active_positions)
                portfolio_return = ((portfolio_value / total_invested) - 1) * 100
                hold_days = (current_date - portfolio_entry_date).days
                hold_months = hold_days / 30.44  # Average days per month
                
                # Check for exit condition
                should_exit = False
                exit_reason = ""
                
                if portfolio_return >= self.profit_target:
                    should_exit = True
                    exit_reason = f"{self.profit_target}% Target Hit"
                elif hold_months >= self.max_hold_months:
                    should_exit = True
                    exit_reason = f"{self.max_hold_months} Month Limit"
                
                if should_exit:
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
                        'entry_deviations': [pos['entry_deviation'] for pos in active_positions],
                        'trend_filtered': True
                    }
                    
                    completed_portfolios.append(completed_portfolio)
                    
                    print(f"\n📤 Portfolio #{portfolio_number} EXIT - {current_date.strftime('%Y-%m-%d')}")
                    print(f"    📊 Return: {portfolio_return:+.1f}% in {hold_days} days")
                    print(f"    💰 Value: ${portfolio_value:,.2f} -> Total: ${current_capital:,.2f}")
                    print(f"    🚪 Reason: {exit_reason}")
                    
                    # Clear active positions
                    active_positions = []
                    portfolio_entry_date = None
                    
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
            
            # Progress updates (quarterly)
            if current_date.month in [1, 4, 7, 10] and current_date.day <= 7:
                total_return = ((total_account_value / self.initial_capital) - 1) * 100
                print(f"📅 {current_date.strftime('%Y-%m')}: ${total_account_value:,.0f} ({total_return:+.1f}%) | "
                      f"Completed: {len(completed_portfolios)} | Active: {len(active_positions)}")
        
        # Final calculations
        final_value = current_capital
        if len(active_positions) > 0:
            # Calculate final portfolio value
            asset_list = [pos['asset'] for pos in active_positions]
            final_prices = self.get_current_prices(backtest_data, asset_list, all_dates[-1])
            final_portfolio_value = self.calculate_portfolio_value(active_positions, final_prices)
            final_value += final_portfolio_value
        
        # Calculate opportunity statistics
        opportunity_stats = {
            'total_opportunities_found': total_opportunities_found,
            'total_opportunities_taken': total_opportunities_taken,
            'opportunity_take_rate': (total_opportunities_taken / total_opportunities_found * 100) if total_opportunities_found > 0 else 0
        }
        
        return {
            'completed_portfolios': completed_portfolios,
            'daily_tracking': daily_tracking,
            'final_value': final_value,
            'total_return': ((final_value / self.initial_capital) - 1) * 100,
            'active_positions': active_positions,
            'opportunity_stats': opportunity_stats
        }
    
    def analyze_results(self, results):
        """Analyze backtest results"""
        if results is None:
            return
        
        print(f"\n📊 TREND-FILTERED BACKTEST RESULTS")
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
        
        # Opportunity statistics
        opp_stats = results['opportunity_stats']
        print(f"\n🎯 OPPORTUNITY STATISTICS:")
        print(f"   Total opportunities found: {opp_stats['total_opportunities_found']:,}")
        print(f"   Opportunities taken: {opp_stats['total_opportunities_taken']:,}")
        print(f"   Take rate: {opp_stats['opportunity_take_rate']:.1f}%")
        
        # Portfolio statistics
        portfolios = results['completed_portfolios']
        if len(portfolios) > 0:
            returns = [p['return_pct'] for p in portfolios]
            hold_times = [p['hold_days'] for p in portfolios]
            
            print(f"\n📈 PORTFOLIO STATS:")
            print(f"   Completed: {len(portfolios)}")
            print(f"   All used trend filter: {all(p.get('trend_filtered', False) for p in portfolios)}")
            print(f"   Avg Return: {np.mean(returns):.2f}%")
            print(f"   Median Return: {np.median(returns):.2f}%")
            print(f"   Best: {max(returns):.2f}%")
            print(f"   Worst: {min(returns):.2f}%")
            print(f"   Win Rate: {len([r for r in returns if r > 0])/len(returns)*100:.1f}%")
            print(f"   Avg Hold Time: {np.mean(hold_times):.0f} days")
            
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
            
            # Show first few portfolios for verification
            print(f"\n🎯 FIRST 5 PORTFOLIOS (Trend Filter Applied):")
            for i, p in enumerate(portfolios[:5]):
                assets_str = ', '.join(p['assets'])
                deviations_str = ', '.join([f"{d:.1f}%" for d in p['entry_deviations']])
                print(f"   #{i+1}: {p['entry_date'].strftime('%Y-%m-%d')} -> {p['exit_date'].strftime('%Y-%m-%d')} "
                      f"({p['hold_days']} days) | {p['return_pct']:.1f}% | {assets_str}")
                print(f"        Deviations: {deviations_str}")
        
        return results
    
    def save_results(self, results):
        """Save results to CSV"""
        if results is None:
            return
        
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        
        # Save daily tracking
        if len(results['daily_tracking']) > 0:
            daily_df = pd.DataFrame(results['daily_tracking'])
            daily_file = os.path.join(results_dir, "TREND_FILTERED_BACKTEST_DAILY.csv")
            daily_df.to_csv(daily_file, index=False)
            print(f"✅ Daily values: {daily_file}")
        
        # Save completed portfolios
        if len(results['completed_portfolios']) > 0:
            portfolios_df = pd.DataFrame(results['completed_portfolios'])
            portfolio_file = os.path.join(results_dir, "TREND_FILTERED_BACKTEST_PORTFOLIOS.csv")
            portfolios_df.to_csv(portfolio_file, index=False)
            print(f"✅ Portfolios: {portfolio_file}")


def run_trend_filtered_backtest():
    """Run the trend-filtered underperformance backtest"""
    print("IWLS UNDERPERFORMANCE BACKTEST WITH TREND FILTER")
    print("="*60)
    
    backtester = TrendFilteredUnderperformanceBacktester(
        initial_capital=10000,
        num_positions=5,
        profit_target=15.0,  # 50% profit target
        max_hold_months=200   # 12 month maximum hold
    )
    
    # Run backtest
    results = backtester.backtest_strategy()
    
    # Analyze and save
    backtester.analyze_results(results)
    backtester.save_results(results)
    
    print(f"\n✨ TREND-FILTERED BACKTEST COMPLETE!")
    print(f"🎯 Key Innovation: Only entered positions in underperforming stocks")
    print(f"   that showed upward price momentum (rising 20-day SMA)")
    print(f"📈 This should improve entry timing and reduce false signals")
    
    return results


if __name__ == "__main__":
    backtest_results = run_trend_filtered_backtest()