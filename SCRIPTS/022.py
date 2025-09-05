import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DynamicRebalancingBacktester:
    """
    Dynamic Rebalancing Strategy:
    1. Enter 3 most under-deviated stocks equally weighted
    2. When ANY stock hits 50% TP, close ALL positions  
    3. Immediately rebalance: keep the non-TP stocks + add most current under-deviated stock
    4. Equal weight all 3 positions with total capital
    5. Repeat process indefinitely
    """
    
    def __init__(self, initial_capital=10000, num_positions=3, profit_target=50.0):
        self.initial_capital = initial_capital
        self.num_positions = num_positions
        self.profit_target = profit_target
        
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
        
        # Keep only records with price and deviation data
        clean_data = df.dropna(subset=['price_deviation_from_trend', 'price']).copy()
        
        # Sort chronologically
        clean_data = clean_data.sort_values(['date', 'asset']).reset_index(drop=True)
        
        print(f"📊 Clean records: {len(clean_data):,}")
        print(f"📅 Date range: {clean_data['date'].min().strftime('%Y-%m-%d')} to {clean_data['date'].max().strftime('%Y-%m-%d')}")
        print(f"🏢 Unique assets: {clean_data['asset'].nunique()}")
        
        return clean_data
    
    def find_most_underperforming(self, df, current_date, exclude_assets=None, lookback_days=7):
        """Find the single most underperforming stock, optionally excluding certain assets"""
        start_date = current_date - timedelta(days=lookback_days)
        end_date = current_date + timedelta(days=lookback_days)
        
        # Get recent data around this date
        recent_data = df[
            (df['date'] >= start_date) & 
            (df['date'] <= end_date)
        ]
        
        if len(recent_data) == 0:
            return None
        
        # Exclude certain assets if specified
        if exclude_assets:
            recent_data = recent_data[~recent_data['asset'].isin(exclude_assets)]
        
        if len(recent_data) == 0:
            return None
        
        # Get the most recent record for each asset
        latest_by_asset = recent_data.loc[
            recent_data.groupby('asset')['date'].idxmax()
        ].reset_index(drop=True)
        
        # Filter for significant underperformance
        underperformers = latest_by_asset[
            latest_by_asset['price_deviation_from_trend'] < -20
        ].copy()
        
        if len(underperformers) == 0:
            return None
        
        # Sort by worst deviation and take the most underperforming
        worst_performer = underperformers.loc[underperformers['price_deviation_from_trend'].idxmin()]
        
        return {
            'asset': worst_performer['asset'],
            'entry_price': worst_performer['price'],
            'deviation': worst_performer['price_deviation_from_trend'],
            'entry_date': worst_performer['date']
        }
    
    def find_initial_underperformers(self, df, current_date, lookback_days=7):
        """Find initial 3 most underperforming stocks"""
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
        
        # Filter for underperformance
        underperformers = latest_by_asset[
            latest_by_asset['price_deviation_from_trend'] < -20
        ].copy()
        
        if len(underperformers) < self.num_positions:
            return []
        
        # Sort by worst deviation and take top 3
        underperformers = underperformers.sort_values('price_deviation_from_trend')
        
        opportunities = []
        for _, row in underperformers.head(self.num_positions).iterrows():
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
    
    def calculate_position_returns(self, positions, current_prices):
        """Calculate current return for each position"""
        position_returns = {}
        
        for position in positions:
            asset = position['asset']
            entry_price = position['entry_price']
            current_price = current_prices.get(asset)
            
            if current_price is not None:
                return_pct = ((current_price / entry_price) - 1) * 100
                position_returns[asset] = {
                    'return_pct': return_pct,
                    'current_price': current_price,
                    'entry_price': entry_price,
                    'position_value': position['shares'] * current_price
                }
            else:
                # If no current price, assume no change
                position_returns[asset] = {
                    'return_pct': 0.0,
                    'current_price': entry_price,
                    'entry_price': entry_price,
                    'position_value': position['investment']
                }
        
        return position_returns
    
    def backtest_strategy(self, start_year=None, end_year=None):
        """Run the dynamic rebalancing backtest"""
        
        # Load and prepare data
        df = self.load_data()
        if df is None:
            return None
        
        data = self.prepare_data(df)
        
        # Set date range
        if start_year is None:
            start_year = data['date'].min().year + 1
        if end_year is None:
            end_year = data['date'].max().year - 1
        
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        # Filter data to backtest period
        backtest_data = data[
            (data['date'] >= start_date) & 
            (data['date'] <= end_date)
        ].copy()
        
        print(f"\n🚀 STARTING DYNAMIC REBALANCING BACKTEST")
        print("="*70)
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Strategy: 3 most underperforming stocks, rebalance when ANY hits {self.profit_target}%")
        print(f"⚖️ Rebalance: Keep non-TP stocks + add new worst performer")
        print(f"📅 Backtest period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get all unique trading dates
        all_dates = sorted(backtest_data['date'].unique())
        print(f"📊 Trading days: {len(all_dates):,}")
        
        # Initialize tracking
        current_capital = self.initial_capital
        active_positions = []
        portfolio_entry_date = None
        rebalance_events = []
        daily_tracking = []
        
        rebalance_count = 0
        
        # Main backtest loop
        for i, current_date in enumerate(all_dates):
            
            # If no active portfolio, enter initial positions
            if len(active_positions) == 0:
                # Look for initial entry opportunity
                initial_opportunities = self.find_initial_underperformers(backtest_data, current_date)
                
                if len(initial_opportunities) >= self.num_positions and current_capital > 1000:
                    # Enter initial portfolio with equal weighting
                    investment_per_position = current_capital / self.num_positions
                    
                    for opp in initial_opportunities:
                        position = {
                            'asset': opp['asset'],
                            'entry_date': current_date,
                            'entry_price': opp['entry_price'],
                            'entry_deviation': opp['deviation'],
                            'investment': investment_per_position,
                            'shares': investment_per_position / opp['entry_price']
                        }
                        active_positions.append(position)
                    
                    portfolio_entry_date = current_date
                    total_invested = sum(pos['investment'] for pos in active_positions)
                    current_capital = 0  # All capital is now invested
                    rebalance_count += 1
                    
                    print(f"\n📥 Initial Portfolio #{rebalance_count} - {current_date.strftime('%Y-%m-%d')}")
                    print(f"    💰 Invested: ${total_invested:,.2f}")
                    print(f"    🎯 Assets: {[pos['asset'] for pos in active_positions]}")
                    print(f"    📊 Deviations: {[f'{pos['entry_deviation']:.1f}%' for pos in active_positions]}")
            
            # If we have active positions, check for rebalancing trigger
            if len(active_positions) > 0:
                # Get current prices for active positions
                asset_list = [pos['asset'] for pos in active_positions]
                current_prices = self.get_current_prices(backtest_data, asset_list, current_date)
                
                # Calculate position returns
                position_returns = self.calculate_position_returns(active_positions, current_prices)
                
                # Check if any position hit the profit target
                tp_triggered = False
                tp_asset = None
                
                for asset, returns in position_returns.items():
                    if returns['return_pct'] >= self.profit_target:
                        tp_triggered = True
                        tp_asset = asset
                        break
                
                if tp_triggered:
                    # REBALANCE EVENT: Close all positions and rebalance
                    
                    # Calculate total portfolio value
                    total_portfolio_value = sum(returns['position_value'] for returns in position_returns.values())
                    current_capital = total_portfolio_value
                    
                    # Find assets to keep (all except the one that hit TP)
                    assets_to_keep = [pos['asset'] for pos in active_positions if pos['asset'] != tp_asset]
                    
                    # Find new most underperforming asset (excluding current holdings)
                    new_opportunity = self.find_most_underperforming(
                        backtest_data, current_date, exclude_assets=assets_to_keep
                    )
                    
                    if new_opportunity is not None:
                        # Create new portfolio: kept assets + new opportunity
                        new_portfolio_assets = assets_to_keep + [new_opportunity['asset']]
                        
                        # Equal weight with all available capital
                        investment_per_position = current_capital / len(new_portfolio_assets)
                        
                        # Record rebalance event
                        rebalance_event = {
                            'rebalance_number': rebalance_count + 1,
                            'rebalance_date': current_date,
                            'trigger_asset': tp_asset,
                            'trigger_return': position_returns[tp_asset]['return_pct'],
                            'old_portfolio': [pos['asset'] for pos in active_positions],
                            'new_portfolio': new_portfolio_assets,
                            'portfolio_value': total_portfolio_value,
                            'days_since_last': (current_date - portfolio_entry_date).days,
                            'kept_assets': assets_to_keep,
                            'new_asset': new_opportunity['asset'],
                            'new_asset_deviation': new_opportunity['deviation']
                        }
                        
                        rebalance_events.append(rebalance_event)
                        
                        # Create new positions
                        new_positions = []
                        
                        # Add kept assets
                        for asset in assets_to_keep:
                            current_price = current_prices[asset]
                            if current_price is not None:
                                position = {
                                    'asset': asset,
                                    'entry_date': current_date,  # New entry date for rebalancing
                                    'entry_price': current_price,
                                    'entry_deviation': 0,  # Not applicable for kept assets
                                    'investment': investment_per_position,
                                    'shares': investment_per_position / current_price
                                }
                                new_positions.append(position)
                        
                        # Add new opportunity
                        new_price = self.get_current_prices(backtest_data, [new_opportunity['asset']], current_date)[new_opportunity['asset']]
                        if new_price is not None:
                            position = {
                                'asset': new_opportunity['asset'],
                                'entry_date': current_date,
                                'entry_price': new_price,
                                'entry_deviation': new_opportunity['deviation'],
                                'investment': investment_per_position,
                                'shares': investment_per_position / new_price
                            }
                            new_positions.append(position)
                        
                        # Update portfolio
                        active_positions = new_positions
                        portfolio_entry_date = current_date
                        current_capital = 0  # All capital invested again
                        rebalance_count += 1
                        
                        print(f"\n⚖️ REBALANCE #{rebalance_count} - {current_date.strftime('%Y-%m-%d')}")
                        print(f"    🎯 Trigger: {tp_asset} hit {position_returns[tp_asset]['return_pct']:.1f}%")
                        print(f"    💰 Portfolio Value: ${total_portfolio_value:,.2f}")
                        print(f"    🔄 Old: {rebalance_event['old_portfolio']}")
                        print(f"    🆕 New: {new_portfolio_assets}")
                        print(f"    📊 New Asset Deviation: {new_opportunity['deviation']:.1f}%")
                        print(f"    📅 Days since last: {rebalance_event['days_since_last']}")
                
                # Calculate current portfolio value for tracking
                total_portfolio_value = sum(returns['position_value'] for returns in position_returns.values())
            else:
                total_portfolio_value = 0
            
            # Daily tracking
            total_account_value = current_capital + total_portfolio_value
            daily_tracking.append({
                'date': current_date,
                'cash': current_capital,
                'portfolio_value': total_portfolio_value,
                'total_value': total_account_value,
                'total_return_pct': ((total_account_value / self.initial_capital) - 1) * 100,
                'active_positions': len(active_positions),
                'rebalance_count': rebalance_count
            })
            
            # Progress updates (yearly)
            if current_date.month == 1 and current_date.day <= 7:
                total_return = ((total_account_value / self.initial_capital) - 1) * 100
                print(f"📅 {current_date.year}: ${total_account_value:,.0f} ({total_return:+.1f}%) | "
                      f"Rebalances: {rebalance_count} | Active: {len(active_positions)}")
        
        # Final calculations
        if len(active_positions) > 0:
            asset_list = [pos['asset'] for pos in active_positions]
            final_prices = self.get_current_prices(backtest_data, asset_list, all_dates[-1])
            final_position_returns = self.calculate_position_returns(active_positions, final_prices)
            final_portfolio_value = sum(returns['position_value'] for returns in final_position_returns.values())
            final_value = current_capital + final_portfolio_value
        else:
            final_value = current_capital
        
        return {
            'rebalance_events': rebalance_events,
            'daily_tracking': daily_tracking,
            'final_value': final_value,
            'total_return': ((final_value / self.initial_capital) - 1) * 100,
            'active_positions': active_positions,
            'final_rebalance_count': rebalance_count
        }
    
    def analyze_results(self, results):
        """Analyze backtest results"""
        if results is None:
            return
        
        print(f"\n📊 DYNAMIC REBALANCING BACKTEST RESULTS")
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
        
        # Rebalancing statistics
        rebalances = results['rebalance_events']
        print(f"\n⚖️ REBALANCING STATS:")
        print(f"   Total rebalances: {results['final_rebalance_count']}")
        print(f"   Successful TPs: {len(rebalances)}")
        
        if len(rebalances) > 0:
            trigger_returns = [r['trigger_return'] for r in rebalances]
            days_between = [r['days_since_last'] for r in rebalances]
            
            print(f"   Avg TP return: {np.mean(trigger_returns):.1f}%")
            print(f"   Avg days between rebalances: {np.mean(days_between):.0f}")
            print(f"   Min days between: {min(days_between)}")
            print(f"   Max days between: {max(days_between)}")
            
            # Show rebalance frequency by year
            rebalance_dates = pd.to_datetime([r['rebalance_date'] for r in rebalances])
            rebalance_years = rebalance_dates.dt.year.value_counts().sort_index()
            
            print(f"\n📅 REBALANCES BY YEAR:")
            for year, count in rebalance_years.items():
                print(f"   {year}: {count} rebalances")
            
            # Show most frequent trigger assets
            trigger_assets = [r['trigger_asset'] for r in rebalances]
            trigger_counts = pd.Series(trigger_assets).value_counts()
            
            print(f"\n🎯 MOST FREQUENT TP TRIGGERS:")
            for asset, count in trigger_counts.head(5).items():
                avg_return = np.mean([r['trigger_return'] for r in rebalances if r['trigger_asset'] == asset])
                print(f"   {asset}: {count} times, {avg_return:.1f}% avg return")
            
            # Show sample rebalance events
            print(f"\n🔄 SAMPLE REBALANCE EVENTS:")
            for i, event in enumerate(rebalances[:5]):
                print(f"   #{event['rebalance_number']}: {event['rebalance_date'].strftime('%Y-%m-%d')}")
                print(f"      Trigger: {event['trigger_asset']} @ {event['trigger_return']:.1f}%")
                print(f"      Portfolio: {' → '.join([str(event['old_portfolio']), str(event['new_portfolio'])])}")
                print(f"      New asset: {event['new_asset']} ({event['new_asset_deviation']:.1f}% deviation)")
                print(f"      Value: ${event['portfolio_value']:,.2f}")
                print()
        
        return results
    
    def save_results(self, results):
        """Save results to CSV"""
        if results is None:
            return
        
        results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
        
        # Save daily tracking
        if len(results['daily_tracking']) > 0:
            daily_df = pd.DataFrame(results['daily_tracking'])
            daily_file = os.path.join(results_dir, "DYNAMIC_REBALANCING_DAILY.csv")
            daily_df.to_csv(daily_file, index=False)
            print(f"✅ Daily values: {daily_file}")
        
        # Save rebalance events
        if len(results['rebalance_events']) > 0:
            rebalance_df = pd.DataFrame(results['rebalance_events'])
            rebalance_file = os.path.join(results_dir, "DYNAMIC_REBALANCING_EVENTS.csv")
            rebalance_df.to_csv(rebalance_file, index=False)
            print(f"✅ Rebalance events: {rebalance_file}")


def run_dynamic_rebalancing_backtest():
    """Run the dynamic rebalancing backtest"""
    print("IWLS DYNAMIC REBALANCING BACKTEST")
    print("="*50)
    
    backtester = DynamicRebalancingBacktester(
        initial_capital=10000,
        num_positions=3,
        profit_target=50.0
    )
    
    # Run backtest
    results = backtester.backtest_strategy()
    
    # Analyze and save
    backtester.analyze_results(results)
    backtester.save_results(results)
    
    print(f"\n✨ DYNAMIC REBALANCING BACKTEST COMPLETE!")
    print(f"🎯 This strategy maintains continuous exposure to underperforming stocks")
    print(f"⚖️ While systematically harvesting profits at {backtester.profit_target}% targets")
    
    return results


if __name__ == "__main__":
    backtest_results = run_dynamic_rebalancing_backtest()