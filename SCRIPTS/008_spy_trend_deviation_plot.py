import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_spy_z_scores():
    """
    Plot SPY Z-scores over time to visualize market regimes
    """
    print("PLOTTING SPY Z-SCORES...")
    
    # File paths
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    input_file = os.path.join(results_dir, "SPY_DAILY_MARKET_REGIME_DATA.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found at {input_file}")
        print("   Please run the SPY preparation script first.")
        return
    
    # Load data
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Loaded {len(df):,} daily records")
    print(f"📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Plot Z-score
    plt.plot(df['date'], df['market_z_score_60d'], linewidth=0.8, color='blue', alpha=0.7)
    
    # Add horizontal lines for reference
    plt.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Z = +2 (Spike)')
    plt.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Z = +1')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Z = 0 (Normal)')
    plt.axhline(y=-1, color='orange', linestyle='--', alpha=0.5, label='Z = -1')
    plt.axhline(y=-2, color='red', linestyle='--', alpha=0.7, label='Z = -2 (Crash)')
    
    # Fill areas for extreme zones
    plt.fill_between(df['date'], 2, 5, alpha=0.1, color='red', label='Spike Zone')
    plt.fill_between(df['date'], -2, -5, alpha=0.1, color='red', label='Crash Zone')
    
    # Formatting
    plt.title('SPY Market Regime Z-Scores Over Time\n(60-day rolling window)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Z-Score (Standard Deviations from Mean)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Set y-axis limits to show extreme values clearly
    plt.ylim(-5, 5)
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(results_dir, "SPY_Z_SCORE_PLOT.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"💾 Plot saved: {output_file}")
    print(f"📊 Z-score range: {df['market_z_score_60d'].min():.2f} to {df['market_z_score_60d'].max():.2f}")
    
    plt.show()

if __name__ == "__main__":
    plot_spy_z_scores()