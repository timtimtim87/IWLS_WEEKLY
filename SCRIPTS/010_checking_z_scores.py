import pandas as pd
import numpy as np
import os

def check_zscore_distribution():
    """
    Check the Z-score distribution to see if it's normal
    """
    print("Z-SCORE DISTRIBUTION DIAGNOSTIC")
    print("=" * 40)
    
    results_dir = "/Users/tim/CODE_PROJECTS/IWLS_WEEKLY/RESULTS"
    spy_file = os.path.join(results_dir, "SPY_DAILY_MARKET_REGIME_DATA.csv")
    
    if not os.path.exists(spy_file):
        print(f"❌ File not found: {spy_file}")
        return
    
    # Load SPY data
    spy_df = pd.read_csv(spy_file)
    z_scores = spy_df['market_z_score_60d'].dropna()
    
    print(f"📊 Total Z-score records: {len(z_scores):,}")
    print(f"📅 Date range: {spy_df['date'].iloc[0]} to {spy_df['date'].iloc[-1]}")
    
    # Basic statistics
    print(f"\n📈 Z-SCORE STATISTICS:")
    print(f"   Mean: {z_scores.mean():>8.3f}")
    print(f"   Std:  {z_scores.std():>8.3f}")
    print(f"   Min:  {z_scores.min():>8.3f}")
    print(f"   Max:  {z_scores.max():>8.3f}")
    
    # Distribution analysis
    print(f"\n📊 DISTRIBUTION ANALYSIS:")
    
    # Within standard deviation bands
    within_1 = ((z_scores >= -1) & (z_scores <= 1)).sum()
    within_2 = ((z_scores >= -2) & (z_scores <= 2)).sum()
    within_3 = ((z_scores >= -3) & (z_scores <= 3)).sum()
    
    pct_1 = within_1 / len(z_scores) * 100
    pct_2 = within_2 / len(z_scores) * 100
    pct_3 = within_3 / len(z_scores) * 100
    
    print(f"   Within ±1: {within_1:>6,} ({pct_1:>5.1f}%) [Normal: ~68%]")
    print(f"   Within ±2: {within_2:>6,} ({pct_2:>5.1f}%) [Normal: ~95%]")
    print(f"   Within ±3: {within_3:>6,} ({pct_3:>5.1f}%) [Normal: ~99.7%]")
    
    # Bin breakdown
    print(f"\n📋 DETAILED BIN BREAKDOWN:")
    
    bins = [
        ("Z < -3", z_scores < -3),
        ("Z -3 to -2", (z_scores >= -3) & (z_scores < -2)),
        ("Z -2 to -1", (z_scores >= -2) & (z_scores < -1)),
        ("Z -1 to 0", (z_scores >= -1) & (z_scores < 0)),
        ("Z 0 to +1", (z_scores >= 0) & (z_scores < 1)),
        ("Z +1 to +2", (z_scores >= 1) & (z_scores < 2)),
        ("Z +2 to +3", (z_scores >= 2) & (z_scores < 3)),
        ("Z > +3", z_scores > 3)
    ]
    
    for bin_name, condition in bins:
        count = condition.sum()
        pct = count / len(z_scores) * 100
        print(f"   {bin_name:>12}: {count:>6,} ({pct:>5.1f}%)")
    
    # Check if distribution looks normal
    print(f"\n🔍 NORMALITY CHECK:")
    if pct_1 < 50:
        print("   ⚠️  Distribution is NOT normal - too few values within ±1")
        print("   💡 This suggests the Z-score calculation needs adjustment")
    elif pct_1 > 80:
        print("   ⚠️  Distribution is too concentrated - might be over-normalized")
    else:
        print("   ✅ Distribution looks reasonably normal")
    
    # Sample some extreme values for inspection
    print(f"\n🔎 EXTREME VALUES SAMPLE:")
    
    # Highest Z-scores
    highest = z_scores.nlargest(5)
    print(f"   Highest Z-scores: {list(highest.round(2))}")
    
    # Lowest Z-scores  
    lowest = z_scores.nsmallest(5)
    print(f"   Lowest Z-scores:  {list(lowest.round(2))}")
    
    # Look at the underlying price-to-MA ratio
    if 'price_ma_pct' in spy_df.columns:
        price_ma_pct = spy_df['price_ma_pct'].dropna()
        print(f"\n📊 UNDERLYING PRICE-TO-MA PERCENTAGE:")
        print(f"   Mean: {price_ma_pct.mean():>6.2f}%")
        print(f"   Std:  {price_ma_pct.std():>6.2f}%")
        print(f"   Min:  {price_ma_pct.min():>6.2f}%")
        print(f"   Max:  {price_ma_pct.max():>6.2f}%")

if __name__ == "__main__":
    check_zscore_distribution()