import yfinance as yf
import pandas as pd
import json
from collections import defaultdict


def fetch_market_data(symbol="BTC-INR", period="max"):
    print(f"⏳ Downloading MAX history for {symbol}...")
    df = yf.download(symbol, period=period, interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    print(f"✅ Data Acquired: {len(df)} hours of trading history!")
    return df

def classify_market_event(row):
    if row['Open'] == 0:
        return "STABLE"
    change_pct = ((row['Close'] - row['Open']) / row['Open']) * 100
    if change_pct >= 1.5:
        return "PUMP_HUGE"
    elif change_pct >= 0.3:
        return "PUMP_SMALL"
    elif change_pct <= -1.5:
        return "DUMP_HUGE"
    elif change_pct <= -0.3:
        return "DUMP_SMALL"
    else:
        return "STABLE"

def mine_patterns(df):
    print("⚙️ Applying Smart Classification...")
    df['Event'] = df.apply(classify_market_event, axis=1)
    
    SEQUENCE_LENGTH = 3
    TARGET_PROFIT = 1.0      # 1% profit target (after fees ~0.6% net)
    STOP_LOSS = -0.6         # 0.6% stop loss
    MIN_OCCURRENCES = 20     # Statistical significance
    MIN_WIN_RATE = 0.55      # 55% minimum win rate
    HOLDING_PERIODS = [1, 2, 3]  # Check 1h, 2h, 3h resolutions
    
    pattern_tracker = defaultdict(lambda: {
        'wins': 0,
        'losses': 0,
        'total': 0,
        'profit_sum': 0.0,
        'loss_sum': 0.0,
        'stopped_out': 0,
        'target_hit': 0
    })
    
    print(f"⏳ Scanning patterns with realistic stop-loss execution...")
    
    for i in range(len(df) - SEQUENCE_LENGTH - max(HOLDING_PERIODS)):
        pattern = tuple(df['Event'].iloc[i : i + SEQUENCE_LENGTH])
        
        # Entry at next candle's open (realistic)
        entry_price = df['Open'].iloc[i + SEQUENCE_LENGTH]
        if entry_price <= 0:
            continue
        
        # Check multiple holding periods
        best_result = None
        
        for holding in HOLDING_PERIODS:
            trade_result = simulate_trade(
                df, i + SEQUENCE_LENGTH, holding, 
                entry_price, TARGET_PROFIT, STOP_LOSS
            )
            
            if best_result is None or trade_result['pnl'] > best_result['pnl']:
                best_result = trade_result
        
        # Record trade outcome
        stats = pattern_tracker[pattern]
        stats['total'] += 1
        
        if best_result['outcome'] == 'WIN':
            stats['wins'] += 1
            stats['target_hit'] += 1
            stats['profit_sum'] += best_result['pnl']
        else:
            stats['losses'] += 1
            if best_result['outcome'] == 'STOPPED':
                stats['stopped_out'] += 1
            stats['loss_sum'] += abs(best_result['pnl'])
    
    print(f"✅ Analyzed {len(pattern_tracker)} unique patterns.")
    
    # Calculate statistics and filter
    golden_patterns = []
    
    for pattern, stats in pattern_tracker.items():
        if stats['total'] < MIN_OCCURRENCES:
            continue
        
        win_rate = stats['wins'] / stats['total']
        
        if win_rate < MIN_WIN_RATE:
            continue
        
        # Calculate REAL expected value
        avg_win = stats['profit_sum'] / stats['wins'] if stats['wins'] > 0 else 0
        avg_loss = stats['loss_sum'] / stats['losses'] if stats['losses'] > 0 else 0
        
        expected_value = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        golden_patterns.append({
            'pattern': list(pattern),
            'win_rate': round(win_rate * 100, 2),
            'wins': stats['wins'],
            'losses': stats['losses'],
            'total_trades': stats['total'],
            'avg_win': round(avg_win, 3),
            'avg_loss': round(avg_loss, 3),
            'expected_value': round(expected_value, 3),
            'stopped_out_pct': round((stats['stopped_out'] / stats['total']) * 100, 2),
            'profit_factor': round(stats['profit_sum'] / stats['loss_sum'], 2) if stats['loss_sum'] > 0 else 0
        })
    
    # Sort by expected value
    golden_patterns.sort(key=lambda x: x['expected_value'], reverse=True)
    
    if not golden_patterns:
        print(f"❌ No patterns found with {MIN_WIN_RATE*100}% win rate and {MIN_OCCURRENCES}+ occurrences.")
        return
    
    
    final_patterns = [p['pattern'] for p in golden_patterns[:50]]
    
    # Save top 50
    filename = "golden_patterns.json"
    with open(filename, "w") as f:
        json.dump(final_patterns, f, indent=2)
    
    print(f"\n💎 SUCCESS! Saved {len(golden_patterns[:50])} validated patterns:")
    print(f"   📊 Average Win Rate: {sum(p['win_rate'] for p in golden_patterns[:50]) / len(golden_patterns[:50]):.2f}%")
    print(f"   💰 Average EV: {sum(p['expected_value'] for p in golden_patterns[:50]) / len(golden_patterns[:50]):.3f}%")
    print(f"   📈 Best Pattern: {golden_patterns[0]['pattern']}")
    print(f"      ├─ Win Rate: {golden_patterns[0]['win_rate']}%")
    print(f"      ├─ Expected Value: {golden_patterns[0]['expected_value']}%")
    print(f"      └─ Trades: {golden_patterns[0]['total_trades']}")

def simulate_trade(df, entry_idx, holding_periods, entry_price, target_pct, stop_pct):
    """
    Simulates a trade with realistic stop-loss execution.
    Returns outcome and P&L.
    """
    outcome = 'LOSS'  # Default
    pnl = 0.0
    
    # Check each candle in the holding period
    for h in range(holding_periods):
        if entry_idx + h >= len(df):
            break
        
        candle_high = df['High'].iloc[entry_idx + h]
        candle_low = df['Low'].iloc[entry_idx + h]
        candle_close = df['Close'].iloc[entry_idx + h]
        
        # Calculate intrabar movements
        high_pct = ((candle_high - entry_price) / entry_price) * 100
        low_pct = ((candle_low - entry_price) / entry_price) * 100
        
        # CRITICAL: Check stop-loss FIRST (touched before target)
        if low_pct <= stop_pct:
            outcome = 'STOPPED'
            pnl = stop_pct  # Negative value
            return {'outcome': outcome, 'pnl': pnl}
        
        # Then check if target hit
        if high_pct >= target_pct:
            outcome = 'WIN'
            pnl = target_pct
            return {'outcome': outcome, 'pnl': pnl}
    
    # If neither hit, exit at final close
    final_close = df['Close'].iloc[entry_idx + holding_periods - 1]
    pnl = ((final_close - entry_price) / entry_price) * 100
    outcome = 'WIN' if pnl > 0 else 'LOSS'
    
    return {'outcome': outcome, 'pnl': pnl}

if __name__ == "__main__":
    df = fetch_market_data()
    if not df.empty:
        mine_patterns(df)
    else:
        print("❌ Error: No data fetched.")


# import yfinance as yf
# import pandas as pd
# import json
# from collections import Counter
# import os

# def fetch_market_data(symbol="BTC-INR", period="max"):
    
#     # Downloads historical market data from Yahoo Finance.
    
#     print(f"⏳ Downloading MAX history for {symbol}...")
    
#     # Fetch unlimited history
#     df = yf.download(symbol, period=period, interval="1h", progress=False)
    
#     # Handle MultiIndex columns (fixes potential issues with newer yfinance versions)
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.get_level_values(0)
    
#     df = df.dropna()
#     print(f"✅ Data Acquired: {len(df)} hours of trading history!")
#     return df

# def classify_market_event(row):
    
#     # Classifies a trading hour based on percentage change.
    
#     if row['Open'] == 0: return "STABLE"

#     change_pct = ((row['Close'] - row['Open']) / row['Open']) * 100

#     if change_pct >= 1.5: return "PUMP_HUGE"
#     elif change_pct >= 0.3: return "PUMP_SMALL"
#     elif change_pct <= -1.5: return "DUMP_HUGE"
#     elif change_pct <= -0.3: return "DUMP_SMALL"
#     else: return "STABLE"

# def mine_patterns(df):

#     # Scans data for patterns and OVERWRITES the JSON file.

#     print("⚙️ Applying Smart Classification...")
#     df['Event'] = df.apply(classify_market_event, axis=1)

#     SEQUENCE_LENGTH = 3
#     TARGET_PROFIT = 0.5  # 0.5% profit target
#     sequences = []
    
#     print(f"⏳ Scanning for patterns leading to {TARGET_PROFIT}% profit...")

#     # Logic to find winning sequences
#     for i in range(len(df) - SEQUENCE_LENGTH - 1):
#         current_pattern = df['Event'].iloc[i : i + SEQUENCE_LENGTH].tolist()
#         next_open = df['Open'].iloc[i + SEQUENCE_LENGTH + 1]
#         next_close = df['Close'].iloc[i + SEQUENCE_LENGTH + 1]
        
#         future_profit = 0
#         if next_open > 0:
#             future_profit = ((next_close - next_open) / next_open) * 100

#         if future_profit >= TARGET_PROFIT:
#             current_pattern.append("PROFIT_HIT")
#             sequences.append(current_pattern)

#     print(f"✅ Found {len(sequences)} winning trades.")

#     if len(sequences) > 0:
#         # Convert to string to count duplicates
#         seq_strings = [json.dumps(seq) for seq in sequences]
#         counts = Counter(seq_strings)

#         golden_patterns = []
        
#         # Keep top 50 patterns that appear at least 5 times
#         for seq_str, count in counts.most_common(50):
#             if count >= 5:
#                 full_seq = json.loads(seq_str)
#                 golden_patterns.append(full_seq[:-1]) # Remove 'PROFIT_HIT'

#         # Save to JSON (Overwrite Mode)
#         filename = 'golden_patterns.json'
        
#         # 'w' mode truncates the file first, ensuring a clean overwrite
#         with open(filename, 'w') as f:
#             json.dump(golden_patterns, f)

#         print(f"\n💎 SUCCESS! Overwrote {filename} with {len(golden_patterns)} fresh patterns.")
#     else:
#         print("❌ No matches found. File not updated.")

# if __name__ == "__main__":
#     df = fetch_market_data()
#     if not df.empty:
#         mine_patterns(df)
#     else:
#         print("❌ Error: No data fetched.")
