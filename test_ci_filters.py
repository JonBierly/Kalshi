"""
Test CI-based trading filters (Filter A + Filter C)
"""
from src.trading.strategy import TradingStrategy, RiskManager

# Create strategy
risk_mgr = RiskManager(
    max_ci_width=0.25,  # 25% max
    min_edge=0.03
)
strategy = TradingStrategy(fractional_kelly=1.0, risk_manager=risk_mgr)

print("=" * 80)
print("Testing CI-Based Trading Filters")
print("=" * 80)

# Test Case 1: YES bet with confident edge
print("\n📊 Test 1: YES bet with 90% confident edge")
print("-" * 80)
model_result_yes = {
    'probability': 0.65,
    'ci_95_lower': 0.55,
    'ci_95_upper': 0.75,
    'ci_90_lower': 0.58,  # > 0.52 (market) ✓
    'ci_90_upper': 0.72,
    'seconds_remaining': 720,
    'score_diff': 3,
    'required_catchup_rate': 0.004
}

signal = strategy.evaluate_market(
    game_id='test1',
    model_result=model_result_yes,
    market_price_yes=52,  # 52¢
    market_price_no=48,   # 48¢
    bankroll=1000,
    current_position=None
)

print(f"Action: {signal.action}")
print(f"Side: {signal.side}")
print(f"Contracts: {signal.contracts}")
print(f"Reason: {signal.reason}")
print(f"Expected: TRADE (ci_90_lower 0.58 > market 0.52)")

# Test Case 2: NO bet with confident edge
print("\n📊 Test 2: NO bet with 90% confident edge")
print("-" * 80)
model_result_no = {
    'probability': 0.38,  # 38% home win
    'ci_95_lower': 0.28,
    'ci_95_upper': 0.48,
    'ci_90_lower': 0.30,
    'ci_90_upper': 0.45,  # < (1 - 0.40) = 0.60 ✓
    'seconds_remaining': 720,
    'score_diff': -3,
    'required_catchup_rate': 0.004
}

signal = strategy.evaluate_market(
    game_id='test2',
    model_result=model_result_no,
    market_price_yes=60,  # 60¢
    market_price_no=40,   # 40¢ (so NO wins if home prob < 60%)
    bankroll=1000,
    current_position=None
)

print(f"Action: {signal.action}")
print(f"Side: {signal.side}")
print(f"Contracts: {signal.contracts}")
print(f"Reason: {signal.reason}")
print(f"Expected: TRADE NO (ci_90_upper 0.45 < 1 - 0.40 = 0.60)")

# Test Case 3: Edge exists but NOT 90% confident (should skip)
print("\n📊 Test 3: Edge exists but NOT 90% confident")
print("-" * 80)
model_result_overlap = {
    'probability': 0.58,  # Slight edge
    'ci_95_lower': 0.45,
    'ci_95_upper': 0.71,
    'ci_90_lower': 0.48,  # < 0.55 (market) ✗ Not confident
    'ci_90_upper': 0.68,
    'seconds_remaining': 720,
    'score_diff': 2,
    'required_catchup_rate': 0.003
}

signal = strategy.evaluate_market(
    game_id='test3',
    model_result=model_result_overlap,
    market_price_yes=55,
    market_price_no=45,
    bankroll=1000,
    current_position=None
)

print(f"Action: {signal.action}")
print(f"Side: {signal.side}")
print(f"Contracts: {signal.contracts}")
print(f"Reason: {signal.reason}")
print(f"Expected: HOLD (ci_90_lower 0.48 < market 0.55, not confident)")

# Test Case 4: Kelly penalty with wide CI
print("\n📊 Test 4: Kelly penalty with wide CI vs narrow CI")
print("-" * 80)

# Wide CI (20% width)
model_wide = {
    'probability': 0.70,
    'ci_95_lower': 0.55,
    'ci_95_upper': 0.75,  # 20% width
    'ci_90_lower': 0.60,  # > 0.50 ✓
    'ci_90_upper': 0.73,
    'seconds_remaining': 720,
    'score_diff': 5,
    'required_catchup_rate': 0.007
}

signal_wide = strategy.evaluate_market(
    game_id='test4a',
    model_result=model_wide,
    market_price_yes=50,
    market_price_no=50,
    bankroll=1000,
    current_position=None
)

# Narrow CI (10% width)
model_narrow = {
    'probability': 0.70,
    'ci_95_lower': 0.63,
    'ci_95_upper': 0.73,  # 10% width
    'ci_90_lower': 0.65,  # > 0.50 ✓
    'ci_90_upper': 0.72,
    'seconds_remaining': 720,
   'score_diff': 5,
    'required_catchup_rate': 0.007
}

signal_narrow = strategy.evaluate_market(
    game_id='test4b',
    model_result=model_narrow,
    market_price_yes=50,
    market_price_no=50,
    bankroll=1000,
    current_position=None
)

print(f"Wide CI (20% width):")
print(f"  Contracts: {signal_wide.contracts}")
print(f"  Uncertainty penalty: {(0.25 - 0.20) / 0.25 * 100:.0f}% of Kelly")
print()
print(f"Narrow CI (10% width):")
print(f"  Contracts: {signal_narrow.contracts}")
print(f"  Uncertainty penalty: {(0.25 - 0.10) / 0.25 * 100:.0f}% of Kelly")
print()
print(f"Expected: Narrow CI → More contracts ({signal_narrow.contracts} > {signal_wide.contracts})")

print("\n" + "=" * 80)
print("✓ All tests complete!")
print("=" * 80)
