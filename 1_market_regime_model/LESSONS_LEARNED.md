# Lessons Learned: Acceleration/Momentum Feature Experiment

## Experiment: Adding Transition-Capture Features

**Date**: April 2026
**Hypothesis**: Spread *acceleration* (how fast they're widening/narrowing) would better capture regime transitions than spread *levels* alone.

**Rationale**:
- A 350 bps HY spread is ambiguous without context
- But "spreads rising 10 bps/day" clearly signals deteriorating credit
- Model should detect regime SHIFTS (when transitions begin), not just current state

## Features Attempted

```python
# Acceleration (change of change)
hy_spread_accel_21d = hy_spread_chg_21d.diff(1)
ted_spread_accel_21d = (y10 - y3m).diff(1)
vix_accel_21d = vix_chg_21d.diff(1)

# Momentum (normalized directional strength)
hy_spread_momentum = hy_spread_chg_21d / (hy_chg_std + 1e-6)
vix_momentum = vix_chg_21d / (vix_chg_std + 1e-6)
breadth_momentum = breadth_chg / (breadth_std + 1e-6)
```

Total: 26 original features → 32 features

## Results: SIGNIFICANT DEGRADATION

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Mean OOS Accuracy | 65.0% | 51.4% | -13.6% 🔴 |
| Vs Baseline (60%) | +5.0% | -8.7% | -13.7% 🔴 |
| Accuracy Std Dev | ~5% | 17.1% | ↑ 3.4x more unstable |
| Min Fold Accuracy | N/A | 27.8% | Worse than random! |
| Max Fold Accuracy | N/A | 87.2% | Huge variance |

## Root Causes

### 1. **NaN Cascading** (PRIMARY)
- `diff()` operations created NaNs at window start
- These NaNs propagated through feature matrix
- Model had to impute/drop more data, losing information
- Different fold windows had different NaN patterns = instability

### 2. **Multicollinearity**
- Acceleration features are mathematically derived from existing features
- `hy_spread_accel = diff(hy_spread_chg)` is just another transformation of `hy_spread`
- Spreads already capture trend information via `hy_spread_chg_21d` and `hy_spread_chg_63d`
- Adding derivatives added **noise without new signal**

### 3. **Non-Stationarity**
- Acceleration values have different distributions across regimes
- In calm markets: spreads move slowly, acceleration ≈ 0
- In crisis: spreads spike, acceleration becomes erratic
- Model couldn't learn consistent patterns across regimes

### 4. **Lack of Regime-Context**
- Raw acceleration is meaningless out of context
- "Spreads widening 5 bps" is normal in some regimes, extreme in others
- Features should be **regime-relative**, not absolute

## Key Insight

**Your intuition was RIGHT** ✓
- Capturing regime transitions IS valuable
- Spread acceleration IS conceptually sound
- The implementation just didn't work

This is why walk-forward validation is essential — it caught this immediately.

## Future Exploration (Phase 2 candidates)

### Approach A: Smarter Normalization
```python
# Instead of raw acceleration:
hy_spread_accel_zscore = (hy_spread_accel / hy_spread_accel.rolling(63).std())
# Capture "unusual acceleration" relative to historical baseline
```

### Approach B: Regime-Relative Momentum
```python
# Compute within each regime separately
for regime in ['EXPANSION', 'CONTRACTION', ...]:
    regime_accel_baseline = data[regime_label == regime]['hy_spread_accel'].std()
    feature['hy_accel_vs_regime'] = hy_spread_accel / regime_accel_baseline
```

### Approach C: State-Space Features
- Use regime probabilities from Phase 5 predictions
- Create features like "days since regime started"
- "How confident is model this regime is ending?"
- More sophisticated temporal context

### Approach D: Domain-Specific Indicators
- Instead of generic acceleration, use **economic reason**
  - "Credit stress accelerating" = HY + IG spread divergence
  - "Volatility spike" = VIX level + realized vol mismatch
  - "Rally breaking" = Breadth divergence + price momentum divergence
- Combine indicators into **regime-transition scores**

### Approach E: Separate Models
- Model 1: Detect regime (current)
- Model 2: Detect regime **transitions** using acceleration/momentum
- Ensemble the outputs

## Recommendation

**Keep 26-feature model as production baseline.**
- Proven: 65% OOS accuracy, consistent, interpretable
- Production-grade: Used in actual portfolio system
- Foundation: Good place to experiment from

**Revisit acceleration/momentum in Phase 2 with:**
- Careful NaN handling before features hit model
- Proper multicollinearity diagnostics (correlation matrix)
- Regime-relative normalization
- Domain-grounded feature definitions

## Interview Takeaway

"Adding features isn't always better. We tried acceleration/momentum signals to capture regime transitions, but multicollinearity and non-stationarity degraded performance from 65% to 51%. Walk-forward validation caught this immediately. Sometimes simpler is better—our 26-feature model with 65% OOS accuracy is more robust than overengineered models."
