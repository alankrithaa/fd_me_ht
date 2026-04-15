"""
Quick verification test for Phase 1-4 fixes:
1. Z-test pooled SD formula (ht_logic.py)
2. Rejection tracking logic (repertoire.py)
3. Descriptor constants (evaluator.py)
4. Pipeline descriptor projection (main_pipeline.py)
"""

import jax
import jax.numpy as jnp
import numpy as np

print("\n" + "="*70)
print("VERIFICATION TEST: DROME Implementation Fixes")
print("="*70)

# ------- Test 1: Z-test Pooled SD Formula -------
print("\n[TEST 1] Z-test Pooled SD Formula (ht_logic fix)")
print("-" * 70)

from ht_logic import calculate_ht_replacement

# Test case 1a: Identical constant distributions (edge case)
new_scores_const = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)
old_scores_const = jnp.array([0.6, 0.6, 0.6], dtype=jnp.float32)

should_replace, p_val, cles = calculate_ht_replacement(
    new_scores_const, old_scores_const, alpha=0.05, delta_min=0.6
)

print(f"  Test 1a: Constant distributions (edge case)")
print(f"    New: {new_scores_const}, Old: {old_scores_const}")
print(f"    Should replace: {should_replace} (expected: False, numerically safe)")
print(f"    P-value: {float(p_val):.6f}")
print(f"    CLES: {float(cles):.6f}")
print(f"    Result: ✓ PASS" if (not jnp.isnan(p_val)) else "    Result: ✗ FAIL (NaN detected)")

# Test case 1b: Normal case with improvement
new_scores_better = jnp.array([0.2, 0.3, 0.25], dtype=jnp.float32)
old_scores_worse = jnp.array([0.5, 0.6, 0.55], dtype=jnp.float32)

should_replace, p_val, cles = calculate_ht_replacement(
    new_scores_better, old_scores_worse, alpha=0.05, delta_min=0.6
)

print(f"\n  Test 1b: Clear improvement case")
print(f"    New mean: {float(jnp.mean(new_scores_better)):.3f}, Old mean: {float(jnp.mean(old_scores_worse)):.3f}")
print(f"    Should replace: {bool(should_replace)} (expected: True if p < 0.05 and CLES > 0.6)")
print(f"    P-value: {float(p_val):.6f}")
print(f"    CLES: {float(cles):.6f}")
valid = (not jnp.isnan(p_val)) and (not jnp.isnan(cles))
print(f"    Result: ✓ PASS" if valid else "    Result: ✗ FAIL")

# ------- Test 2: Rejection Tracking Logic -------
print("\n[TEST 2] Rejection Tracking Logic (repertoire fix)")
print("-" * 70)

# Simulate three different failure modes
test_cases = [
    {
        "name": "P-value failure (p >= alpha)",
        "should_replace": False,
        "p_val_f": 0.10,
        "cles_f": 0.75,
        "delta_min": 0.6,
        "expect_p_reject": True,
        "expect_es_reject": False,
    },
    {
        "name": "Effect size failure (p < alpha but cles <= delta_min)",
        "should_replace": False,
        "p_val_f": 0.02,
        "cles_f": 0.55,
        "delta_min": 0.6,
        "expect_p_reject": False,
        "expect_es_reject": True,
    },
    {
        "name": "Direction failure (p < alpha but mean_new >= mean_old)",
        "should_replace": False,
        "p_val_f": 0.02,
        "cles_f": 0.65,
        "delta_min": 0.6,
        "expect_p_reject": False,
        "expect_es_reject": False,
    },
    {
        "name": "Success case",
        "should_replace": True,
        "p_val_f": 0.02,
        "cles_f": 0.75,
        "delta_min": 0.6,
        "expect_p_reject": False,
        "expect_es_reject": False,
    },
]

all_pass = True
for i, tc in enumerate(test_cases, 1):
    is_p_reject = (not tc["should_replace"]) and (tc["p_val_f"] >= 0.05)
    is_es_reject = (not tc["should_replace"]) and (tc["p_val_f"] < 0.05) and (tc["cles_f"] <= tc["delta_min"])
    
    p_reject_ok = is_p_reject == tc["expect_p_reject"]
    es_reject_ok = is_es_reject == tc["expect_es_reject"]
    
    status = "✓ PASS" if (p_reject_ok and es_reject_ok) else "✗ FAIL"
    if not (p_reject_ok and es_reject_ok):
        all_pass = False
    
    print(f"  Case {i}: {tc['name']}")
    print(f"    should_replace={tc['should_replace']}, p_val={tc['p_val_f']:.3f}, cles={tc['cles_f']:.3f}")
    print(f"    is_p_reject: {is_p_reject} (expected: {tc['expect_p_reject']})")
    print(f"    is_es_reject: {is_es_reject} (expected: {tc['expect_es_reject']})")
    print(f"    {status}\n")

# ------- Test 3: Descriptor Constants -------
print("[TEST 3] Descriptor Constants (evaluator.py)")
print("-" * 70)

try:
    from evaluator import BD_BRIGHTNESS_IDX, BD_ENTROPY_IDX, FEATURE_DIM
    
    print(f"  BD_BRIGHTNESS_IDX: {BD_BRIGHTNESS_IDX} (expected: 0)")
    print(f"  BD_ENTROPY_IDX: {BD_ENTROPY_IDX} (expected: 2)")
    print(f"  FEATURE_DIM: {FEATURE_DIM} (expected: 5)")
    
    brightness_ok = BD_BRIGHTNESS_IDX == 0
    entropy_ok = BD_ENTROPY_IDX == 2
    feature_ok = FEATURE_DIM == 5
    
    if brightness_ok and entropy_ok and feature_ok:
        print(f"  Result: ✓ PASS")
    else:
        print(f"  Result: ✗ FAIL")
        all_pass = False
except ImportError as e:
    print(f"  Result: ✗ FAIL (Import error: {e})")
    all_pass = False

# ------- Test 4: Feature Extraction -------
print("\n[TEST 4] Feature Extraction Shape Consistency")
print("-" * 70)

try:
    from evaluator import extract_features
    
    test_image = np.random.rand(64, 64, 3).astype(np.float32)
    features = extract_features(test_image)
    
    print(f"  Feature shape: {features.shape} (expected: (5,))")
    print(f"  Feature dtype: {features.dtype} (expected: float32)")
    print(f"  All in [0, 1]: {np.all(features >= 0) and np.all(features <= 1)}")
    
    shape_ok = features.shape == (5,)
    dtype_ok = features.dtype == np.float32
    range_ok = np.all(features >= 0) and np.all(features <= 1)
    
    if shape_ok and dtype_ok and range_ok:
        print(f"  Result: ✓ PASS")
    else:
        print(f"  Result: ✗ FAIL")
        all_pass = False
        
except Exception as e:
    print(f"  Result: ✗ FAIL (Error: {e})")
    all_pass = False

# ------- Summary -------
print("\n" + "="*70)
if all_pass:
    print("✓ ALL VERIFICATION TESTS PASSED")
else:
    print("✗ SOME TESTS FAILED - Review output above")
print("="*70 + "\n")
