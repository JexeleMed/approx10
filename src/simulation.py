"""
Simulation environment for the Approxx 10 approximate multiplier.

Metrics:
  ER  – Error Rate
  MED – Mean Error Distance
  NED – Normalized Error Distance: MED / 2^15
"""

import random
from multiplier import exact_multiply, approx10_multiply

N_SAMPLES = 10_000
BIT_WIDTH  = 8
SEED       = 42
NED_DENOM  = 1 << (2 * BIT_WIDTH - 1)  # 2^15 = 32768


def generate_test_vectors(n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    max_val = (1 << BIT_WIDTH) - 1
    return [(rng.randint(0, max_val), rng.randint(0, max_val)) for _ in range(n)]


def run_simulation(vectors: list[tuple[int, int]]) -> dict:
    errors = []
    error_count = 0

    for a, b in vectors:
        ed = approx10_multiply(a, b) - exact_multiply(a, b)
        errors.append(ed)
        if ed != 0:
            error_count += 1

    n = len(vectors)
    med = sum(abs(e) for e in errors) / n
    return {
        "n_samples":      n,
        "error_count":    error_count,
        "er":             error_count / n * 100,
        "med":            med,
        "ned":            med / NED_DENOM,
        "min_ed":         min(errors),
        "max_ed":         max(errors),
        "mean_ed_signed": sum(errors) / n,
    }


def print_report(stats: dict) -> None:
    print("=" * 54)
    print("  Symulacja mnożnika Approxx 10 (8-bit × 8-bit)")
    print("=" * 54)
    print(f"  Liczba próbek        : {stats['n_samples']:>10,}")
    print(f"  Błędnych wyników     : {stats['error_count']:>10,}")
    print("-" * 54)
    print(f"  ER  (Error Rate)     : {stats['er']:>10.4f} %")
    print(f"  MED (Mean |ED|)      : {stats['med']:>10.4f}")
    print(f"  NED (MED / 2^15)     : {stats['ned']:>10.6f}")
    print("-" * 54)
    print(f"  Śr. błąd ze znakiem  : {stats['mean_ed_signed']:>+10.4f}  (bias)")
    print(f"  Minimalny ED         : {stats['min_ed']:>+10}")
    print(f"  Maksymalny ED        : {stats['max_ed']:>+10}")
    print("=" * 54)


if __name__ == "__main__":
    vectors = generate_test_vectors(N_SAMPLES, SEED)
    stats   = run_simulation(vectors)
    print_report(stats)
