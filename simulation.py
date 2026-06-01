"""
Środowisko symulacyjne dla przybliżonego mnożnika Approxx 10.

Metryki błędu (standard approximate computing):
  ER  – Error Rate: odsetek par, dla których wynik przybliżony ≠ dokładny
  MED – Mean Error Distance: średnia wartość bezwzględna błędu |approx - exact|
  NED – Normalized Error Distance: MED / 2^(2n-1), gdzie n=8
        (normalizacja względem „średniej" maksymalnej wartości iloczynu 8-bit)
"""

import random
from multiplier import exact_multiply, approx10_multiply

# ── Parametry symulacji ───────────────────────────────────────────────────────
N_SAMPLES = 10_000
BIT_WIDTH  = 8
SEED       = 42

# Mianownik NED: 2^(2n-1) = 2^15 = 32768
NED_DENOM = 1 << (2 * BIT_WIDTH - 1)


def generate_test_vectors(n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)         # deterministyczne: te same wektory przy każdym uruchomieniu
    max_val = (1 << BIT_WIDTH) - 1   # 255
    return [(rng.randint(0, max_val), rng.randint(0, max_val)) for _ in range(n)]


def run_simulation(vectors: list[tuple[int, int]]) -> dict:
    errors = []
    error_count = 0

    for a, b in vectors:
        exact  = exact_multiply(a, b)
        approx = approx10_multiply(a, b)
        ed = approx - exact          # Error Distance (ze znakiem)
        abs_ed = abs(ed)
        errors.append(ed)
        if ed != 0:
            error_count += 1

    n = len(vectors)
    er  = error_count / n * 100
    med = sum(abs(e) for e in errors) / n
    ned = med / NED_DENOM

    # Dodatkowe statystyki pomocnicze
    min_ed = min(errors)
    max_ed = max(errors)
    mean_ed_signed = sum(errors) / n    # ujemna wartość = systematyczne niedoszacowanie przez mnożnik

    return {
        "n_samples":       n,
        "error_count":     error_count,
        "er":              er,
        "med":             med,
        "ned":             ned,
        "min_ed":          min_ed,
        "max_ed":          max_ed,
        "mean_ed_signed":  mean_ed_signed,
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
