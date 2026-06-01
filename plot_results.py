"""
plot_results.py – wizualizacje dla projektu Approxx 10.

Generuje 6 wykresów PNG:
  scatter_approxx10.png    – exact vs. approx, punkty pokolorowane wg |ED|
  histogram_errors.png     – rozkład Error Distance (ED) dla wszystkich 65 536 par
  heatmap_errors.png       – mapa ciepła ED w przestrzeni wejść 0–255 × 0–255
  ecm_comparison.png       – histogram i metryki: z ECM vs. bez ECM
  image_psnr.png           – mnożenie pixel-wise z metryką PSNR
  error_vs_magnitude.png   – ED w funkcji wartości dokładnego iloczynu
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

SEED = 42

# ── Precompute 256×256 lookup tables (wektoryzacja operacji bitowych) ─────────
def _build_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A, B = np.meshgrid(
        np.arange(256, dtype=np.int64),
        np.arange(256, dtype=np.int64),
        indexing='ij',
    )
    exact = A * B

    upper = np.zeros((256, 256), dtype=np.int64)
    # Zewnętrzne nawiasy konieczne: << ma wyższy priorytet niż &,
    # bez nich numpy liczyłoby A>>i & (((B>>j)&1) << (i+j)) – błędny wynik.
    for i in range(8):
        for j in range(max(8 - i, 0), 8):
            upper += (((A >> i) & 1) & ((B >> j) & 1)) << (i + j)

    or1 = np.zeros((256, 256), dtype=np.int64)
    for i, j in [(0, 7), (1, 6), (2, 5), (3, 4)]:
        or1 |= ((A >> i) & 1) & ((B >> j) & 1)

    or2 = np.zeros((256, 256), dtype=np.int64)
    for i, j in [(4, 3), (5, 2), (6, 1), (7, 0)]:
        or2 |= ((A >> i) & 1) & ((B >> j) & 1)

    approx_ecm    = (upper + 6 + (or1 + or2) * 256) & 0xFFFF
    approx_no_ecm = (upper + 6) & 0xFFFF
    return exact, approx_ecm, approx_no_ecm


print("Precomputing lookup tables...", end=" ", flush=True)
EXACT, APPROX, APPROX_NO_ECM = _build_matrices()
ED        = (APPROX     - EXACT).astype(np.int32)
ED_NO_ECM = (APPROX_NO_ECM - EXACT).astype(np.int32)
print("done.\n")

# Losowe indeksy do scatter / error-vs-magnitude (reprodukowalne)
_rng  = np.random.default_rng(SEED)
_a_s  = _rng.integers(0, 256, 10_000)
_b_s  = _rng.integers(0, 256, 10_000)


# ── 1. Scatter: exact vs. approx ─────────────────────────────────────────────
def plot_scatter() -> None:
    ex  = EXACT[_a_s, _b_s]
    ap  = APPROX[_a_s, _b_s]
    ed  = np.abs(ED[_a_s, _b_s])

    fig, ax = plt.subplots(figsize=(7, 7))
    # Kolor jako trzeci wymiar informacji: sama oś Y nie ujawnia, gdzie skupiają się duże błędy.
    sc = ax.scatter(ex, ap, c=ed, cmap='plasma', s=3, alpha=0.45, vmin=0, vmax=500)
    fig.colorbar(sc, ax=ax, label="|Error Distance|")
    ax.plot([0, 65025], [0, 65025], 'r-', lw=1.5, label='y = x (ideal)')
    ax.set_xlabel("Wynik dokładny")
    ax.set_ylabel("Wynik Approxx 10")
    ax.set_title("Approxx 10 vs. mnożenie dokładne\n(10 000 losowych par, kolor = |ED|)")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 65536)
    ax.set_ylim(0, 65536)
    ax.set_aspect('equal')
    ax.grid(True, ls='--', lw=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig("scatter_approxx10.png", dpi=150)
    plt.close()
    print("Zapisano: scatter_approxx10.png")


# ── 2. Histogram rozkładu błędów ─────────────────────────────────────────────
def plot_histogram() -> None:
    ed_flat = ED.flatten().astype(float)  # wszystkie 65 536 par – rozkład dokładny, nie próbka
    mean_ed = ed_flat.mean()
    med     = np.abs(ed_flat).mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ed_flat, bins=120, color='steelblue', alpha=0.85, edgecolor='none')
    # Długi lewy ogon = duże błędy dla wysokich a×b, gdzie lower_part >> 6.
    ax.axvline(0,       color='limegreen', lw=1.8, ls='--', label='ED = 0')
    ax.axvline(mean_ed, color='crimson',   lw=1.8, ls='--',
               label=f'Średnia ED = {mean_ed:.1f}')
    ax.set_xlabel("Error Distance  (approx − exact)")
    ax.set_ylabel("Liczba przypadków")
    ax.set_title("Rozkład błędów Approxx 10  (wszystkie 65 536 kombinacji)")
    ax.legend(fontsize=10)
    ax.text(0.97, 0.95, f"MED = {med:.1f}\nNED = {med/2**15:.5f}",
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.grid(True, axis='y', ls='--', lw=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig("histogram_errors.png", dpi=150)
    plt.close()
    print("Zapisano: histogram_errors.png")


# ── 3. Heatmap 256×256 ───────────────────────────────────────────────────────
def plot_heatmap() -> None:
    vmin = int(np.percentile(ED,  2))
    vmax = int(np.percentile(ED, 99))
    # percentile(2/99) ucina skrajne wartości – reszta mapy zyskuje kontrast.
    # vcenter=0: biały = ED=0, czerwień = nadszacowanie, niebieski = niedoszacowanie.
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=max(vmax, 1))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ED, origin='lower', cmap='RdBu_r', norm=norm,
                   extent=[0, 255, 0, 255], aspect='equal')
    fig.colorbar(im, ax=ax, label="Error Distance  (approx − exact)")
    # Geometryczne wzory (przekątne pasy) = bezpośredni ślad struktury iloczynów cząstkowych.
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title("Mapa ciepła błędów Approxx 10\n"
                 "(czerwień = ED > 0, niebieski = ED < 0)")
    plt.tight_layout()
    plt.savefig("heatmap_errors.png", dpi=150)
    plt.close()
    print("Zapisano: heatmap_errors.png")


# ── 4. ECM comparison ────────────────────────────────────────────────────────
def plot_ecm_comparison() -> None:
    ed_e  = ED.flatten().astype(float)
    ed_ne = ED_NO_ECM.flatten().astype(float)

    med_e,  ned_e  = np.abs(ed_e).mean(),  np.abs(ed_e).mean()  / 2**15
    med_ne, ned_ne = np.abs(ed_ne).mean(), np.abs(ed_ne).mean() / 2**15

    bins = np.linspace(min(ed_ne.min(), ed_e.min()),
                       max(ed_ne.max(), ed_e.max()), 120)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # Lewy panel: nakładające się histogramy – przesunięcie o ≈350 to E[korekcja ECM].
    # Prawy panel: wartości liczbowe MED i NED gotowe do zacytowania w prezentacji.
    ax1.hist(ed_ne, bins=bins, alpha=0.6, color='tomato',
             label=f'Bez ECM  (MED = {med_ne:.0f})')
    ax1.hist(ed_e,  bins=bins, alpha=0.6, color='steelblue',
             label=f'Z ECM    (MED = {med_e:.0f})')
    ax1.axvline(0, color='black', lw=1.2, ls='--')
    ax1.set_xlabel("Error Distance")
    ax1.set_ylabel("Liczba przypadków")
    ax1.set_title("Rozkład ED: z ECM vs. bez ECM")
    ax1.legend(fontsize=10)
    ax1.grid(True, axis='y', ls='--', lw=0.5, alpha=0.5)

    labels  = ['MED', 'NED × 10⁴']
    vals_ne = [med_ne, ned_ne * 1e4]
    vals_e  = [med_e,  ned_e  * 1e4]
    x, w    = np.arange(2), 0.35
    ax2.bar(x - w/2, vals_ne, w, label='Bez ECM', color='tomato',    alpha=0.85)
    ax2.bar(x + w/2, vals_e,  w, label='Z ECM',   color='steelblue', alpha=0.85)
    for rect in ax2.patches:
        h = rect.get_height()
        ax2.text(rect.get_x() + w / 2, h * 1.02, f"{h:.2f}",
                 ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("Poprawa metryk dzięki ECM")
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', ls='--', lw=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig("ecm_comparison.png", dpi=150)
    plt.close()
    print("Zapisano: ecm_comparison.png")


# ── 5. Image multiplication + PSNR ──────────────────────────────────────────
def plot_image_psnr() -> None:
    t = np.linspace(0, 4 * np.pi, 128)
    X, Y   = np.meshgrid(t, t)
    img_a  = ((np.sin(X) * np.cos(Y) * 0.5 + 0.5) * 255).astype(np.uint8)
    img_b  = np.random.default_rng(SEED + 1).integers(0, 256, (128, 128), dtype=np.uint8)

    # Fancy-indexing: EXACT[img_a, img_b] dla każdego piksela (i,j) pobiera
    # EXACT[img_a[i,j], img_b[i,j]] – indeksowanie element-wise, bez pętli Pythona.
    exact_img  = EXACT[img_a,  img_b]
    approx_img = APPROX[img_a, img_b]
    diff_img   = np.abs(approx_img.astype(float) - exact_img.astype(float))

    mse  = np.mean((exact_img.astype(float) - approx_img.astype(float)) ** 2)
    # MAX = 255×255 = 65025; PSNR = 10·log₁₀(MAX² / MSE) – standard dla iloczynów 8-bit.
    psnr = 10 * np.log10(65025.0 ** 2 / mse) if mse > 0 else float('inf')

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"Mnożenie pixel-wise: Approxx 10 vs. dokładne  |  PSNR = {psnr:.2f} dB",
        fontsize=13,
    )
    titles = ["Obraz A (wejście)", "Maska B (wejście)",
              "Wynik dokładny",    "Wynik Approxx 10"]
    imgs   = [img_a, img_b, exact_img, approx_img]
    vmaxes = [255, 255, 65025, 65025]
    for ax, img, title, vmax in zip(axes, imgs, titles, vmaxes):
        ax.imshow(img, cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("image_psnr.png", dpi=150)
    plt.close()
    print(f"Zapisano: image_psnr.png  (PSNR = {psnr:.2f} dB)")


# ── 6. Error vs. magnitude of exact product ──────────────────────────────────
def plot_error_vs_magnitude() -> None:
    exact_flat = EXACT.flatten()
    ed_flat    = ED.flatten()

    # Próbka 20 k punktów – wystarczy, żeby widoczna była struktura pasów.
    # Trzy poziomy zagęszczenia punktów = korekcje ECM: 0, +256, +512.
    rng2 = np.random.default_rng(SEED + 2)
    idx  = rng2.choice(len(ed_flat), size=20_000, replace=False)

    # Binowana średnia ED po wartości iloczynu (trend)
    bins     = np.linspace(0, 65025, 150)
    bin_idx  = np.digitize(exact_flat, bins)
    mean_ed_bin = np.array([
        ed_flat[bin_idx == k].mean() if (bin_idx == k).any() else 0
        for k in range(1, len(bins))
    ])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(exact_flat[idx], ed_flat[idx],
               s=1, alpha=0.06, color='steelblue', label='ED (próbka 20k)')
    ax.plot(bin_centers, mean_ed_bin,
            color='crimson', lw=1.8, label='Średnia ED (po binach)')
    ax.axhline(0,               color='limegreen', lw=1.4, ls='--', label='ED = 0')
    ax.axhline(ed_flat.mean(),  color='orange',    lw=1.4, ls='--',
               label=f'Globalna śr. ED = {ed_flat.mean():.1f}')
    ax.set_xlabel("Wynik dokładny  (a × b)")
    ax.set_ylabel("Error Distance  (approx − exact)")
    ax.set_title("Błąd Approxx 10 w funkcji wartości dokładnego iloczynu")
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', lw=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig("error_vs_magnitude.png", dpi=150)
    plt.close()
    print("Zapisano: error_vs_magnitude.png")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_scatter()
    plot_histogram()
    plot_heatmap()
    plot_ecm_comparison()
    plot_image_psnr()
    plot_error_vs_magnitude()
    print("\nGotowe – wygenerowano 6 wykresów.")
