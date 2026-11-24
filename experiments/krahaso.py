"""
Krahason GA dhe DE ne rastin e lehte (DRO, N=5, grid 11), me te njejtin buxhet vleresimesh.
Secili algoritem ekzekutohet 3 here me seed te ndryshem.
Ekzekutimi:  python -m experiments.krahaso
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from src.evaluation.spoc_esa import orbital_golomb_array
from src.optimization.ga import run_ga
from src.optimization.de import run_de

# Orbita DRO
ic = [0.896508460944940632764, 0., 0.,
      0.000000000000013951082, 0.474817948848534454598, 0.]
period = 2.6905181697222775
N, M, grid_size = 5, 3, 11

udp = orbital_golomb_array(
    n_sat=N, ic=ic, T=period * (M - 1), grid_size=grid_size,
    scaling_factor=1e-4, n_meas=M, inflation_factor=1.23, verbose=False,
)

BUXHETI = 20000          # vleresime per secilin ekzekutim
SEEDS = [1, 2, 3]
OPTIMUMI = -3 * (N * (N - 1) + 1) / (2 * grid_size - 1) ** 2   # -0.14286 per N=5, grid 11

algoritmet = {
    "GA": lambda s: run_ga(udp, max_vleresime=BUXHETI, pop_size=100, seed=s, verbose=False),
    "DE": lambda s: run_de(udp, max_vleresime=BUXHETI, pop_size=50, seed=s, verbose=False),
}

rezultatet = {}
for emri, ekzekuto in algoritmet.items():
    rezultatet[emri] = []
    for s in SEEDS:
        start = time.time()
        _, best_f, historia = ekzekuto(s)
        print(f"{emri}  seed {s}:  fitness {best_f:.5f}   ({time.time() - start:.0f} s)")
        rezultatet[emri].append((best_f, historia))

print(f"\nOptimumi teorik: {OPTIMUMI:.5f}")
for emri, lista in rezultatet.items():
    vlerat = [f for f, _ in lista]
    print(f"{emri}:  mesatarja {np.mean(vlerat):.5f}   me i miri {min(vlerat):.5f}"
          f"   ({100 * np.mean(vlerat) / OPTIMUMI:.0f}% e optimumit)")

# Grafiku: nje vije per cdo ekzekutim
os.makedirs("results", exist_ok=True)
ngjyrat = {"GA": "tab:blue", "DE": "tab:orange"}
for emri, lista in rezultatet.items():
    for j, (_, historia) in enumerate(lista):
        vl, fit = zip(*historia)
        plt.plot(vl, fit, color=ngjyrat[emri], alpha=0.8, label=emri if j == 0 else None)
plt.axhline(OPTIMUMI, color="gray", linestyle="--", label="optimumi teorik")
plt.xlabel("Numri i vleresimeve")
plt.ylabel("Fitness-i me i mire (me poshte = me mire)")
plt.title(f"GA kundrejt DE - DRO, N={N}, grid {grid_size}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/krahasimi_ga_de.png", dpi=150, bbox_inches="tight")
print("Grafiku u ruajt te results/krahasimi_ga_de.png")
plt.show()
