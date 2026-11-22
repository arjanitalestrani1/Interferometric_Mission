"""
Ekzekuton GA ne rastin e lehte (DRO, N=5, grid 11) dhe ruan grafikun e konvergjences.
Ekzekutimi:  python -m experiments.run_ga
"""
import os
import time
import matplotlib.pyplot as plt
from src.evaluation.spoc_esa import orbital_golomb_array
from src.optimization.ga import run_ga

# Orbita DRO
ic = [0.896508460944940632764, 0., 0.,
      0.000000000000013951082, 0.474817948848534454598, 0.]
period = 2.6905181697222775
N, M, grid_size = 5, 3, 11

udp = orbital_golomb_array(
    n_sat=N, ic=ic, T=period * (M - 1), grid_size=grid_size,
    scaling_factor=1e-4, n_meas=M, inflation_factor=1.23, verbose=False,
)

start = time.time()
best_x, best_f, historia = run_ga(udp, max_vleresime=20000, pop_size=100, seed=1)
koha = time.time() - start

print(f"\nFitness-i me i mire: {best_f:.5f}")
print(f"Koha: {koha:.1f} sekonda")

# Grafiku i konvergjences
os.makedirs("results", exist_ok=True)
vl, fit = zip(*historia)
plt.plot(vl, fit)
plt.xlabel("Numri i vleresimeve")
plt.ylabel("Fitness-i me i mire (me poshte = me mire)")
plt.title(f"GA - DRO, N={N}, grid {grid_size}")
plt.grid(True, alpha=0.3)
plt.savefig("results/ga_konvergjenca.png", dpi=150, bbox_inches="tight")
print("Grafiku u ruajt te results/ga_konvergjenca.png")
plt.show()
