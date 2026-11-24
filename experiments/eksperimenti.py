"""
Eksperimenti i plote: krahason GA, DE, GA + T dhe DE + T ne nje rast te zgjedhur.

Shembuj:
    python -m experiments.eksperimenti --orbita DRO  --N 5  --grid 11
    python -m experiments.eksperimenti --orbita DRO  --N 40 --grid 21
    python -m experiments.eksperimenti --orbita HALO --N 40 --grid 21

Opsione tjera:  --buxheti 20000   (vleresime per ekzekutim)
                --seeds 1 2 3     (cilat seed-e te perdoren)

Rezultatet ruhen ne results/:  <rasti>.csv  dhe  <rasti>.png
"""
import argparse
import csv
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from src.evaluation.spoc_esa import orbital_golomb_array
from src.optimization.ga import run_ga
from src.optimization.de import run_de
from src.optimization.transformimi import ProblemiTransformuar

ORBITAT = {
    "DRO": dict(ic=[0.896508460944940632764, 0., 0.,
                    0.000000000000013951082, 0.474817948848534454598, 0.],
                period=2.6905181697222775),
    "HALO": dict(ic=[1.0829551779304256e+00, -6.9232801936027592e-27, -2.0231744561698364e-01,
                     9.7888791827480806e-15, -2.0102644884016105e-01, 2.4744866465838825e-14],
                 period=2.383491010514447),
}

parser = argparse.ArgumentParser()
parser.add_argument("--orbita", default="DRO", choices=ORBITAT.keys())
parser.add_argument("--N", type=int, default=5)
parser.add_argument("--grid", type=int, default=11)
parser.add_argument("--buxheti", type=int, default=20000)
parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
args = parser.parse_args()

M = 3
orb = ORBITAT[args.orbita]
udp = orbital_golomb_array(
    n_sat=args.N, ic=orb["ic"], T=orb["period"] * (M - 1), grid_size=args.grid,
    scaling_factor=1e-4, n_meas=M, inflation_factor=1.23, verbose=False,
)
udp_t = ProblemiTransformuar(udp)
B = args.buxheti

algoritmet = {
    "GA":     lambda s: run_ga(udp,   max_vleresime=B, pop_size=100, seed=s, verbose=False),
    "DE":     lambda s: run_de(udp,   max_vleresime=B, pop_size=50,  seed=s, verbose=False),
    "GA + T": lambda s: run_ga(udp_t, max_vleresime=B, pop_size=100, seed=s, verbose=False),
    "DE + T": lambda s: run_de(udp_t, max_vleresime=B, pop_size=50,  seed=s, verbose=False),
}

rasti = f"{args.orbita}_N{args.N}_grid{args.grid}"
print(f"Rasti: {rasti}   buxheti: {B} vleresime   seeds: {args.seeds}\n")

rreshtat, historite = [], {}
for emri, ekzekuto in algoritmet.items():
    historite[emri] = []
    for s in args.seeds:
        start = time.time()
        best_x, best_f, historia = ekzekuto(s)
        koha = time.time() - start
        print(f"{emri:7s} seed {s}:  fitness {best_f:.5f}   ({koha:.0f} s)")
        rreshtat.append({"algoritmi": emri, "seed": s, "fitness": best_f, "koha_s": round(koha, 1)})
        historite[emri].append(historia)

# Permbledhja
# Kufiri teorik: N satelite japin te shumten N*(N-1)+1 pika ne planin uv, qe ka (2*grid-1)^2 qeliza.
# Per N te vogel (p.sh. N=5) ky eshte optimumi i sakte; per N=40 eshte vetem kufi i siperm.
qelizat = (2 * args.grid - 1) ** 2
kufiri = -3 * min(args.N * (args.N - 1) + 1, qelizat) / qelizat
print(f"\nKufiri teorik: {kufiri:.5f}")
for emri in algoritmet:
    vlerat = [r["fitness"] for r in rreshtat if r["algoritmi"] == emri]
    print(f"{emri:7s}  mesatarja {np.mean(vlerat):.5f}   devijimi {np.std(vlerat):.5f}   me i miri {min(vlerat):.5f}")

# Ruajtja ne CSV
os.makedirs("results", exist_ok=True)
with open(f"results/{rasti}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["algoritmi", "seed", "fitness", "koha_s"])
    w.writeheader()
    w.writerows(rreshtat)

# Grafiku
stili = {"GA": ("tab:blue", ":"), "DE": ("tab:orange", ":"),
         "GA + T": ("tab:blue", "-"), "DE + T": ("tab:orange", "-")}
for emri, lista in historite.items():
    ngjyra, vija = stili[emri]
    for j, historia in enumerate(lista):
        vl, fit = zip(*historia)
        plt.plot(vl, fit, color=ngjyra, linestyle=vija, alpha=0.8, label=emri if j == 0 else None)
if args.N <= 10:
    plt.axhline(kufiri, color="gray", linestyle="--", label="optimumi teorik")
plt.xlabel("Numri i vleresimeve")
plt.ylabel("Fitness-i me i mire (me poshte = me mire)")
plt.title(f"{args.orbita}, N={args.N}, grid {args.grid}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"results/{rasti}.png", dpi=150, bbox_inches="tight")
print(f"\nU ruajten: results/{rasti}.csv  dhe  results/{rasti}.png")
plt.show()