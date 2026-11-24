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
from src.evaluation.spoc_esa import orbital_golomb_array

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

print(f"Rasti: {args.orbita}, N={args.N}, grid {args.grid}")
print(f"Numri i variablave: {len(udp.get_bounds()[0])}")