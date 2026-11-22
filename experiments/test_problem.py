"""
Testi i pare: krijon problemin GRAIL dhe vlereson zgjidhje te rastesishme.
"""
import time
import numpy as np
from src.evaluation.spoc_esa import orbital_golomb_array

# Orbita DRO (rasti me i lehte)
ic = [0.896508460944940632764, 0., 0.,
      0.000000000000013951082, 0.474817948848534454598, 0.]
period = 2.6905181697222775

N = 5
M = 3
grid_size = 11

udp = orbital_golomb_array(
    n_sat=N, ic=ic, T=period * (M - 1), grid_size=grid_size,
    scaling_factor=1e-4, n_meas=M, inflation_factor=1.23, verbose=False,
)

lb, ub = udp.get_bounds()
lb, ub = np.array(lb), np.array(ub)
rng = np.random.default_rng(42)

# Ngrohja: thirrja e pare ngarkon bibliotekat, nuk e matim
udp.fitness(rng.uniform(lb, ub))


def testo(pershkrimi, gjenero, n=20):
    rezultatet = []
    start = time.time()
    for _ in range(n):
        rezultatet.append(udp.fitness(gjenero())[0])
    koha = (time.time() - start) / n
    print(f"\n{pershkrimi}")
    print(f"  me i miri: {min(rezultatet):.5f}   mesatarja: {np.mean(rezultatet):.5f}")
    print(f"  koha per nje vleresim: {koha * 1000:.2f} ms")


# A) Krejt e rastesishme (shpejtesi deri ne +-10)
testo("A) Pozicione dhe shpejtesi krejt te rastesishme",
      lambda: rng.uniform(lb, ub))


# B) Pozicione te rastesishme, shpejtesi te vogla (+-0.1)
def shpejtesi_te_vogla():
    x = rng.uniform(lb, ub)
    x[3 * N:] *= 0.01
    return x


testo("B) Pozicione te rastesishme, shpejtesi te vogla", shpejtesi_te_vogla)


# C) Pozicione te rastesishme, shpejtesi zero
def shpejtesi_zero():
    x = rng.uniform(lb, ub)
    x[3 * N:] = 0.0
    return x


testo("C) Pozicione te rastesishme, shpejtesi zero", shpejtesi_zero)