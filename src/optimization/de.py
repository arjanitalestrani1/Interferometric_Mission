"""
Differential Evolution (DE) adaptiv - varianti jDE (Brest et al., 2006).

Ideja: femija krijohet duke i shtuar nje individi DIFERENCEN mes dy individeve tjere.
Kur popullata eshte e shperndare, diferencat jane te medha (kerkim i gjere);
kur popullata afrohet te zgjidhja, diferencat zvogelohen (kerkim i imet).
Keshtu madhesia e hapit pershtatet vete, pa sigma fikse si te GA.

Hapat per cdo individ x_i ne cdo gjenerate:
  1. Mutacioni  : v = a + F * (b - c)      (a, b, c individe te ndryshem te rastesishem)
  2. Kryqezimi  : femija merr nga v me probabilitet CR, perndryshe nga x_i
  3. Perzgjedhja: femija zevendeson x_i vetem nese ka fitness me te mire ose te barabarte

jDE: cdo individ ka F dhe CR te vetat, qe ndryshojne rastesisht dhe ruhen
vetem kur prodhojne femije te suksesshem. Keshtu algoritmi meson parametrat vete.
"""
import numpy as np


def inicializo(lb, ub, pop_size, n_sat, rng, shkalla_shpejtesise=1.0):
    """Popullata fillestare. Me shkalla_shpejtesise < 1, shpejtesite nisin te vogla
    (pozicionet jane 3*n_sat variablat e para, shpejtesite 3*n_sat te fundit)."""
    pop = rng.uniform(lb, ub, size=(pop_size, lb.size))
    pop[:, 3 * n_sat:] *= shkalla_shpejtesise
    return pop


def run_de(udp, max_vleresime=20000, pop_size=50, shkalla_shpejtesise=1.0,
           tau_F=0.1, tau_CR=0.1, seed=None, verbose=True):
    """
    Ekzekuton jDE mbi problemin `udp`.

    Kthen:
        best_x   - zgjidhja me e mire e gjetur
        best_f   - fitness-i i saj
        historia - lista me (numri i vleresimeve, fitness-i me i mire deri atehere)
    """
    rng = np.random.default_rng(seed)
    lb, ub = map(np.array, udp.get_bounds())
    dim = lb.size

    pop = inicializo(lb, ub, pop_size, udp.n_sat, rng, shkalla_shpejtesise)
    fitness = np.array([udp.fitness(x)[0] for x in pop])
    vleresime = pop_size

    # Parametrat e cdo individi (jDE)
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)

    best_i = np.argmin(fitness)
    best_x, best_f = pop[best_i].copy(), fitness[best_i]
    historia = [(vleresime, best_f)]
    gjenerata = 0

    while vleresime + pop_size <= max_vleresime:
        gjenerata += 1
        for i in range(pop_size):
            # Pershtatja e parametrave (jDE): me probabilitet tau provohet vlere e re
            F_i = rng.uniform(0.1, 1.0) if rng.random() < tau_F else F[i]
            CR_i = rng.random() if rng.random() < tau_CR else CR[i]

            # 1. Mutacioni DE/rand/1: tre individe te ndryshem nga i
            a, b, c = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            v = pop[a] + F_i * (pop[b] - pop[c])
            v = np.clip(v, lb, ub)

            # 2. Kryqezimi binomial: te pakten nje variabel merret nga v
            maska = rng.random(dim) < CR_i
            maska[rng.integers(dim)] = True
            femija = np.where(maska, v, pop[i])

            # 3. Perzgjedhja: femija mbetet vetem nese nuk eshte me i keq
            f = udp.fitness(femija)[0]
            vleresime += 1
            if f <= fitness[i]:
                pop[i], fitness[i] = femija, f
                F[i], CR[i] = F_i, CR_i      # ruhen parametrat qe dhane sukses
                if f < best_f:
                    best_f, best_x = f, femija.copy()

        historia.append((vleresime, best_f))
        if verbose and gjenerata % 20 == 0:
            print(f"[DE] gjenerata {gjenerata:4d} | vleresime {vleresime:6d} | me i miri {best_f:.5f}")

    return best_x, best_f, historia
