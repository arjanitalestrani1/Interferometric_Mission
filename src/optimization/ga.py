"""
Algoritmi Gjenetik (GA) - versioni baze, per krahasim me algoritmet tjera.

Hapat ne cdo gjenerate:
  1. Vleresimi   - llogaritet fitness-i i cdo individi
  2. Elitizmi    - individet me te mire kalojne direkt ne gjeneraten e re
  3. Perzgjedhja - prinderit zgjidhen me turnir
  4. Kryqezimi   - femija eshte perzierje e dy prinderve
  5. Mutacioni   - disa vlera te femijes ndryshohen pak rastesisht
"""
import numpy as np


def turnir(popullata, fitness, rng, k=3):
    """Zgjedh k individe rastesisht dhe kthen me te mirin (fitness me i vogel)."""
    kandidatet = rng.integers(0, len(popullata), size=k)
    fitues = kandidatet[np.argmin(fitness[kandidatet])]
    return popullata[fitues]


def kryqezim(prindi1, prindi2, rng):
    """Kryqezim aritmetik: per cdo variabel merret nje perzierje e rastesishme e dy prinderve."""
    alfa = rng.random(prindi1.size)
    return alfa * prindi1 + (1 - alfa) * prindi2


def mutacion(individi, lb, ub, rng, prob, sigma):
    """Me probabilitet `prob`, cdo variabel merr zhurme Gaussiane.
    `sigma` eshte relative ndaj gjeresise se intervalit te variables."""
    maska = rng.random(individi.size) < prob
    zhurma = rng.normal(0.0, sigma * (ub - lb), size=individi.size)
    individi = individi + maska * zhurma
    return np.clip(individi, lb, ub)


def run_ga(udp, max_vleresime=20000, pop_size=100, prob_mutacioni=0.1,
           sigma=0.05, n_elite=2, k_turnir=3, seed=None, verbose=True):
    """
    Ekzekuton GA mbi problemin `udp`.

    Kthen:
        best_x      - zgjidhja me e mire e gjetur
        best_f      - fitness-i i saj
        historia    - lista me (numri i vleresimeve, fitness-i me i mire deri atehere)
    """
    rng = np.random.default_rng(seed)
    lb, ub = map(np.array, udp.get_bounds())
    dim = lb.size

    # Popullata fillestare: krejt e rastesishme brenda kufijve
    popullata = rng.uniform(lb, ub, size=(pop_size, dim))
    fitness = np.array([udp.fitness(x)[0] for x in popullata])
    vleresime = pop_size

    best_i = np.argmin(fitness)
    best_x, best_f = popullata[best_i].copy(), fitness[best_i]
    historia = [(vleresime, best_f)]
    gjenerata = 0

    while vleresime + (pop_size - n_elite) <= max_vleresime:
        gjenerata += 1

        # Elitizmi: ruajme n_elite individet me te mire
        renditja = np.argsort(fitness)
        e_re = [popullata[i].copy() for i in renditja[:n_elite]]
        fitness_e_re = [fitness[i] for i in renditja[:n_elite]]

        # Krijojme femijet
        while len(e_re) < pop_size:
            p1 = turnir(popullata, fitness, rng, k_turnir)
            p2 = turnir(popullata, fitness, rng, k_turnir)
            femija = kryqezim(p1, p2, rng)
            femija = mutacion(femija, lb, ub, rng, prob_mutacioni, sigma)
            e_re.append(femija)
            fitness_e_re.append(udp.fitness(femija)[0])
            vleresime += 1

        popullata = np.array(e_re)
        fitness = np.array(fitness_e_re)

        i = np.argmin(fitness)
        if fitness[i] < best_f:
            best_f, best_x = fitness[i], popullata[i].copy()
        historia.append((vleresime, best_f))

        if verbose and gjenerata % 10 == 0:
            print(f"[GA] gjenerata {gjenerata:4d} | vleresime {vleresime:6d} | me i miri {best_f:.5f}")

    return best_x, best_f, historia
