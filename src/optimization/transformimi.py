"""
Transformimi i hapesires se kerkimit.

PROBLEMI: ne hapesiren origjinale algoritmi kerkon (pozicion, shpejtesi) per cdo satelit.
Shpejtesite e mira jane shume te vogla (rreth +-0.02, kur kufijte jane +-10), keshtu qe
pothuajse cdo zgjidhje e provuar i con satelitet jashte kornizes pas nje orbite.

IDEJA: levizja relative e sateliteve eshte LINEARE. ESA e llogarit me heyoka nje matrice 6x6
(STM - State Transition Matrix) per cdo matje k:

    pozicioni_k = A_k @ p + B_k @ v        (A_k, B_k jane pjeset 3x3 te STM-se)

Prandaj, ne vend qe algoritmi te zgjedhe shpejtesine v, e leme te zgjedhe
KU DO TE JETE SATELITI NE MATJEN E DYTE (q1). Shpejtesia llogaritet nga ekuacioni:

    v = B_1^(-1) @ (inflation * q1 - A_1 @ p)

Keshtu, per cdo zgjidhje qe provon algoritmi, satelitet jane BRENDA kornizes ne matjen
e pare dhe te dyte. Algoritmi duhet te kujdeset vetem per matjen e trete.

Variablat e reja jane te gjitha ne [-1, 1]:
    y = [p (pozicioni ne matjen 1),  q1 (pozicioni ne matjen 2)]
"""
import numpy as np


class ProblemiTransformuar:
    """Mbeshtjell problemin e ESA-s. Algoritmet e perdorin njesoj si `udp`
    (kane get_bounds, fitness dhe n_sat), por kerkojne ne hapesiren e re."""

    def __init__(self, udp):
        self.udp = udp
        self.n_sat = udp.n_sat
        stm_1 = udp.stms[1]                       # STM ne matjen e dyte
        self.A1 = stm_1[:3, :3]                   # si ndikon pozicioni fillestar
        self.B1_inv = np.linalg.inv(stm_1[:3, 3:])  # anasjellta e ndikimit te shpejtesise
        self.inflation = udp.inflation_factor

    def get_bounds(self):
        return [-1.0] * 6 * self.n_sat, [1.0] * 6 * self.n_sat

    def dekodo(self, y):
        """Nga variablat e reja (p, q1) ne variablat origjinale te ESA-s (p, v)."""
        N = self.n_sat
        p = y[:3 * N].reshape(3, N)               # rreshti 0 = x e te gjithe sateliteve, etj.
        q1 = y[3 * N:].reshape(3, N)
        v = self.B1_inv @ (self.inflation * q1 - self.A1 @ p)
        x = np.concatenate([p.ravel(), v.ravel()])
        return np.clip(x, -10.0, 10.0)            # brenda kufijve te ESA-s

    def fitness(self, y):
        return self.udp.fitness(self.dekodo(y))
