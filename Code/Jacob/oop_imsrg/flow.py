from tensornetwork import *
import numpy as np
from oop_imsrg.hamiltonian import *
from oop_imsrg.occupation_tensors import *
from oop_imsrg.generator import *

class Flow(object):
    """Parent class for organization purposes. Ideally, all Flow
    classes should inherit from this class. In this way, AssertionErrors
    can be handled in a general way."""

    def flow():
        print("Function that iterates flow equation once")

class Flow_IMSRG2(Flow):
    """Calculates the flow equations for the IMSRG(2)."""

    def __init__(self, h, occ_t):
        """Class constructor. Instantiates Flow_IMSRG2 object.

        Arguments:

        h -- Hamiltonian object
        occ_t -- OccupationTensors object"""

        assert isinstance(h, Hamiltonian), "Arg 0 must be PairingHamiltonian object"
        assert isinstance(occ_t, OccupationTensors), "Arg 1 must be OccupationTensors object"

        # self.f = h.f
        # self.G = h.G

        self._holes = h.holes
        self._particles = h.particles

        self._occA = occ_t.occA
        self._occB = occ_t.occB
        self._occC = occ_t.occC
        self._occD = occ_t.occD

    # @property
    # def f(self):
    #     return self._f

    # @property
    # def G(self):
    #     return self._G

    # @f.setter
    # def f(self, f):
    #     self._f = f

    # @G.setter
    # def G(self, G):
    #     self._G = G

    def flow(self, gen):
        """Iterates the IMSRG2 flow equations once.

        Arugments:

        gen -- Generator object; generator produces the flow

        Returns:

        (dE, -- zero-body tensor
         df, -- one-body tensor
         dG) -- two-body tensor"""

        assert isinstance(gen, Generator), "Arg 0 must be WegnerGenerator object"

        # f = self.f
        # G = self.G
        f = gen.f
        G = gen.G

        eta1B, eta2B = gen.calc_eta()

        # occA = occ_t.occA
        # occB = occ_t.occB
        # occC = occ_t.occC
        # occD = occ_t.occD

        occA = self._occA
        occB = self._occB
        occC = self._occC
        occD = self._occD

        # - Calculate dE/ds
        # first term
        sum1_0b_1 = ncon([occA, eta1B], [(0, 1, -1, -2), (0, 1)]).numpy()
        sum1_0b = ncon([sum1_0b_1, f], [(0, 1), (1, 0)]).numpy()

        # second term
        sum2_0b_1 = np.matmul(eta2B, occD)
        sum2_0b = ncon([sum2_0b_1, G], [(0, 1, 2, 3), (2, 3, 0, 1)]).numpy()

        dE = sum1_0b + 0.5*sum2_0b


        # - Calculate df/ds
        # first term
        sum1_1b_1 = ncon([eta1B, f], [(-1, 0), (0, -2)]).numpy()
        sum1_1b_2 = np.transpose(sum1_1b_1)
        sum1_1b = sum1_1b_1 + sum1_1b_2

        # second term (might need to fix)
        sum2_1b_1 = ncon([eta1B, G], [(0, 1), (1, -1, 0, -2)]).numpy()
        sum2_1b_2 = ncon([f, eta2B], [(0, 1), (1, -1, 0, -2)]).numpy()
        sum2_1b_3 = sum2_1b_1 - sum2_1b_2
        sum2_1b = ncon([occA, sum2_1b_3],[(-1, -2, 0, 1), (0,1)]).numpy()

        # third term
        sum3_1b_1 = ncon([occC, G], [(-1, -2, -3, 0, 1, 2), (0, 1, 2, -4)]).numpy()
        sum3_1b_2 = ncon([eta2B, sum3_1b_1], [(2, -1, 0, 1), (0, 1, 2, -2)]).numpy()
        sum3_1b_3 = np.transpose(sum3_1b_2)
        sum3_1b = sum3_1b_2 + sum3_1b_3

        df = sum1_1b + sum2_1b + 0.5*sum3_1b


        # - Calculate dG/ds
        # first term (P_ij piece)
        sum1_2b_1 = ncon([eta1B, G], [(-1, 0), (0, -2, -3, -4)]).numpy()
        sum1_2b_2 = ncon([f, eta2B], [(-1, 0), (0, -2, -3, -4)]).numpy()
        sum1_2b_3 = sum1_2b_1 - sum1_2b_2
        sum1_2b_4 = np.transpose(sum1_2b_3, [1, 0, 2, 3])
        sum1_2b_5 = sum1_2b_3 - sum1_2b_4

        # first term (P_kl piece)
        sum1_2b_6 = ncon([eta1B, G], [(0, -3), (-1, -2, 0, -4)]).numpy()
        sum1_2b_7 = ncon([f, eta2B], [(0, -3), (-1, -2, 0, -4)]).numpy()
        sum1_2b_8 = sum1_2b_6 - sum1_2b_7
        sum1_2b_9 = np.transpose(sum1_2b_8, [0, 1, 3, 2])
        sum1_2b_10 = sum1_2b_8 - sum1_2b_9

        sum1_2b = sum1_2b_5 - sum1_2b_10

        # second term
        sum2_2b_1 = ncon([occB,     G], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
        sum2_2b_2 = ncon([occB, eta2B], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
        sum2_2b_3 = ncon([eta2B,  sum2_2b_1], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
        sum2_2b_4 = ncon([G,      sum2_2b_2], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
        sum2_2b = sum2_2b_3 - sum2_2b_4

        # third term
        sum3_2b_1 = ncon([eta2B, G], [(0, -1, 1, -3), (1, -2, 0, -4)]).numpy()
        sum3_2b_2 = np.transpose(sum3_2b_1, [1, 0, 2, 3])
        sum3_2b_3 = np.transpose(sum3_2b_1, [0, 1, 3, 2])
        sum3_2b_4 = np.transpose(sum3_2b_1, [1, 0, 3, 2])
        sum3_2b_5 = sum3_2b_1 - sum3_2b_2 - sum3_2b_3 + sum3_2b_4
        sum3_2b = ncon([occA, sum3_2b_5], [(0, 1, -1, -2), (0, 1, -3, -4)]).numpy()

        dG = sum1_2b + 0.5*sum2_2b + sum3_2b

        return (dE, df, dG)

class Flow_IMSRG3(Flow_IMSRG2):

    def __init__(self, gen):
        super().__init__(gen)
