from tensornetwork import *
import numpy as np
from oop_imsrg.hamiltonian import *
from oop_imsrg.occupation_tensors import *

class Generator(object):
    """Parent class for organization purposes. Ideally, all Generator
    classes should inherit from this class. In this way, AssertionErrors
    can be handled in a general way."""
    def calc_eta():
        print("Function that calculates the generator")

class WegnerGenerator(Generator):
    """Calculate Wegner's generator for a normal ordered Hamiltonian."""

    def __init__(self, h, occ_t):
        """Class constructor. Instantiate WegnerGenerator object.

        Arguments:

        h -- Hamiltonian object (must be normal-ordered)
        occ_t -- OccupationTensor object"""

        assert isinstance(h, Hamiltonian), "Arg 0 must be Hamiltonian object"
        assert isinstance(occ_t, OccupationTensors), "Arg 1 must be OccupationTensors object"

        self.f = h.f
        self.G = h.G

        self._holes = h.holes
        self._particles = h.particles

        self._occA = occ_t.occA
        self._occB = occ_t.occB
        self._occC = occ_t.occC
        self._occD = occ_t.occD

    @property
    def f(self):
        """Returns:

        f -- one-body tensor elements (initialized by Hamiltonian object)"""
        return self._f

    @property
    def G(self):
        """Returns:

        f -- two-body tensor elements (initialized by Hamiltonian object)"""
        return self._G

    @f.setter
    def f(self, f):
        """Sets the one-body tensor."""
        self._f = f

    @G.setter
    def G(self, G):
        """Sets the two-body tensor."""
        self._G = G

    def __decouple_OD(self):
        """Decouple the off-/diagonal elements from each other in
        the one- and two-body tensors. This procedure is outlined in
        An Advanced Course in Computation Nuclear Physics, Ch.10.

        Returns:

        (fd, -- diagonal part of f
         fod, -- off-diagonal part of f
         Gd, -- diagonal part of G
         God) -- off-diagonal part of G"""

        f = self.f
        G = self.G
        holes = self._holes
        particles = self._particles

        # - Decouple off-diagonal 1B and 2B pieces
        fod = np.zeros(f.shape)
        fod[np.ix_(particles, holes)] += f[np.ix_(particles, holes)]
        fod[np.ix_(holes, particles)] += f[np.ix_(holes, particles)]
        fd = f - fod

        God = np.zeros(G.shape)
        God[np.ix_(particles, particles, holes, holes)] += G[np.ix_(particles, particles, holes, holes)]
        God[np.ix_(holes, holes, particles, particles)] += G[np.ix_(holes, holes, particles, particles)]
        Gd = G - God

        return (fd, fod, Gd, God)

    def calc_eta(self):
        """Calculate the generator. The terms are defined in An
        Advanced Course in Computation Nuclear Physics, Ch.10.
        See also dx.doi.org/10.1016/j.physrep.2015.12.007

        Returns:

        (eta1B, -- one-body generator
         eta2B) -- two-body generator"""

        fd, fod, Gd, God = self.__decouple_OD()

        holes = self._holes
        particles = self._particles

        occA = self._occA
        occB = self._occB
        occC = self._occC

        # - Calculate 1B generator
        # first term
        sum1_1b_1 = ncon([fd, fod], [(-1, 0), (0, -2)]).numpy()
        sum1_1b_2 = np.transpose(sum1_1b_1)
        sum1_1b = sum1_1b_1 - sum1_1b_2

        # second term
        sum2_1b_1 = ncon([fd, God], [(0, 1), (1, -1, 0, -2)]).numpy()
        sum2_1b_2 = ncon([fod, Gd], [(0, 1), (1, -1, 0, -2)]).numpy()
        sum2_1b_3 = sum2_1b_1 - sum2_1b_2
        sum2_1b = ncon([occA, sum2_1b_3],[(-1, -2, 0, 1), (0,1)]).numpy()

        # third term
        sum3_1b_1 = ncon([occC, God], [(-1, -2, -3, 0, 1, 2), (0, 1, 2, -4)]).numpy()
        sum3_1b_2 = ncon([Gd, sum3_1b_1], [(2, -1, 0, 1), (0, 1, 2, -2)]).numpy()
        sum3_1b_3 = np.transpose(sum3_1b_2)
        sum3_1b = sum3_1b_2 - sum3_1b_3

        eta1B = sum1_1b + sum2_1b + 0.5*sum3_1b

        # - Calculate 2B generator
        # first term (P_ij piece)
        sum1_2b_1 = ncon([fd, God], [(-1, 0), (0, -2, -3, -4)]).numpy()
        sum1_2b_2 = ncon([fod, Gd], [(-1, 0), (0, -2, -3, -4)]).numpy()
        sum1_2b_3 = sum1_2b_1 - sum1_2b_2
        sum1_2b_4 = np.transpose(sum1_2b_3, [1, 0, 2, 3])
        sum1_2b_5 = sum1_2b_3 - sum1_2b_4

        # first term (P_kl piece)
        sum1_2b_6 = ncon([fd, God], [(0, -3), (-1, -2, 0, -4)]).numpy()
        sum1_2b_7 = ncon([fod, Gd], [(0, -3), (-1, -2, 0, -4)]).numpy()
        sum1_2b_8 = sum1_2b_6 - sum1_2b_7
        sum1_2b_9 = np.transpose(sum1_2b_8, [0, 1, 3, 2])
        sum1_2b_10 = sum1_2b_8 - sum1_2b_9

        sum1_2b = sum1_2b_5 - sum1_2b_10

        # second term
        sum2_2b_1 = ncon([occB, God], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
        sum2_2b_2 = ncon([occB,  Gd], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
        sum2_2b_3 = ncon([Gd,  sum2_2b_1], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
        sum2_2b_4 = ncon([God, sum2_2b_2], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
        sum2_2b = sum2_2b_3 - sum2_2b_4

        # third term
        sum3_2b_1 = ncon([Gd, God], [(0, -1, 1, -3), (1, -2, 0, -4)]).numpy()
        sum3_2b_2 = np.transpose(sum3_2b_1, [1, 0, 2, 3])
        sum3_2b_3 = np.transpose(sum3_2b_1, [0, 1, 3, 2])
        sum3_2b_4 = np.transpose(sum3_2b_1, [1, 0, 3, 2])
        sum3_2b_5 = sum3_2b_1 - sum3_2b_2 - sum3_2b_3 + sum3_2b_4
        sum3_2b = ncon([occA, sum3_2b_5], [(0, 1, -1, -2), (0, 1, -3, -4)]).numpy()

        eta2B = sum1_2b + 0.5*sum2_2b + sum3_2b

        return (eta1B, eta2B)


    # def calc_eta1B(self):
    #     fd, fod, Gd, God = self.__decouple_OD()

    #     holes = self._holes
    #     particles = self._particles

    #     occA = self._occA
    #     occC = self._occC

    #     # - Calculate 1B generator
    #     # first term
    #     sum1_1b_1 = ncon([fd, fod], [(-1, 0), (0, -2)]).numpy()
    #     sum1_1b_2 = np.transpose(sum1_1b_1)
    #     sum1_1b = sum1_1b_1 - sum1_1b_2

    #     # second term
    #     sum2_1b_1 = ncon([fd, God], [(0, 1), (1, -1, 0, -2)]).numpy()
    #     sum2_1b_2 = ncon([fod, Gd], [(0, 1), (1, -1, 0, -2)]).numpy()
    #     sum2_1b_3 = sum2_1b_1 - sum2_1b_2
    #     sum2_1b = ncon([occA, sum2_1b_3],[(-1, -2, 0, 1), (0,1)]).numpy()

    #     # third term
    #     sum3_1b_1 = ncon([occC, God], [(-1, -2, -3, 0, 1, 2), (0, 1, 2, -4)]).numpy()
    #     sum3_1b_2 = ncon([Gd, sum3_1b_1], [(2, -1, 0, 1), (0, 1, 2, -2)]).numpy()
    #     sum3_1b_3 = np.transpose(sum3_1b_2)
    #     sum3_1b = sum3_1b_2 - sum3_1b_3

    #     eta1B = sum1_1b + sum2_1b + 0.5*sum3_1b

    #     return eta1B

    # def calc_eta2B(self):
    #     fd, fod, Gd, God = self.__decouple_OD()

    #     holes = self._holes
    #     particles = self._particles

    #     occA = self._occA
    #     occB = self._occB

    #     # - Calculate 2B generator
    #     # first term (P_ij piece)
    #     sum1_2b_1 = ncon([fd, God], [(-1, 0), (0, -2, -3, -4)]).numpy()
    #     sum1_2b_2 = ncon([fod, Gd], [(-1, 0), (0, -2, -3, -4)]).numpy()
    #     sum1_2b_3 = sum1_2b_1 - sum1_2b_2
    #     sum1_2b_4 = np.transpose(sum1_2b_3, [1, 0, 2, 3])
    #     sum1_2b_5 = sum1_2b_3 - sum1_2b_4

    #     # first term (P_kl piece)
    #     sum1_2b_6 = ncon([fd, God], [(0, -3), (-1, -2, 0, -4)]).numpy()
    #     sum1_2b_7 = ncon([fod, Gd], [(0, -3), (-1, -2, 0, -4)]).numpy()
    #     sum1_2b_8 = sum1_2b_6 - sum1_2b_7
    #     sum1_2b_9 = np.transpose(sum1_2b_8, [0, 1, 3, 2])
    #     sum1_2b_10 = sum1_2b_8 - sum1_2b_9

    #     sum1_2b = sum1_2b_5 - sum1_2b_10

    #     # second term
    #     sum2_2b_1 = ncon([occB, God], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
    #     sum2_2b_2 = ncon([occB,  Gd], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
    #     sum2_2b_3 = ncon([Gd,  sum2_2b_1], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
    #     sum2_2b_4 = ncon([God, sum2_2b_2], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
    #     sum2_2b = sum2_2b_3 - sum2_2b_4

    #     # third term
    #     sum3_2b_1 = ncon([Gd, God], [(0, -1, 1, -3), (1, -2, 0, -4)]).numpy()
    #     sum3_2b_2 = np.transpose(sum3_2b_1, [1, 0, 2, 3])
    #     sum3_2b_3 = np.transpose(sum3_2b_1, [0, 1, 3, 2])
    #     sum3_2b_4 = np.transpose(sum3_2b_1, [1, 0, 3, 2])
    #     sum3_2b_5 = sum3_2b_1 - sum3_2b_2 - sum3_2b_3 + sum3_2b_4
    #     sum3_2b = ncon([occA, sum3_2b_5], [(0, 1, -1, -2), (0, 1, -3, -4)]).numpy()

    #     eta2B = sum1_2b + 0.5*sum2_2b + sum3_2b

    #     return eta2B

    # @property
    # def eta1B(self):
    #     return self._eta1B

    # @property
    # def eta2B(self):
    #     return self._eta2B

    # @eta1B.setter
    # def eta1B(self, occA):
    #     fd = self._fd
    #     Gd = self._Gd
    #     fod = self._fod
    #     God = self._God

    #     holes = self._holes
    #     particles = self._particles

    #     occA = self._occA
    #     occC = self._occC

    #     # - Calculate 1B generator
    #     # first term
    #     sum1_1b_1 = ncon([fd, fod], [(-1, 0), (0, -2)]).numpy()
    #     sum1_1b_2 = np.transpose(sum1_1b_1)
    #     sum1_1b = sum1_1b_1 - sum1_1b_2

    #     # second term
    #     sum2_1b_1 = ncon([fd, God], [(0, 1), (1, -1, 0, -2)]).numpy()
    #     sum2_1b_2 = ncon([fod, Gd], [(0, 1), (1, -1, 0, -2)]).numpy()
    #     sum2_1b_3 = sum2_1b_1 - sum2_1b_2
    #     sum2_1b = ncon([occA, sum2_1b_3],[(-1, -2, 0, 1), (0,1)]).numpy()

    #     # third term
    #     sum3_1b_1 = ncon([occC, God], [(-1, -2, -3, 0, 1, 2), (0, 1, 2, -4)]).numpy()
    #     sum3_1b_2 = ncon([Gd, sum3_1b_1], [(2, -1, 0, 1), (0, 1, 2, -2)]).numpy()
    #     sum3_1b_3 = np.transpose(sum3_1b_2)
    #     sum3_1b = sum3_1b_2 - sum3_1b_3

    #     eta1B = sum1_1b + sum2_1b + 0.5*sum3_1b

    #     self._eta1B = eta1B

    # @eta2B.setter
    # def eta2B(self, eta2B):
    #     fd = self._fd
    #     Gd = self._Gd
    #     fod = self._fod
    #     God = self._God

    #     holes = self._holes
    #     particles = self._particles

    #     occA = self._occA
    #     occB = self._occB

    #     # - Calculate 2B generator
    #     # first term (P_ij piece)
    #     sum1_2b_1 = ncon([fd, God], [(-1, 0), (0, -2, -3, -4)]).numpy()
    #     sum1_2b_2 = ncon([fod, Gd], [(-1, 0), (0, -2, -3, -4)]).numpy()
    #     sum1_2b_3 = sum1_2b_1 - sum1_2b_2
    #     sum1_2b_4 = np.transpose(sum1_2b_3, [1, 0, 2, 3])
    #     sum1_2b_5 = sum1_2b_3 - sum1_2b_4

    #     # first term (P_kl piece)
    #     sum1_2b_6 = ncon([fd, God], [(0, -3), (-1, -2, 0, -4)]).numpy()
    #     sum1_2b_7 = ncon([fod, Gd], [(0, -3), (-1, -2, 0, -4)]).numpy()
    #     sum1_2b_8 = sum1_2b_6 - sum1_2b_7
    #     sum1_2b_9 = np.transpose(sum1_2b_8, [0, 1, 3, 2])
    #     sum1_2b_10 = sum1_2b_8 - sum1_2b_9

    #     sum1_2b = sum1_2b_5 - sum1_2b_10

    #     # second term
    #     sum2_2b_1 = ncon([occB, God], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
    #     sum2_2b_2 = ncon([occB,  Gd], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
    #     sum2_2b_3 = ncon([Gd,  sum2_2b_1], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
    #     sum2_2b_4 = ncon([God, sum2_2b_2], [(-1, -2, 0, 1), (0, 1, -3, -4)]).numpy()
    #     sum2_2b = sum2_2b_3 - sum2_2b_4

    #     # third term
    #     sum3_2b_1 = ncon([Gd, God], [(0, -1, 1, -3), (1, -2, 0, -4)]).numpy()
    #     sum3_2b_2 = np.transpose(sum3_2b_1, [1, 0, 2, 3])
    #     sum3_2b_3 = np.transpose(sum3_2b_1, [0, 1, 3, 2])
    #     sum3_2b_4 = np.transpose(sum3_2b_1, [1, 0, 3, 2])
    #     sum3_2b_5 = sum3_2b_1 - sum3_2b_2 - sum3_2b_3 + sum3_2b_4
    #     sum3_2b = ncon([occA, sum3_2b_5], [(0, 1, -1, -2), (0, 1, -3, -4)]).numpy()

    #     eta2B = sum1_2b + 0.5*sum2_2b + sum3_2b

    #     self._eta2B = eta2B
