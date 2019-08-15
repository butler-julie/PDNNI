import numpy as np

class OccupationTensors(object):
    """Functions as a container for important occupation tensors
    defined by the reference state (usually, ground state) of
    the Hamiltonian. The occupation basis is a vector n that
    contains a 1 if the corresponding single particle basis
    index is occupied, or 0 if the same index is unoccupied,
    in the reference state."""

    def __init__(self, sp_basis, reference):
        """Class constructor. Instantiates an OccupationTensors object.

        Arugments:

        sp_basis -- list containing indices of single particle basis
        reference -- contains reference state as string of 1/0's"""

        self._reference = reference
        self._sp_basis = sp_basis

        self._occA = self.__get_occA()
        self._occB = self.__get_occB()
        self._occC = self.__get_occC()
        self._occD = self.__get_occD(flag=1)

    @property
    def occA(self):
        """Returns: 

        occA -- represents n_a - n_b."""
        return self._occA
    
    @property
    def occB(self):
        """Returns: 

        occB -- represents 1 - n_a - n_b."""
        return self._occB
        
    @property
    def occC(self):
        """Returns:

        occC -- represents n_a*n_b + (1-n_a-n_b)*n_c"""
        return self._occC
    
    @property
    def occD(self):
        """Returns:

        occD -- represents na*nb*(1-nc-nd) + na*nb*nc*nd"""
        return self._occD
    
    def __get_occA(self, flag=0):
        """Builds the occupation tensor occA.

        Keyword arguments:

        flag -- toggle rank 4 or rank 2 tensor behavior (default: 0)

        Returns:

        occA -- n_a - n_b"""

        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        if flag == 0: # default
            occA = np.zeros((n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    occA[a,b,a,b] = ref[a] - ref[b]
        
        if flag == 1:
            occA = np.zeros((n,n))

            for a in bas1B:
                for b in bas1B:
                    occA[a,b] = ref[a] - ref[b]

        return occA    
        
    def __get_occB(self, flag=0):
        """Builds the occupation tensor occB.

        Keyword arguments:

        flag -- toggle rank 4 or rank 2 tensor behavior (default: 0)

        Returns:

        occB -- 1 - n_a - n_b"""

        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        if flag == 0: # default
            occB = np.zeros((n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    occB[a,b,a,b] = 1 - ref[a] - ref[b]
        
        if flag == 1:
            occB = np.zeros((n,n))

            for a in bas1B:
                for b in bas1B:
                    occB[a,b] = 1 - ref[a] - ref[b]

        return occB        
            
    def __get_occC(self, flag=0):
        """Builds the occupation tensor occC.

        Keyword arguments:

        flag -- toggle rank 6 or rank 3 tensor behavior (default: 0)

        Returns:

        occC -- n_a*n_b + (1-n_a-n_b)*n_c"""

        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        if flag == 0: # default
            occC = np.zeros((n,n,n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    for c in bas1B:
                        occC[a,b,c,a,b,c] = ref[a]*ref[b] + (1-ref[a]-ref[b])*ref[c]

        
        if flag == 1: 
            occC = np.zeros((n,n,n))

            for a in bas1B:
                for b in bas1B:
                    for c in bas1B:
                        occC[a,b,c] = ref[a]*ref[b] + (1-ref[a]-ref[b])*ref[c]
        return occC

    def __get_occD(self, flag=0):
        """Builds the occupation tensor occD.

            Keyword arguments:

            flag -- toggle rank 8 or rank 4 tensor behavior (default: 0)

            Returns:

            occD -- na*nb*(1-nc-nd) + na*nb*nc*nd"""

        bas1B = self._sp_basis
        ref = self._reference
        n = len(bas1B)

        if flag == 0: # default 
            occD = np.zeros((n,n,n,n,n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    for c in bas1B:
                        for d in bas1B:
                            occD[a,b,c,d,a,b,c,d] = ref[a]*ref[b]*(1-ref[c]-ref[d])+ref[a]*ref[b]*ref[c]*ref[d]
        
        if flag == 1:
            occD = np.zeros((n,n,n,n))

            for a in bas1B:
                for b in bas1B:
                    for c in bas1B:
                        for d in bas1B:
                            occD[a,b,c,d] = ref[a]*ref[b]*(1-ref[c]-ref[d])+ref[a]*ref[b]*ref[c]*ref[d]

        return occD    