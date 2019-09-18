#################################################
# Hamiltonian Creator
# Julie Hartley 
# Adapted from code by Jacob Davison
# August 5, 2019
# Version 1.0.0
#
# Creates the Hamiltonian for the pairing model for 
# a basis of arbitrary size given n particles and n 
# holes.  There will also be n energy levels (n 
# must be an even number).
#################################################

#################################################
# OUTLINE
# remove_duplicates (lst): (helper fuction) Returns a 
# list containing only one copy of each element in 
# lst (removes duplicate elements).
# 
# create_basis (n_holes, n_particles): Creates the 
# basis for the pairing model Hamiltonian.
# 
# create_element(state1, state2, n_tot, d, g): Creates 
# an element of the pairing model Hamiltonian.
# 
# create_hamiltonian (n, d, g): Creates the Hamiltonian 
# for the pairing model.
#################################################

#############################
# IMPORTS
#############################
# SYSTEM IMPORTS
# For error handling
from sys import exit
# THIRD-PARTY IMPORTS
# For arrays and calculations
import numpy as np
# Needed for basis creation
from itertools import permutations 

#############################
# METHODS
#############################
# REMOVE_DUPLICATES
def remove_duplicates(lst):
    """
        Inputs:
            lst (a list or Numpy array)
        Returns:
            final_list (a list): a copy of lst but with all the duplicate elements 
                removed
        Returns a list containing only one copy of each element in lst (removes 
        duplicate elements).
    """ 
    final_list = [] 
    for l in lst: 
        # if the element is not already in final_list
        # i.e. it is a new element
        if l not in final_list: 
            final_list.append(l) 
    return final_list 

# CREATE_BASIS
def create_basis (n_holes, n_particles):
    """
        Inputs:
            n_holes (an int): the number of holes in the pairing model
            n_particles (an int): the number of particles in the pairing model
            NOTE: n_holes must equal n_particles
        Returns:
            basis (a numpy array): the basis for the Hamiltonian
        Creates the basis for the pairing model Hamiltonian.
    """
    # Handle user errors of inputs
    if n_holes != n_particles:
        print ("Number of holes not same as number of particles.")
        print ("Ending execution.")
        exit()

    initial_basis = []

    # Need the number of pairs of particles (which is also the number of pairs of holes)
    # Pairs cannot be broken
    n_pairs = int(n_holes/2)

    # 1 represents an occupied state (particle), 0 an unoccupied state (hole)
    # Need to make sure the particles and holes are grouped into unbreakable pairs
    for i in range (n_pairs):
        initial_basis.append([1,1])
    for i in range (n_pairs):
        initial_basis.append([0,0])

    # Find all possible permutations of the initial basis
    # Note: because of the grouping done above, the pairs are not broken
    basis = list(permutations(initial_basis))

    # Remove any duplicate permutations
    basis = remove_duplicates(basis)

    # Remove the pair structure (no longer needed)
    for i in range(len(basis)):
        print(i)
        basis[i] = np.reshape(basis[i], -1)

    return basis

# CREATE_ELEMENT
def create_element(state1, state2, n_tot, d, g):
    """
        Inputs:
            state1, state2 (lists): rows of the basis corresponding to the
                needed element of the Hamiltonian 
            n_tot (an int): the total number of particles and holes
            d (a float): the energy level spacing
            g (a float): interaction coefficient
        Returns:
            h (a float): the element of the Hamiltonian at the desired location
        Creates an element of the pairing model Hamiltonian.
    """
    # one body part (non-zero only for the diagonal of the Hamiltonian)
    h0 = 0
    if np.array_equal(state1,state2):
        for i in range(0,n_tot):
            p = int(i/2) + 1
            if(state2[i] == 1):
                h0 += d*(p-1)
                
    # two body part
    h1 = -0.25*g*np.inner(state1, state2)
    
    # element is the sum of the one body part and the two body part
    h = h0+h1

    return h

# CREATE_HAMILTONIAN
def create_hamiltonian (n, d, g):
    """
        Inputs:
            n (an even int): the number of particles/holes
            d (a float): Energy level spacing
            g (a float): Interaction coefficient
        Returns:
            H (a 2D numpy array): the pairing model Hamiltonian        
        Creates the Hamiltonian for the pairing model.
    """
    if n%2 != 0:
        print (" There must be an even number of particles and holes.")
        print ("Ending execution.")
        exit()

    # Variable initialization
    n_holes = n
    n_particles = n
    n_tot = n*2

    # Create the basis
    basis = create_basis (n_holes, n_particles)
    
    # Skeleton Hamiltonian
    l = len(basis)
    H = np.zeros((l,l))

    # Get the value for each element of the Hamiltonian
    for i in range(0,l):
        for j in range(0,l):
            print (i,j)
            H[i,j] = create_element(basis[i],basis[j], n_tot, d, g)

    return H

# Create Initial Hamiltonians n=2-20
H = create_hamiltonian(4, 1.0, 0.5)
save_name = 'PairingModel_' + str(4)
np.save(save_name, H)
del H

