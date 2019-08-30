#############################
#                           #
# IMPORTS                   #
#                           #
#############################
# THIRD-PARTY IMPORTS
# For calculations
import numpy as np
# For graphing
#import matplotlib.pyplot as plt
#from pylab import *
# For SRG and matrix manipulation
from numpy import array, dot, diag, reshape
from scipy.linalg import eigvalsh
from scipy.integrate import odeint, ode
# Creating random numbers (for file naming)
import random
import time



# SYSTEM IMPORTS
import time, os
dir = os.path.dirname (os.path.realpath (__file__))

# COMMUTATOR
def commutator(a,b):
    """
        Inputs:
            a, b (2D-arrays): the matrices to take the commutator of (order dependent)
        Returns:
            Unnamed: the commutator of a and b
        Returns the commutator of matrices a and b
        [a,b] = ab - ba
        The order of the arguments is important
    """
    return dot(a,b) - dot(b,a)

# SRG_FLOW_EQUATION
def srg_flow_equation(s, y, dim):
    """
        Inputs:
            y (an array):  the current value of the Hamiltonian
            s (an array): the flow parameters values at which the SRG flow equation is
                to be solved at
            dim (an int): the dimension of one side of the square SRG matrix
        Returns:
            Unnamed (): the results from solving the SRG flow equation
        Solves the SRG flow equation at given values of the flow parameter.
        Taken from the code srg_pairing.py by H. Hergert
    """

    # reshape the solution vector into a dim x dim matrix
    H = reshape(y, (DIMENSION, DIMENSION))

    # extract diagonal Hamiltonian...
    Hd  = diag(diag(H))

    # ... and construct off-diagonal the Hamiltonian
    Hod = H-Hd

    # calculate the generator
    eta = commutator(Hd, Hod)

    # dH is the derivative in matrix form 
    dH  = commutator(eta, H)

    # convert dH into a linear array for the ODE solver
    dydt = reshape(dH, -1)
    
    return dydt

def ode1_1 (initial_y, s_end, ds): # NOTE: arguments have to be switched in flow eq.
    print (s_end, ds)
    f = ode(srg_flow_equation, jac=None)
    f.set_integrator('vode', method='bdf', order=5, nsteps=50000)
    f.set_initial_value(initial_y, 0)
    f.set_f_params(DIMENSION)
    t1 = s_end + ds
    dt = ds
    ys = []
    while f.successful() and f.t < t1:
        ys.append(f.integrate (f.t + dt))

    # reshape individual solution vectors into dim x dim Hamiltonian
    # matrices
    #ys  = reshape(ys, (-1, DIMENSION,DIMENSION))

    return ys 

    
def srg (initial_hamiltonian, ds, sfinal, file_out):
    """
    """
    # Pairing model set-up code taken from srg_pairing.py by H. Hergert
    print ("Importing Hamiltonian")
    # The intial Hamiltonian
    H0    = np.load(initial_hamiltonian)
    #print (H0)
    dim   = H0.shape[0]

    # turn initial Hamiltonian into a linear array
    y0  = reshape(H0, -1)                 

    # Create the training data

    print ("SRG Flow")
    start= time.time()
    Hs = ode1_1 (y0, sfinal, ds)
    print ('\nTime', time.time()-start, '\n' )

    np.save(file_out, Hs)
