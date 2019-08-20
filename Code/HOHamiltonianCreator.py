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
from sys import argv



# SYSTEM IMPORTS
import time, os
dir = os.path.dirname (os.path.realpath (__file__))


DIMENSION = int(argv[1])

#############################
#                           #
# FUNCTIONS                 #
#                           #
#############################
#Function for initialization of parameters
def initialize():
    RMin = 0.0
    RMax = 10.0
    lOrbital = 0
    Dim = DIMENSION
    return RMin, RMax, lOrbital, Dim
# Here we set up the harmonic oscillator potential
def potential(r):
    return r*r
    #return 0


# HAMILTONIAN
def Hamiltonian():
    #Get the boundary, orbital momentum and number of integration points
    RMin, RMax, lOrbital, Dim = initialize()

    #Initialize constants
    Step    = RMax/(Dim+1)
    DiagConst = 2.0 / (Step*Step)
    NondiagConst =  -1.0 / (Step*Step)
    OrbitalFactor = lOrbital * (lOrbital + 1.0)

    #Calculate array of potential values
    v = np.zeros(Dim)
    r = np.linspace(RMin,RMax,Dim)
    for i in range(Dim):
        r[i] = RMin + (i+1) * Step;
        v[i] = potential(r[i]) + OrbitalFactor/(r[i]*r[i]);

    #Setting up a tridiagonal matrix and finding eigenvectors and eigenvalues
    Hamiltonian = np.zeros((Dim,Dim))
    Hamiltonian[0,0] = DiagConst + v[0];
    Hamiltonian[0,1] = NondiagConst;
    for i in range(1,Dim-1):
        Hamiltonian[i,i-1]  = NondiagConst;
        Hamiltonian[i,i]    = DiagConst + v[i];
        Hamiltonian[i,i+1]  = NondiagConst;
    Hamiltonian[Dim-1,Dim-2] = NondiagConst;
    Hamiltonian[Dim-1,Dim-1] = DiagConst + v[Dim-1];

    return Hamiltonian
    
H0 = Hamiltonian()
name = 'HO' + str(DIMENSION) + '.npy'
np.save(name, H0)
