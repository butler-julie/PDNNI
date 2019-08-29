##################################################
# Support
# Julie Hartley
# Version 1.0.0
# August 13, 2019
#
# Contains support methods that are or could be needed
# by other files in PDNNI
##################################################

##################################################
# OUTLINE
#
# print_dims(x): Prints the dimensions of a 2D square array.
#
# get_dims(x): Returns the dimensions of a 2D square array.
##################################################

import numpy as np
from numpy.random import randint
from math import floor

def generate_subset (input1_name, input2_name, percentage, output1_name, output2_name):
    input1 = np.load(input1_name)
    input2 = np.load(input2_name)

    L = len(input1)

    num_points = floor(L * percentage)

    rand = randint(low = 1, high = L-1, size = num_points)
    rand = np.append(rand, 0)
    rand = np.append(rand, L-1)

    output1 = input1[rand]
    output2 = input2[rand]

    np.save(output1_name, output1)
    np.save(output2_name, output2)

    	

# PRINT_DIMS
def print_dims (x):
    """
        Inputs:
            x: a 2D square array or list
        Prints the dimensions of a 2D square array.
    """
    print(len(x), len(x[0]))

# GET_DIMS
def get_dims(x):
    """
        Inputs:
            x: a 2D square array or list
        Returns the dimensions of a 2D square array.
    """
    return len(x), len(x[0])
