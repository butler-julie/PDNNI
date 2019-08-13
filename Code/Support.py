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
