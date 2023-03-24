import sys
sys.path.append('.')
sys.path.append('./muscle_modelling')

import numpy as np

SHANK_MASS = 18
COM_DIST = 0.6

def gravity_moment(theta):
    """
    Calculate moment of gravity based on theta

    :param theta: andgle in radians
    :return moment: moment caused by gravity
    """

    mass = SHANK_MASS
    centre_of_mass_distance = COM_DIST
    g = 9.81  # acceleration of gravity
    moment = mass * g * centre_of_mass_distance * np.cos(theta)

    return moment