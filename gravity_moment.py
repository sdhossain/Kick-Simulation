import sys
sys.path.append('.')
sys.path.append('./muscle_modelling')

import numpy as np

SHANK_MASS = 4.33
COM_DIST = 0.22
PHI = (3*np.pi)/2

def gravity_moment(theta, thigh_offset):
    """
    Calculate moment of gravity based on theta

    :param theta: andgle in radians
    :param thigh_offset: initial offset angle of thigh in radians
    :return moment: moment caused by gravity
    """

    mass = SHANK_MASS
    centre_of_mass_distance = COM_DIST
    g = 9.81  # acceleration of gravity
    moment = mass * g * centre_of_mass_distance * np.cos((theta + thigh_offset) - np.pi/2)

    return moment
