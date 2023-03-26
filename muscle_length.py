import sys
sys.path.append('.')
sys.path.append('./muscle_modelling')

import numpy as np

QUAD_SHANK_INSERTION = (0.1, 0.01)
KNEE_ORIGIN = (0, 0)
THIGH_LENGTH = 0.5
PHI = (3*np.pi)/2

def quad_muscle_length(theta, thigh_offset):
    """
    Calculates quadricp muscle length based on angle
    :param theta: angle between shank and thigh
    :param thigh_offset: initial offset angle of thigh in radians
    :return quad_muscle: length of muscle
    """

    # obtain angle by which co-ordinate axis shifts
    gamma = (theta + thigh_offset) - PHI

    # define rotation matrix
    rotation = np.array([[np.cos(gamma), -np.sin(gamma)], 
                          [np.sin(gamma), np.cos(gamma)]])

    # coordinates in global reference frame
    origin = np.dot(rotation, np.array([QUAD_SHANK_INSERTION[0], QUAD_SHANK_INSERTION[1]]).reshape((2,1)))
    insertion = np.array([KNEE_ORIGIN[0], KNEE_ORIGIN[1]]).reshape((2,1))

    difference = origin - insertion
    quad_muscle_length = np.sqrt(difference[0]**2 + difference[1]**2)

    return quad_muscle_length[0]