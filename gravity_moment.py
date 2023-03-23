import numpy as np

def gravity_moment(theta):
    """
    Calculate moment of gravity based on theta

    :param theta: andgle in radians
    :return moment: moment caused by gravity
    """

    mass = 75  # body mass (kg; excluding feet)
    centre_of_mass_distance = 1  # distance from ankle to body segment centre of mass (m)
    g = 9.81  # acceleration of gravity
    moment = mass * g * centre_of_mass_distance * np.cos(theta)

    return moment