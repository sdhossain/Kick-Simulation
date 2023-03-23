import numpy as np

def soleus_length(theta):
    """
    Calculates soleus length based on angle
    :param theta: body angle (up from prone horizontal)
    :return soleus: length of muscle
    """

    # define rotation matrix
    rotation = np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta), np.cos(theta)]])
    
    # coordinates in global reference frame
    origin = np.dot(rotation, np.array([0.3, 0.03]).T)
    insertion = np.array([-0.05, -0.02])
    
    difference = origin - insertion
    soleus_length = np.sqrt(difference[0]**2 + difference[1]**2)
    
    return soleus_length

def tibialis_length(theta):
    """
    Calculates soleus length based on angle
    :param theta: body angle (up from prone horizontal)
    :return tibialis: length of muscle
    """

    # define rotation matrix
    rotation = np.array([[np.cos(theta), -np.sin(theta)], 
                          [np.sin(theta), np.cos(theta)]])

    # coordinates in global reference frame
    origin = np.dot(rotation, np.array([0.3, -0.03]).reshape((2,1)))
    insertion = np.array([0.06, -0.03]).reshape((2,1))

    difference = origin - insertion
    tibialis_anterior_length = np.sqrt(difference[0]**2 + difference[1]**2)

    return tibialis_anterior_length[0]