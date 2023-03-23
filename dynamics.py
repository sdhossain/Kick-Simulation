import numpy as np

from muscle_length import soleus_length, tibialis_length
from force_length import force_length_tendon
from get_velocity import get_velocity

def dynamics(x, soleus, tibialis, force_length_regression, force_velocity_regression):
    """
    Computes time-derivative of state-vector based on model

    :param x: state vector (ankle angle, angular velocity, 
                            soleus normalized CE length, 
                            TA normalized CE length)
    :param soleus: soleus HillTypeModel object
    :param tibialis: tibialis HillTypeModel object
    :param force_length_regression: regression function for force length
    :param force_velocity_regression: regression function for for velocity
    :return x_dot: time-derivate of state vector
    """

    soleus_activation = 0.05
    tibialis_activation = 0.4
    ankle_inertia = 90

    # we first obtain the normalized tendon lengths for both muscles
    # they depend on normalized muscle length and musculotendon length

    lt_sol = soleus.norm_tendon_length(soleus_length(x[0]), x[2])
    lt_ta = tibialis.norm_tendon_length(tibialis_length(x[0]), x[3])

    # we calculate torque caused by each muscle as well as gravity
    # note that f_ext is 0 so that term is not present (0'd out)

    torque_sol = 16000*force_length_tendon(lt_sol)*0.05
    torque_ta = 2000*force_length_tendon(lt_ta)*0.03
    torque_mg = 75*9.81*1*np.cos(x[0])

    # once we have torques, x_dot[1] is easy to calculate and the rest are
    # implemented in accordance to what is present in the lectures

    x_dot = [0, 0, 0, 0]
    x_dot[0] = x[1]
    x_dot[1] = (torque_sol-torque_ta-torque_mg)/ankle_inertia
    x_dot[2] = get_velocity(soleus_activation, x[2], lt_sol, 
                    force_length_regression, force_velocity_regression)
    x_dot[3] = get_velocity(tibialis_activation, x[3], 
                    lt_ta, force_length_regression, force_velocity_regression)

    return x_dot