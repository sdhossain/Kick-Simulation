from sre_constants import IN
import sys
sys.path.append('.')
sys.path.append('./muscle_modelling')

import numpy as np

from gravity_moment import gravity_moment
from muscle_length import quad_muscle_length
from muscle_modelling.force_length import force_length_tendon
from muscle_modelling.get_velocity import get_velocity

KNEE_INERTIA = 90
QUAD_MOMENT_ARM = 0.05
FEMORIS_MAX_FORCE = 10000
LATERALIS_MAX_FORCE = 10000
MEDIALIS_MAX_FORCE = 10000
INTERMEDIUS_MAX_FORCE = 10000

def dynamics(x, thigh_offset, femoris, lateralis, medialis, 
             intermedius, femoris_activation, lateralis_activation,
             medialis_activation, intermedius_activation,
             force_length_regression, force_velocity_regression,
             ):
    """
    Computes time-derivative of state-vector based on model

    :param x: state vector (ankle angle, 
                            angular velocity, 
                            femoris normalized CE length, 
                            lateralis normalized CE length,
                            medialis normalized CE length,
                            intermedius normalized CE length)
    :param thigh_offset: initial offset angle of thigh in radians
    :param femoris: femoris HillTypeModel object
    :param lateralis: lateralis HillTypeModel object
    :param medialis: medialis HillTypeModel object
    :param intermedius: intermedius HillTypeModel object
    :param femoris_activation: femoris muscle activation between (0, 1)
    :param lateralis_activation: lateralis muscle activation between (0, 1)
    :param medialis_activation: medialis muscle activation between (0, 1)
    :param intermedius_activation: intermedius muscle activation between (0, 1)
    :param force_length_regression: regression function for force length
    :param force_velocity_regression: regression function for for velocity
    :return x_dot: time-derivate of state vector
    """

    '''
    if x[0] > np.pi:
        print("too high - boundary reached")
        return x

    if x[0] < 0:
        print("too low - boundary reached")
        return x
    '''

    # we first obtain the normalized tendon lengths for both muscles
    # they depend on normalized muscle length and musculotendon length

    lt_femoris = femoris.norm_tendon_length(
            quad_muscle_length(x[0], thigh_offset), x[2])
    lt_lateralis = lateralis.norm_tendon_length(
            quad_muscle_length(x[0], thigh_offset), x[3])
    lt_medialis = medialis.norm_tendon_length(
            quad_muscle_length(x[0], thigh_offset), x[4])
    lt_intermedius = intermedius.norm_tendon_length(
            quad_muscle_length(x[0], thigh_offset), x[5])

    # we calculate torque caused by each muscle as well as gravity
    # note that f_ext is 0 so that term is not present (0'd out)

    torque_femoris = FEMORIS_MAX_FORCE*force_length_tendon(lt_femoris)*QUAD_MOMENT_ARM
    torque_lateralis = LATERALIS_MAX_FORCE*force_length_tendon(lt_lateralis)*QUAD_MOMENT_ARM
    torque_medialis = MEDIALIS_MAX_FORCE*force_length_tendon(lt_medialis)*QUAD_MOMENT_ARM
    torque_intermedius = INTERMEDIUS_MAX_FORCE*force_length_tendon(lt_intermedius)*QUAD_MOMENT_ARM

    torque_quad = torque_femoris + torque_lateralis + torque_medialis + torque_intermedius
    torque_gravity = gravity_moment(x[0], thigh_offset)

    # once we have torques, x_dot[1] is easy to calculate and the rest are
    # implemented in accordance to what is present in the lectures

    x_dot = np.array([
        x[1],
        ((torque_quad + torque_gravity)/KNEE_INERTIA)[0],
        get_velocity(femoris_activation, x[2], lt_femoris, 
                    force_length_regression, force_velocity_regression)[0],
        get_velocity(lateralis_activation, x[3], lt_lateralis, 
                    force_length_regression, force_velocity_regression)[0],
        get_velocity(medialis_activation, x[4], lt_medialis, 
                    force_length_regression, force_velocity_regression)[0],
        get_velocity(intermedius_activation, x[5], lt_intermedius, 
                    force_length_regression, force_velocity_regression)[0],
    ])

    return x_dot