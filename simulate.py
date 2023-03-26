import sys
sys.path.append('.')
sys.path.append('./muscle_modelling')

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from muscle_length import quad_muscle_length
from muscle_modelling.hill_type_muscle import HillTypeMuscle
from dynamics import dynamics

QUAD_REST_ANGLE = math.pi
FEMORIS_MAX_FORCE = 10000
LATERALIS_MAX_FORCE = 10000
MEDIALIS_MAX_FORCE = 10000
INTERMEDIUS_MAX_FORCE = 10000
FEMORIS_TENDON_PERCENT = 0.2451
FEMORIS_MUSCLE_PERCENT = 0.7549
LATERALIS_TENDON_PERCENT = 0.2059
LATERALIS_MUSCLE_PERCENT = 0.7941
MEDIALIS_TENDON_PERCENT = 0.1181
MEDIALIS_MUSCLE_PERCENT = 0.8819
INTERMEDIUS_TENDON_PERCENT = 0.1572
INTERMEDIUS_MUSCLE_PERCENT = 0.8428
SHANK_LENGTH = 0.5048

def simulate(T, initialCondition, thigh_offset, get_femoris_activation, 
                get_lateralis_activation, get_medialis_activation, get_intermedius_activation, 
                force_length_regression, force_velocity_regression):
    """
    Runs a simulation of the model and plots results.

    :param T: total time to simulate, in seconds
    :param initialCondition: initial state of thigh and muscles
    :param thigh_offset: initial offset angle of thigh in radians
    :param get_femoris_activation: femoris activation function w.r.t time
    :param get_lateralis_activation: lateralis activation function w.r.t time
    :param get_medialis_activation: medialis activation function w.r.t time
    :param get_intermedius_activation: intermedius activation function w.r.t time
    :param force_length_regression: function that regresses force from length
    :param force_velocity_regression: function that regresses force from velocity
    """

    rest_quad_muscle_length = quad_muscle_length(QUAD_REST_ANGLE, thigh_offset)
    rest_length_femoris = rest_quad_muscle_length
    rest_length_lateralis = rest_quad_muscle_length
    rest_length_medialis = rest_quad_muscle_length
    rest_length_intermedius = rest_quad_muscle_length

    femoris = HillTypeMuscle(
                FEMORIS_MAX_FORCE, 
                FEMORIS_MUSCLE_PERCENT*rest_length_femoris, 
                FEMORIS_TENDON_PERCENT*rest_length_femoris)
    lateralis = HillTypeMuscle(
                LATERALIS_MAX_FORCE, 
                LATERALIS_MUSCLE_PERCENT*rest_length_lateralis, 
                LATERALIS_TENDON_PERCENT*rest_length_lateralis)
    medialis = HillTypeMuscle(
                MEDIALIS_MAX_FORCE, 
                MEDIALIS_MUSCLE_PERCENT*rest_length_medialis, 
                MEDIALIS_TENDON_PERCENT*rest_length_medialis)
    intermedius = HillTypeMuscle(
                INTERMEDIUS_MAX_FORCE, 
                INTERMEDIUS_MUSCLE_PERCENT*rest_length_intermedius, 
                INTERMEDIUS_TENDON_PERCENT*rest_length_intermedius)

    def f(x, t):
        """
        Analytical Equation to Use in Solver
        """

        femoris_activation = get_femoris_activation(t)
        lateralis_activation = get_lateralis_activation(t)
        medialis_activation = get_medialis_activation(t)
        intermedius_activation = get_intermedius_activation(t)

        return dynamics(x, thigh_offset, femoris, lateralis, medialis, 
             intermedius, femoris_activation, lateralis_activation,
             medialis_activation, intermedius_activation,
             force_length_regression, force_velocity_regression,
             )

    tspan = [0, T]
    time = np.linspace(tspan[0], tspan[-1], 10000)
    y = odeint(f, initialCondition, time, full_output=False)

    theta = y[:,0]
    angular_velocity = y[:, 1]
    foot_velocity = np.zeros(y.shape[0])

    for i in range(y.shape[0]):
        foot_velocity[i] = SHANK_LENGTH*angular_velocity[i]

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    axs[0].plot(time, theta, linewidth=1.5)
    axs[0].set_ylabel('Body Angle (rad)')

    axs[1].plot(time, foot_velocity, linewidth=1.5)
    axs[1].legend(loc='upper left')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity (m/s)')

    return fig, {
        "max_velocity" : max(foot_velocity),
        "time" : time,
        "theta" : theta,
        "velocity": foot_velocity
    }
