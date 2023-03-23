import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from muscle_length import soleus_length, tibialis_length
from hill_type_muscle import HillTypeMuscle
from dynamics import dynamics
from gravity_moment import gravity_moment

def simulate(T, force_length_regression, force_velocity_regression):
    """
    Runs a simulation of the model and plots results.

    :param T: total time to simulate, in seconds
    :param force_length_regression: function that regresses force from length
    :param force_velocity_regression: function that regresses force from velocity
    """

    rest_length_soleus = soleus_length(math.pi/2)
    rest_length_tibialis = tibialis_length(math.pi/2)

    soleus = HillTypeMuscle(
        16000, 0.6*rest_length_soleus, 0.4*rest_length_soleus)
    tibialis = HillTypeMuscle(
        2000, 0.6*rest_length_tibialis, 0.4*rest_length_tibialis)

    def f(x, t):
        return dynamics(x, soleus, tibialis, 
            force_length_regression, force_velocity_regression)

    tspan = [0, T]
    time = np.linspace(tspan[0], tspan[-1], 10000)
    initialCondition = [math.pi/2, 0, 1, 1]
    y = odeint(f, initialCondition, time, full_output=False)

    theta = y[:,0]
    soleus_norm_length_muscle = y[:,2]
    tibialis_norm_length_muscle = y[:,3]

    soleus_moment_arm = 0.05
    tibialis_moment_arm = 0.03
    soleus_moment = np.zeros(y.shape[0])
    tibialis_moment = np.zeros(y.shape[0])

    for i in range(y.shape[0]):
        soleus_moment[i] = soleus_moment_arm * soleus.get_force(soleus_length(theta[i]), soleus_norm_length_muscle[i])
        tibialis_moment[i] = -tibialis_moment_arm * tibialis.get_force(tibialis_length(theta[i]), tibialis_norm_length_muscle[i])

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    axs[0].plot(time, theta, linewidth=1.5)
    axs[0].set_ylabel('Body Angle (rad)')

    axs[1].plot(time, soleus_moment, 'r', linewidth=1.5, label='Soleus')
    axs[1].plot(time, tibialis_moment, 'g', linewidth=1.5, label='Tibialis')
    axs[1].plot(time, gravity_moment(theta), 'k', linewidth=1.5, label='Gravity')
    axs[1].legend(loc='upper left')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Torques (Nm)')

    plt.show()