import sys
sys.path.append('.')
sys.path.append('./muscle_modelling')

import numpy as np
import itertools

from simulate import simulate
from muscle_modelling.regression import get_muscle_force_length_regression,\
                       get_muscle_force_velocity_regression

THIGH_OFFSET = np.pi/6

def sample_simulation():
    force_length_regression = get_muscle_force_length_regression()
    force_velocity_regression = get_muscle_force_velocity_regression()

    # SAMPLE SIMULATION

    T = 1
    initialCondition = [np.pi/8, 0.1, 1, 1, 1, 1]
    thigh_offset = THIGH_OFFSET
    get_femoris_activation = lambda x: 1
    get_lateralis_activation = lambda x: 1
    get_medialis_activation = lambda x: 1
    get_intermedius_activation = lambda x: 1

    fig, results = simulate(T, initialCondition, thigh_offset, get_femoris_activation, 
                get_lateralis_activation, get_medialis_activation, get_intermedius_activation, 
                force_length_regression, force_velocity_regression)

    fig.savefig('sample_simulation.png')

def sample_simulation_sweep():
    force_length_regression = get_muscle_force_length_regression()
    force_velocity_regression = get_muscle_force_velocity_regression()

    # SAMPLE SWEEP

    # define parameter sets
    T = 1
    initial_thetas = [1, 2, 3]
    thigh_offsets = [1, 2, 3]
    femoris_activation_funcs = [lambda x: 1, lambda x: np.cos((x*np.pi)/2)]
    lateralis_activation_funcs = [lambda x: 1, lambda x: np.cos((x*np.pi)/2)]
    medialis_activation_funcs = [lambda x: 1, lambda x: np.cos((x*np.pi)/2)]
    intermedius_activation_funcs = [lambda x: 1, lambda x: np.cos((x*np.pi)/2)]

    # Generate all possible combinations using itertools.product
    param_combinations = list(itertools.product(
        initial_thetas, thigh_offsets, femoris_activation_funcs, 
        lateralis_activation_funcs, medialis_activation_funcs, 
        intermedius_activation_funcs))

    # Execute each combination
    for combination in param_combinations:
        initialCondition = [combination[0], 0.1, 1, 1, 1, 1]
        thigh_offset = combination[1]
        get_femoris_activation = combination[2]
        get_lateralis_activation = combination[3]
        get_medialis_activation = combination[4]
        get_intermedius_activation = combination[5]

        fig, results = simulate(T, initialCondition, thigh_offset, get_femoris_activation, 
                    get_lateralis_activation, get_medialis_activation, get_intermedius_activation, 
                    force_length_regression, force_velocity_regression)

if __name__ == "__main__":
    sample_simulation_sweep()