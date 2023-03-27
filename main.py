import sys
sys.path.append('.')
sys.path.append('./muscle_modelling')

import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

from simulate import simulate
from muscle_modelling.regression import get_muscle_force_length_regression,\
                       get_muscle_force_velocity_regression

THIGH_OFFSET = np.pi/6

def sample_simulation():
    force_length_regression = get_muscle_force_length_regression()
    force_velocity_regression = get_muscle_force_velocity_regression()

    # SAMPLE SIMULATION

    T = 0.75
    initialCondition = [np.pi/8, 0.1, 1, 1, 1, 1]
    thigh_offset = THIGH_OFFSET
    get_femoris_activation = lambda x: 1
    get_lateralis_activation = lambda x: 1
    get_medialis_activation = lambda x: 1
    get_intermedius_activation = lambda x: 1

    fig, results = simulate(T, initialCondition, thigh_offset, get_femoris_activation, 
                get_lateralis_activation, get_medialis_activation, get_intermedius_activation, 
                force_length_regression, force_velocity_regression)

    print(results['max_velocity'])
    fig.savefig('sample_simulation.png')

def sample_simulation_sweep():
    force_length_regression = get_muscle_force_length_regression()
    force_velocity_regression = get_muscle_force_velocity_regression()

    # SAMPLE SWEEP

    # define parameter sets
    T = 0.75
    initial_thetas = [np.pi/4]
    thigh_offsets = [-np.pi/6, 0, np.pi/6]
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

def sweep_thigh_offset():
    force_length_regression = get_muscle_force_length_regression()
    force_velocity_regression = get_muscle_force_velocity_regression()

    # SAMPLE SWEEP

    # define parameter sets
    T = 1
    initial_thetas = [np.pi/4]
    thigh_offsets = [-np.pi/12, 0, np.pi/12, np.pi/10, np.pi/8, np.pi/6]
    femoris_activation_funcs = [lambda x: 1]
    lateralis_activation_funcs = [lambda x: 1]
    medialis_activation_funcs = [lambda x: 1]
    intermedius_activation_funcs = [lambda x: 1]

    # Generate all possible combinations using itertools.product
    param_combinations = list(itertools.product(
        initial_thetas, thigh_offsets, femoris_activation_funcs, 
        lateralis_activation_funcs, medialis_activation_funcs, 
        intermedius_activation_funcs))

    max_velocities = np.zeros(len(thigh_offsets))

    # Execute each combination
    for i, combination in enumerate(tqdm(param_combinations)):
        initialCondition = [combination[0], 0, 1, 1, 1, 1]
        thigh_offset = combination[1]
        get_femoris_activation = combination[2]
        get_lateralis_activation = combination[3]
        get_medialis_activation = combination[4]
        get_intermedius_activation = combination[5]

        fig, results = simulate(T, initialCondition, thigh_offset, get_femoris_activation, 
                    get_lateralis_activation, get_medialis_activation, get_intermedius_activation, 
                    force_length_regression, force_velocity_regression)

        max_velocities[i] = results['max_velocity']

    fig = plt.figure()
    plt.plot(thigh_offsets, max_velocities, linewidth=1.5)
    plt.ylabel('Maximum Foot Velocity (m/s)')
    plt.xlabel('Thigh Offset (radians)')
    plt.show()
    fig.savefig('thigh_offset_sweep.png')


def sweep_initial_theta():
    force_length_regression = get_muscle_force_length_regression()
    force_velocity_regression = get_muscle_force_velocity_regression()

    # SAMPLE SWEEP

    # define parameter sets
    T = 1
    initial_thetas = [0, np.pi/8, np.pi/6, np.pi/4, np.pi/2, np.pi]
    thigh_offsets = [np.pi/6]
    femoris_activation_funcs = [lambda x: 1]
    lateralis_activation_funcs = [lambda x: 1]
    medialis_activation_funcs = [lambda x: 1]
    intermedius_activation_funcs = [lambda x: 1]

    # Generate all possible combinations using itertools.product
    param_combinations = list(itertools.product(
        initial_thetas, thigh_offsets, femoris_activation_funcs, 
        lateralis_activation_funcs, medialis_activation_funcs, 
        intermedius_activation_funcs))

    max_velocities = np.zeros(len(initial_thetas))

    # Execute each combination
    for i, combination in enumerate(tqdm(param_combinations)):
        initialCondition = [combination[0], 0, 1, 1, 1, 1]
        thigh_offset = combination[1]
        get_femoris_activation = combination[2]
        get_lateralis_activation = combination[3]
        get_medialis_activation = combination[4]
        get_intermedius_activation = combination[5]

        fig, results = simulate(T, initialCondition, thigh_offset, get_femoris_activation, 
                    get_lateralis_activation, get_medialis_activation, get_intermedius_activation, 
                    force_length_regression, force_velocity_regression)

        max_velocities[i] = results['max_velocity']

    fig = plt.figure()
    plt.plot(initial_thetas, max_velocities, linewidth=1.5)
    plt.ylabel('Maximum Foot Velocity (m/s)')
    plt.xlabel('Initial Theta (radians)')
    plt.show()
    fig.savefig('initial_theta_sweep.png')

def sweep_muscle_activations():
    force_length_regression = get_muscle_force_length_regression()
    force_velocity_regression = get_muscle_force_velocity_regression()

    # SAMPLE SWEEP

    # define parameter sets
    T = 0.75
    muscles = ['femoris', 'lateralis', 'medialis', 'intermedius']

    max_velocities = np.zeros(4)

    # Execute each combination
    for i, muscle in enumerate(tqdm(muscles)):
        initialCondition = [0, 0, 1, 1, 1, 1]
        thigh_offset = np.pi/6

        if muscle == 'femoris':
            get_femoris_activation = lambda x: 1
        else:
            get_femoris_activation = lambda x: 0
        if muscle == 'lateralis':
            get_lateralis_activation = lambda x: 1
        else:
            get_lateralis_activation = lambda x: 0
        if muscle == 'medialis':
            get_medialis_activation = lambda x: 1
        else:
            get_medialis_activation = lambda x: 0
        if muscle == 'intermedius':
            get_intermedius_activation = lambda x: 1
        else:
            get_intermedius_activation = lambda x: 0

        fig, results = simulate(T, initialCondition, thigh_offset, get_femoris_activation, 
                    get_lateralis_activation, get_medialis_activation, get_intermedius_activation, 
                    force_length_regression, force_velocity_regression)

        max_velocities[i] = results['max_velocity']

    fig = plt.figure()
    plt.plot(muscles, max_velocities, linewidth=1.5)
    plt.ylabel('Maximum Foot Velocity (m/s)')
    plt.xlabel('Muscle')
    plt.show()
    fig.savefig('muscle_sweep.png')


def sweep_final():
    force_length_regression = get_muscle_force_length_regression()
    force_velocity_regression = get_muscle_force_velocity_regression()

    # SAMPLE SWEEP

    # define parameter sets
    T = 1
    initial_thetas = [0, np.pi/6, np.pi/4]
    thigh_offsets = [-np.pi/12, 0, np.pi/6, np.pi/4]
    femoris_activation_funcs = [lambda x: 1]
    lateralis_activation_funcs = [lambda x: 1]
    medialis_activation_funcs = [lambda x: 1]
    intermedius_activation_funcs = [lambda x: 1]

    # Generate all possible combinations using itertools.product
    param_combinations = list(itertools.product(
        initial_thetas, thigh_offsets, femoris_activation_funcs, 
        lateralis_activation_funcs, medialis_activation_funcs, 
        intermedius_activation_funcs))

    max_velocities = np.zeros(len(param_combinations))

    # Execute each combination
    for i, combination in enumerate(tqdm(param_combinations)):
        initialCondition = [combination[0], 0, 1, 1, 1, 1]
        thigh_offset = combination[1]
        get_femoris_activation = combination[2]
        get_lateralis_activation = combination[3]
        get_medialis_activation = combination[4]
        get_intermedius_activation = combination[5]

        fig, results = simulate(T, initialCondition, thigh_offset, get_femoris_activation, 
                    get_lateralis_activation, get_medialis_activation, get_intermedius_activation, 
                    force_length_regression, force_velocity_regression)

        max_velocities[i] = results['max_velocity']

    max_velocity_idx = np.argmax(max_velocities)
    best_params = param_combinations[max_velocity_idx]
    print("The Best Params Are:")
    print("Initial Theta: ", best_params[0])
    print("Thigh Offset: ", best_params[1])

if __name__ == "__main__":
    sweep_thigh_offset()
    sweep_initial_theta()
    sweep_muscle_activations()
    sweep_final()