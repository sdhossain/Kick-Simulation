import sys
sys.path.append('.')
sys.path.append('./muscle_modelling')

import numpy as np

from simulate import simulate
from muscle_modelling.regression import get_muscle_force_length_regression,\
                       get_muscle_force_velocity_regression

THIGH_OFFSET = np.pi/6

def main():
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

    results = simulate(T, initialCondition, thigh_offset, get_femoris_activation, 
                get_lateralis_activation, get_medialis_activation, get_intermedius_activation, 
                force_length_regression, force_velocity_regression)

if __name__ == "__main__":
    main()