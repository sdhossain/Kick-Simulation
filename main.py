import sys
sys.path.append('.')
sys.path.append('./muscle_modelling')

from simulate import simulate
from muscle_modelling.regression import get_muscle_force_length_regression,\
                       get_muscle_force_velocity_regression

def main():
    force_length_regression = get_muscle_force_length_regression()
    force_velocity_regression = get_muscle_force_velocity_regression()

    T = 5
    simulate(T, force_length_regression, force_velocity_regression)

if __name__ == "__main__":
    main()