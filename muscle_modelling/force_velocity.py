import sys
sys.path.append('.')
sys.path.append('./muscle_modelling')

import numpy as np
from muscle_modelling.model_eval import model_eval

def force_velocity_muscle(vm, force_velocity_regression):
    """
    Compute the force-velocity scale factor for a muscle based on its velocity.

    :param vm: muscle (contractile element) velocity
    :param force_velocity_regression: regression function for for force velocity
    :return force_velocity_scale_factor: the force-velocity scale factor
    """

    vm = np.array([vm])
    if vm.shape[1] > vm.shape[0]:
        vm = vm.T

    force_velocity_scale_factor = model_eval(
                        'Sigmoid', vm, force_velocity_regression)

    return force_velocity_scale_factor