import scipy
from force_length import force_length_muscle, \
                         force_length_parallel, \
                         force_length_tendon
from force_velocity import force_velocity_muscle

def get_velocity(a, lm, lt, lr, vr):
    """
    Returns velocity given activation, muscle-length, tendon-length,
    length and velocity regression functions.

    :param a: activation for the muscle
    :param lm: normalized length of muscle (contractile element)
    :param lt: normalized length of tendon (series elastic element)
    :param lr: force-length-regression function
    :param vr: force-velocity-regression function
    :return root: velocity
    """
    
    beta = 0.1

    def fun(vm):
        return 1*(a*force_length_muscle(lm, lr)*force_velocity_muscle(vm, vr) \
                  + force_length_parallel(lm)+ beta*vm) - force_length_tendon(lt)

    vm = 0
    root = scipy.optimize.fsolve(fun, vm)

    return root