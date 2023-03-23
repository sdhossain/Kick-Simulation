import numpy as np

def force_length_muscle(lm, force_length_regression):
    """
    Compute the force-length scale factor for a muscle based on its length.

    :param lm: muscle (contractile element) length
    :param force_length_regression: regression function for for force length
    :return force_length_scale_factor: the force-length scale factor
    """
    
    if len(np.array(lm).shape) <= 0:
      lm = np.array([lm])

    force_length_scale_factor = force_length_regression(lm)

    return force_length_scale_factor


def force_length_parallel(lm):
    """
    Compute the normalized force produced by the parallel elastic element of a muscle
    based on its normalized length.

    :param lm: normalized length of muscle (contractile element)
    :return normalize_PE_force: normalized force produced by parallel elastic element
    """

    if len(np.array(lm).shape) <= 0:
      lm = np.array([lm])

    normalize_PE_force = np.zeros(lm.shape)

    for i in range(len(lm)):
        if lm[i] < 1:
            normalize_PE_force[i] = 0
        else:
            normalize_PE_force[i] = 3*((lm[i] - 1)**2)/(0.6 + lm[i] - 1)

    return normalize_PE_force


def force_length_tendon(lt):
    """
    Compute the normalized tension produced by the tendon (series elastic element)
    based on its normalized length.

    :param lt: normalized length of tendon (series elastic element)
    :return normalize_tendon_tension: normalized tension produced by tendon
    """

    if len(np.array(lt).shape) <= 0:
      lt = np.array([lt])

    normalize_tendon_tension = np.zeros(lt.shape)

    for i in range(len(lt)):
        if lt[i] < 1:
            normalize_tendon_tension[i] = 0
        else:
            normalize_tendon_tension[i] = 10*(lt[i]-1) + 240*((lt[i]-1)**2)

    return normalize_tendon_tension