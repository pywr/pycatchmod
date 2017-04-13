from ._catchmod import SoilMoistureDeficitStore, LinearStore, NonLinearStore, SubCatchment, Catchment, OudinCatchment

__version__ = "0.2"

def run_catchmod(C, rainfall, pet, dates=None):
    """Convenience function for running catchmod

    Parameters
    C : pycatchmod.Catchment
    rainfall : numpy.array
    pet : numpy.array
    """
    import numpy as np

    C.reset()

    # number of scenarios
    N = C.subcatchments[0].soil_store.initial_upper_deficit.shape[0]
    assert(rainfall.shape[1] == N)

    # input timesteps
    M = rainfall.shape[0]
    # output timesteps
    if dates is not None:
        M2 = len(dates)
    else:
        M2 = M

    perc = np.zeros((len(C.subcatchments), N))
    outflow = np.zeros_like(perc)
    total_outflow = np.zeros(M)
    flow = np.zeros([M2, N])
    flows = np.zeros([M2, len(C.subcatchments)])

    # TODO: add option to enable/disable extra leap days

    i = 0
    for j in range(M2):
        if M != M2:
            date = dates[j]
            if date.month == 2 and date.day == 29:
                # input data is missing leap days, use previous day
                i -= 1

        r = rainfall[i, ...].reshape(N).astype(np.float64)
        p = pet[i, ...].reshape(N).astype(np.float64)

        C.step(r, p, perc, outflow)

        flows[j, ...] = outflow[:,0]
        flow[j, ...] = outflow.sum(axis=0).reshape(rainfall.shape[1:])

        i += 1

    return flow
