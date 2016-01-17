import numpy as np
cimport numpy as np

from scipy import stats

cdef int WET = 0
cdef int DRY = 1


cdef class AnnualHarmonicModel:
    """
    Simple annual harmonic series model of a parameter.

    """
    cdef double mean
    cdef double[:] amplitude
    cdef double[:] phase

    def __init__(self, double mean, double[:] amplitude=None, double[:] phase=None):
        self.mean = mean
        if amplitude is not None or phase is not None:
            if amplitude.shape[0] != phase.shape[0]:
                raise ValueError('Shape of amplitude and phase arrays must be the same.')

        self.amplitude = amplitude
        self.phase = phase

    cpdef double value(self, double day_of_year):
        cdef double T = 365/(2*np.pi)
        cdef int j
        cdef double out
        cdef int M = self.amplitude.shape[0]

        out = self.mean
        for j in range(self.amplitude.shape[0]):
            out += self.amplitude[j]*np.cos(day_of_year*(j+1)/T + self.phase[j])

        return out


cdef class RainfallSimulator:
    """
    Simple two state Markov Chain based rainfall simulator.

    The simulator is based on that described by Richardson (1980). The retains an internal state of either wet or dry
    the transition probabilities are given for wet days based on a preceding wet or dry day. An exponential distirbution
    is used for the intensity of rainfall on wet days.

    The transition probabilities and intensity distribution parameter (lambda) are described using the AnnualHarmonicModel
    class which permits seasonal variation in the parameters.

    Richardson, C. W. (1981), Stochastic simulation of daily precipitation, temperature, and solar radiation,
    Water Resour. Res., 17(1), 182–190, doi:10.1029/WR017i001p00182.
    """
    cdef AnnualHarmonicModel wet_given_dry
    cdef AnnualHarmonicModel wet_given_wet
    cdef AnnualHarmonicModel intensity
    cdef int[:] _state


    def __init__(self, int size, AnnualHarmonicModel wet_given_dry, AnnualHarmonicModel wet_given_wet,
                 AnnualHarmonicModel intensity):
        self.wet_given_dry = wet_given_dry
        self.wet_given_wet = wet_given_wet
        self.intensity = intensity
        # Todo make this user definable or based on parameters given.
        self._state = np.ones(size, dtype=np.int32)*DRY

    def step(self, double day_of_year, double[:] out=None):
        """
        Step the simulator the next Julian day of the year.

        The simulator only retains the wet/dry state and does not retain the day of the year, which therefore must
        be given to this method. The optional out array is returned by the function and contains the values of
        stochastic rainfall simulated by the model.
        """
        cdef int i, j
        cdef double p_wet, p_wd, p_ww, lmbda
        cdef int N = self._state.shape[0]
        
        # Generate random variables for today's state
        cdef double[:] p = np.random.rand(N)

        if out is None:
            out = np.empty(N)

        # Calculate the wet probabilities for this day of the year
        p_wd = self.wet_given_dry.value(day_of_year)
        p_ww = self.wet_given_dry.value(day_of_year)
        lmbda = self.intensity.value(day_of_year)

        for i in range(N):
            # Select the probability today is wet based on the current (now yesterday's) state
            if self._state[i] == DRY:
                p_wet = p_wd
            else:
                p_wet = p_ww

            # Apply the random variable to the probability estimate
            if p[i] <= p_wet:
                # Today is wet; estimate rainfall
                out[i] = stats.expon.rvs(size=1, scale=1.0/lmbda)
                self._state[i] = WET
            else:
                # Today is dry; no rainfall
                out[i] = 0.0
                self._state[i] = DRY

        return out


