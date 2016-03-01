from libc.math cimport cos, sqrt, M_PI

import numpy as np
cimport numpy as np

from scipy import stats

cdef int WET = 0
cdef int DRY = 1


cdef class AnnualHarmonicModel:
    """
    Simple annual harmonic series model of a parameter.

    The harmonic series is given by the equation,

    $$ v = C_0 + \sum^m_{j=1}{C_j \cos(\frac{ij}{T} + \theta_j}) $$

    """
    cdef double mean
    cdef double[:] amplitude
    cdef double[:] phase

    def __init__(self, double mean, double[:] amplitude=None, double[:] phase=None):
        """
        :param mean: Mean of the series
        :param amplitude: Array of amplitudes for each frequency in the series
        :param phase: Array of phase offsets for each frequency in the series
        """
        self.mean = mean
        if amplitude is not None or phase is not None:
            if amplitude.shape[0] != phase.shape[0]:
                raise ValueError('Shape of amplitude and phase arrays must be the same.')

        self.amplitude = amplitude
        self.phase = phase

    cpdef double value(self, double day_of_year):
        """ Evaluate the model for the given Julian day of the year.
        """
        cdef double T = 365/(2*M_PI)
        cdef int j
        cdef double out
        cdef int M = self.amplitude.shape[0]

        out = self.mean
        for j in range(self.amplitude.shape[0]):
            out += self.amplitude[j]*cos(day_of_year*(j+1)/T + self.phase[j])

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
        p_ww = self.wet_given_wet.value(day_of_year)
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


cdef class TemperatureSimulator:
    """
    Simple lag-1 auto-correlated mean temperature simulator.
    """
    cdef AnnualHarmonicModel mean_wet
    cdef AnnualHarmonicModel mean_dry
    cdef AnnualHarmonicModel std_wet
    cdef AnnualHarmonicModel std_dry
    cdef double autocorr1
    cdef double[:] _state

    def __init__(self, int size, double autocorr1, AnnualHarmonicModel mean_wet, AnnualHarmonicModel mean_dry,
                 AnnualHarmonicModel std_wet, AnnualHarmonicModel std_dry):
        self.mean_wet = mean_wet
        self.mean_dry = mean_dry
        self.std_wet = std_wet
        self.std_dry = std_dry
        self.autocorr1 = autocorr1
        # Todo make this user definable or based on parameters given.
        # Current default is based on mead_dry value
        self._state = np.ones(size)*mean_dry.value(1)


    cpdef step(self, double day_of_year, int[:] wet_dry, double[:] out=None):
        """
        Step the simulator the next Julian day of the year.

        The simulator only retains the temperature state and does not retain the day of the year, which therefore must
        be given to this method. The optional out array is returned by the function and contains the values of
        stochastic mean temperature simulated by the model.
        """
        cdef int i
        cdef double mw, md, stdw, stdd, x, b
        cdef int N = self._state.shape[0]

        if out is None:
            out = np.empty(N)

        mw = self.mean_wet.value(day_of_year)
        md = self.mean_dry.value(day_of_year)
        stdw = self.std_wet.value(day_of_year)
        stdd = self.std_dry.value(day_of_year)
        b = sqrt(1 - self.autocorr1**2)
        for i in range(N):
            # Calculate the random residual for this generation (including lag-1 correlation with previous state)
            x = self.autocorr1*self._state[i] + b*stats.norm.rvs(size=1)
            # Convert to absolute temperature based on wet/dry state
            if wet_dry[i] == WET:
                out[i] = x*stdw + mw
            else:
                out[i] = x*stdd + md
            # update residual state for next step
            self._state[i] = x
        return out


cdef class RainfallTemperatureSimulator:
    cdef RainfallSimulator rain_sim
    cdef TemperatureSimulator temp_sim

    def __init__(self, RainfallSimulator rain_sim, TemperatureSimulator temp_sim):
        if rain_sim._state.shape[0] != temp_sim._state.shape[0]:
            raise ValueError("Rainfall and temperature simulators must be setup the same size.")
        self.rain_sim = rain_sim
        self.temp_sim = temp_sim

    cpdef step(self, double day_of_year, double[:] rainfall=None, double[:] temperature=None):
        cdef int N = self.rain_sim._state.shape[0]

        # Setup optional output arrays
        if rainfall is None:
            rainfall = np.empty(N)
        if temperature is None:
            temperature = np.empty(N)

        self.rain_sim.step(day_of_year, rainfall)
        # The internal state of the rainfall simulator is now today's wet/dry status.
        # This is used for temperature simulation.
        self.temp_sim.step(day_of_year, self.rain_sim._state, temperature)

        return rainfall, temperature