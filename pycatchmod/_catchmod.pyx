import numpy as np
cimport numpy as np

EPS = 1e-12

cdef class SoilMoistureDeficitStore:
    # Current soil moisture deficit
    # Units: mm
    cdef public double[:] upper_deficit
    cdef public double[:] lower_deficit

    # A fixed fraction of precipitation that bypasses the soil horizon even during periods of soil moisture deficit
    # Units: % [0, 1]
    cdef public double direct_percolation

    # Value of deficit above which evaporation occurs at a reduced rate
    # Units: mm
    cdef public double potential_drying_constant

    # The reduced rate at which soil moisture is evaporated once the potential drying constant has been exceeded
    # Units: mm/mm
    cdef public double gradient_drying_curve


    def __init__(self, double[:] initial_upper_deficit, double[:] initial_lower_deficit, **kwargs):
        self.upper_deficit = initial_upper_deficit
        self.lower_deficit = initial_lower_deficit

        self.direct_percolation = kwargs.pop('direct_percolation', 0.0)
        self.potential_drying_constant = kwargs.pop('potential_drying_constant', 0.0)
        self.gradient_drying_curve = kwargs.pop('gradient_drying_curve', 1.0)

    cpdef step(self, double[:] rainfall, double[:] pet, double area, double[:] percolation):
        """
        Step the soil moisture store one day.

        :param rainfall: memoryview of rainfall in mm/day
        :param pet: memoryview of PET in mm/day
        :param percolation: memoryview of outputted total percolation from the SoilMositureDeficitStore
        """

        cdef int i
        cdef int n = self.upper_deficit.shape[0]
        cdef double effective_rainfall, direct_percolation

        for i in range(n):
            # First calculate direct percolation, this proportion bypasses the store entirely
            direct_percolation = rainfall[i] * self.direct_percolation
            percolation[i] = direct_percolation

            # Effective rainfall = rainfall less PET and direct_percolation
            effective_rainfall = rainfall[i] - pet[i] - direct_percolation

            # The effective_rainfall variable is consumed (reduced) by the following processes to determine if there
            # is a net saturated percolation at the end.

            if effective_rainfall > 0.0:
                # Wetting
                # Upper deficit replenishes first
                if self.upper_deficit[i] > effective_rainfall:
                    # Upper deficit greater than the effective rainfall, reduce the deficit
                    self.upper_deficit[i] -= effective_rainfall
                    effective_rainfall = 0.0
                elif self.upper_deficit[i] > 0.0:
                    # Upper deficit smaller than effective rainfall. Upper deficit is removed, and effective rainfall
                    # reduced. Any excess can reduce deficit in lower store
                    effective_rainfall -= self.upper_deficit[i]
                    self.upper_deficit[i] = 0.0

                # If there is remaining effective rainfall, then reduce any lower deficits
                if effective_rainfall > 0.0:
                    if self.lower_deficit[i] > effective_rainfall:
                        # Lower deficit greater than the remaining effective rainfall, reduce the deficit
                        self.lower_deficit[i] -= effective_rainfall
                        effective_rainfall = 0.0
                    elif self.lower_deficit[i] > 0.0:
                        # Lower deficit smaller than remaining effective rainfall. Lower deficit is removed, and
                        # effective rainfall is reduced. Any excess can contribute to saturated percolation
                        effective_rainfall -= self.lower_deficit[i]
                        self.lower_deficit[i] = 0.0

                # If there is still any remaining effective rainfall, then it is saturated percolation
                if effective_rainfall > 0.0:
                    percolation[i] += effective_rainfall

            else:
                # Drying
                if self.upper_deficit[i] < self.potential_drying_constant + effective_rainfall:
                    # Upper deficit sufficiently less than the PDC threshold, just increase the deficit
                    self.upper_deficit[i] -= effective_rainfall
                    effective_rainfall = 0.0
                elif self.upper_deficit[i] < self.potential_drying_constant:
                    # Upper deficit near to PDC threshold
                    effective_rainfall += self.potential_drying_constant - self.upper_deficit[i]
                    self.upper_deficit[i] = self.potential_drying_constant

                # if there is remaining negative effective rainfall dry the lower store at reduced rate
                if effective_rainfall < 0.0:
                    # there is no limit to the size of the lower store
                    self.lower_deficit[i] -= effective_rainfall * self.gradient_drying_curve

            # Finally multiply by area to get volume rate
            percolation[i] *= area

cdef class LinearStore:
    # Current outflow of the store at the beginning of a time-step of the linear store
    cdef public double[:] previous_outflow

    # Represents temporary storage in the unsaturated zone
    # Units: days
    cdef public double linear_storage_constant

    def __init__(self, double[:] initial_outflow, **kwargs):
        self.previous_outflow = initial_outflow

        self.linear_storage_constant = kwargs.pop('linear_storage_constant', 1.0)
        if self.linear_storage_constant < 0.0:
            raise ValueError("Invalid value for linear storage constant. Must be > 0.0")

    cpdef step(self, double[:] inflow, double[:] outflow):
        """
        """
        cdef double b = np.exp(-1.0/self.linear_storage_constant)
        cdef int i
        cdef int n = self.previous_outflow.shape[0]

        for i in range(n):
            # Calculate outflow from this store as the average over the timestep
            outflow[i] = inflow[i] - self.linear_storage_constant*(inflow[i] - self.previous_outflow[i])*(1.0 - b)
            # Record the end of timestep flow ready for the next timestep
            self.previous_outflow[i] = inflow[i] - (inflow[i] - self.previous_outflow[i])*b


cdef class NonLinearStore:
    # Current volume of the linear store
    cdef public double[:] previous_outflow

    # Represents storage in the saturated zone/aquifer
    # Units: days km^{-2}
    cdef public double nonlinear_storage_constant

    def __init__(self, double[:] initial_outflow, **kwargs):
        self.previous_outflow = initial_outflow

        self.nonlinear_storage_constant = kwargs.pop('nonlinear_storage_constant', 1.0)
        if self.nonlinear_storage_constant == 0.0:
            raise ValueError("Invalid value for nonlinear storage constant. Must be > 0.0")

    cpdef step(self, double[:] inflow, double[:] outflow):
        """
        """
        cdef double a, b, V, t
        # TODO variable time-step
        cdef double T = 1.0
        cdef double Q2
        cdef int i
        cdef int n = self.previous_outflow.shape[0]

        for i in range(n):

            if self.previous_outflow[i] > 0.0:
                if inflow[i] < 0.0:
                    # Case (b)
                    a = np.arctan(np.sqrt(-self.previous_outflow[i]/inflow[i])) - np.sqrt(-inflow[i]/self.nonlinear_storage_constant)
                    if a > 0.0:
                        Q2 = -inflow[i]*np.tan(a)**2
                    else:
                        Q2 = 0.0
                elif inflow[i] > 0.0:
                    # Case (c)
                    a = np.sqrt(self.previous_outflow[i]) - np.sqrt(inflow[i])
                    a /= np.sqrt(self.previous_outflow[i]) + np.sqrt(inflow[i])

                    b = -2.0*T*np.sqrt(inflow[i]/self.nonlinear_storage_constant)

                    Q2 = inflow[i]*(1 + a*np.exp(b))**2
                    Q2 /= (1 - a*np.exp(b))**2
                else:
                    # Case (a)  inflow[i] == 0.0
                    Q2 = self.nonlinear_storage_constant
                    Q2 /= (np.sqrt(self.nonlinear_storage_constant/self.previous_outflow[i]) + T)**2
            else:
                # Case (d) - less than zero initial outflow
                # Ensure the sqrt is done with a +ve number (or zero)

                V = -np.sqrt(np.abs(self.previous_outflow[i])*self.nonlinear_storage_constant)
                V += inflow[i]*T

                if V > 0.0:
                    # New volume means flow will return after a period of time, t
                    t = T - V/inflow[i]

                    # Wilby (1994) is unclear what 'a' should be in case (d)
                    # Testing reveals it should be unity.
                    a = 1.0
                    b = -2.0*(T - t)*np.sqrt(inflow[i]/self.nonlinear_storage_constant)

                    Q2 = inflow[i]*(1 - a*np.exp(b))**2
                    Q2 /= (1 + a*np.exp(b))**2
                else:
                    Q2 = 0.0

            # Wilby 1994 gives no analytical solution to the mean flow in a timestep, therefore set the outflow
            # for this timestep as the average of the outflow at beginning and end of timestep
            outflow[i] = (self.previous_outflow[i]+Q2)/2.0
            # Update previous outflow for next time-step
            self.previous_outflow[i] = Q2


cdef class SubCatchment:
    cdef SoilMoistureDeficitStore _soil
    cdef LinearStore _linear
    cdef NonLinearStore _nonlinear
    cdef float area

    def __init__(self, area, double[:] initial_upper_deficit, double[:] initial_lower_deficit,
                 double[:] initial_linear_outflow, double[:] initial_nonlinear_outflow, **kwargs):
        self.area = area
        self._soil = SoilMoistureDeficitStore(initial_upper_deficit, initial_lower_deficit, **kwargs)
        self._linear = LinearStore(initial_linear_outflow, **kwargs)
        self._nonlinear = NonLinearStore(initial_nonlinear_outflow, **kwargs)

    cpdef step(self, double[:] rainfall, double[:] pet, double[:] percolation, double[:] outflow):
        """ Step the subcatchment one timestep
        """
        self._soil.step(rainfall, pet, self.area, percolation)
        self._linear.step(percolation, outflow)
        self._nonlinear.step(outflow, outflow)


cdef class Catchment:
    cdef list subcatchments

    def __init__(self, subcatchments):
        self.subcatchments = list(subcatchments)

    cpdef step(self, double[:] rainfall, double[:] pet, double[:, :] percolation, double[:, :] outflow):
        """ Step the subcatchment one timestep
        """
        cdef int i
        cdef SubCatchment subcatchment
        for i, subcatchment in enumerate(self.subcatchments):
            subcatchment.step(rainfall, pet, percolation[i, :], outflow[i, :])



