from libc.math cimport exp, atan, tan, sin, acos, cos, fabs, sqrt, M_PI

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
            percolation[i] = 0.0

            # Effective rainfall = rainfall less PET
            effective_rainfall = rainfall[i] - pet[i]

            # The effective_rainfall variable is consumed (reduced) by the following processes to determine if there
            # is a net saturated percolation at the end.

            if effective_rainfall > 0.0:
                # Wetting
                # First calculate direct percolation, this proportion bypasses the store entirely
                direct_percolation = effective_rainfall * self.direct_percolation
                percolation[i] += direct_percolation
                effective_rainfall -= direct_percolation

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
            raise ValueError("Invalid value for linear storage constant. Must be >= 0.0")

    cpdef step(self, double[:] inflow, double[:] outflow):
        """
        """

        cdef double b
        try:
            b = exp(-1.0/self.linear_storage_constant)
        except ZeroDivisionError:
            b = 0.0

        cdef int i
        cdef int n = self.previous_outflow.shape[0]
        for i in range(n):
            # Calculate outflow from this store as the average over the timestep
            outflow[i] = inflow[i] - self.linear_storage_constant*(inflow[i] - self.previous_outflow[i])*(1.0 - b)
            # Record the end of timestep flow ready for the next timestep
            if self.previous_outflow[i] < 1e-9:
                self.previous_outflow[i] = 1e-8
            else:
                self.previous_outflow[i] = inflow[i] - (inflow[i] - self.previous_outflow[i])*b

    property size:
        def __get__(self):
            return self.previous_outflow.shape[0]


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
        cdef double ZERO = 1e-9
        cdef double Q2
        cdef int i
        cdef int n = self.previous_outflow.shape[0]
        for i in range(n):

            if self.previous_outflow[i] > 0.0:
                if inflow[i] < -ZERO:
                    # Case (b)
                    a = atan(sqrt(-self.previous_outflow[i]/inflow[i]))
                    a -= sqrt(-inflow[i]/self.nonlinear_storage_constant)
                    if a > 0.0:
                        Q2 = -inflow[i]*tan(a)**2
                    else:
                        Q2 = 0.0
                elif inflow[i] > ZERO:
                    # Case (c)
                    a = sqrt(self.previous_outflow[i]) - sqrt(inflow[i])
                    a /= sqrt(self.previous_outflow[i]) + sqrt(inflow[i])

                    b = -2.0*T*sqrt(inflow[i]/self.nonlinear_storage_constant)

                    Q2 = inflow[i]*((1 + a*exp(b)) / (1 - a*exp(b)))**2
                else:
                    # Case (a)  inflow[i] == 0.0
                    Q2 = self.nonlinear_storage_constant
                    Q2 /= (sqrt(self.nonlinear_storage_constant/self.previous_outflow[i]) + T)**2
            else:
                # Case (d) - less than zero initial outflow
                # Ensure the sqrt is done with a +ve number (or zero)

                V = -sqrt(fabs(self.previous_outflow[i])*self.nonlinear_storage_constant)
                V += inflow[i]*T

                if V > 0.0:
                    # New volume means flow will return after a period of time, t
                    t = T - V/inflow[i]

                    # Wilby (1994) is unclear what 'a' should be in case (d)
                    # Testing reveals it should be unity.
                    a = 1.0
                    b = -2.0*(T - t)*sqrt(inflow[i]/self.nonlinear_storage_constant)

                    Q2 = inflow[i]*((1 - a*exp(b)) / (1 + a*exp(b)))**2
                else:
                    Q2 = 0.0

            # Wilby 1994 gives no analytical solution to the mean flow in a timestep, therefore set the outflow
            # for this timestep as the average of the outflow at beginning and end of timestep
            outflow[i] = (self.previous_outflow[i]+Q2)/2.0
            # Update previous outflow for next time-step
            self.previous_outflow[i] = Q2

    property size:
        def __get__(self):
            return self.previous_outflow.shape[0]

cdef class SubCatchment:
    cdef public basestring name
    cdef SoilMoistureDeficitStore _soil
    cdef LinearStore _linear
    cdef NonLinearStore _nonlinear
    cdef float area

    def __init__(self, area, double[:] initial_upper_deficit, double[:] initial_lower_deficit,
                 double[:] initial_linear_outflow, double[:] initial_nonlinear_outflow, **kwargs):
        self.area = area
        self.name = kwargs.pop('name', '')
        self._soil = SoilMoistureDeficitStore(initial_upper_deficit, initial_lower_deficit, **kwargs)
        self._linear = LinearStore(initial_linear_outflow, **kwargs)
        self._nonlinear = NonLinearStore(initial_nonlinear_outflow, **kwargs)

        if self._linear.size != self._nonlinear.size:
            raise ValueError('Initial conditions for linear and non-linear store are different sizes.')

    cpdef int step(self, double[:] rainfall, double[:] pet, double[:] percolation, double[:] outflow) except -1:
        """ Step the subcatchment one timestep
        """
        cdef int i
        cdef int n = self.size

        self._soil.step(rainfall, pet, self.area, percolation)
        self._linear.step(percolation, outflow)
        for i in range(n):
            outflow[i] *= self.area

        self._nonlinear.step(outflow, outflow)
        return 0

    property size:
        def __get__(self):
            return self._linear.size

    property soil_store:
        def __get__(self):
            return self._soil

    property linear_store:
        def __get__(self):
            return self._linear

    property nonlinear_store:
        def __get__(self):
            return self._nonlinear

cdef class Catchment:
    def __init__(self, subcatchments, name=''):
        if not all(sc.size == subcatchments[0].size for sc in subcatchments):
            raise ValueError('Subcatchments must all be the same size.')

        self.subcatchments = list(subcatchments)
        self.name = name

    cpdef int step(self, double[:] rainfall, double[:] pet, double[:, :] percolation, double[:, :] outflow) except -1:
        """ Step the catchment one timestep
        """
        cdef int i
        cdef SubCatchment subcatchment
        for i, subcatchment in enumerate(self.subcatchments):
            subcatchment.step(rainfall, pet, percolation[i, :], outflow[i, :])

        return 0

    property size:
        def __get__(self):
            return self.subcatchments[0].size

cpdef double declination(int day_of_year):
    """
    Declination, the angular position of the sun at solar noon (i.e., when the sun is on the
    local meridian) with respect to the plane of the equator, north positive; −23.45 deg ≤ δ ≤ 23.45 deg

    Reference,
        Duffie & Beckman, Solar Engineering of Thermal Processes (Fourth Edition), 2013
            Equation 1.6.1a

    """
    return M_PI*(23.45/180.0) * sin(2*M_PI*(284 + day_of_year)/365)


cpdef double sunset_hour_angle(double latitude, double declination):
    """

    Reference,
        Duffie & Beckman, Solar Engineering of Thermal Processes (Fourth Edition), 2013
            Equation 1.6.10
    """
    return acos(-tan(latitude)*tan(declination))


cpdef double daily_extraterrestrial_radiation(int day_of_year, double latitude, double declination,
                                              double sunset_hour_angle):
    """
    Compute daily total extraterrestrial radiation in MJ m-2 day-1

    Reference,
        Duffie & Beckman, Solar Engineering of Thermal Processes (Fourth Edition), 2013
            Equation 1.10.3
    """
    cdef double solar_constant = 1367  # W/m2

    cdef double H = 24 * 3600 * solar_constant / M_PI
    H *= 1 + 0.033*cos(2*M_PI*day_of_year/365)
    H *= cos(latitude)*cos(declination)*sin(sunset_hour_angle) + sunset_hour_angle*sin(latitude)*sin(declination)
    return H/1e6  # Convert to MJ m-2 day-1


cpdef pet_oudin(int day_of_year, double latitude, double[:] temperature, double[:] pet):
    """ Estimate PET using the formula propsed by Oudin (2005)

    This model uses daily mean temperature to estimate PET based on the Julien day of year and latitude. The later
    are used to estimate extraterrestrial solar radiation.

    Reference,
        Ludovic Oudin et al, Which potential evapotranspiration input for a lumped rainfall–runoff model?:
        Part 2—Towards a simple and efficient potential evapotranspiration model for rainfall–runoff modelling,
        Journal of Hydrology, Volume 303, Issues 1–4, 1 March 2005, Pages 290-306, ISSN 0022-1694,
        http://dx.doi.org/10.1016/j.jhydrol.2004.08.026.
        (http://www.sciencedirect.com/science/article/pii/S0022169404004056)

    """
    cdef int i
    cdef double dec, w, R
    cdef double gamma = 2.45 # the latent heat flux (MJ kg−1)
    cdef double rho = 1000.0 # density of water (kg m-3)

    # Calculate the extraterrestrial radiation for all temperature estimates
    dec = declination(day_of_year)
    w = sunset_hour_angle(latitude, dec)
    R = daily_extraterrestrial_radiation(day_of_year, latitude, dec, w)

    for i in range(temperature.shape[0]):
        if temperature[i] > -5.0:
            pet[i] = R/(gamma*rho)
            pet[i] *= (temperature[i] + 5.0)/100.0
            pet[i] *= 1000.0  # m/day -> mm/day
        else:
            pet[i] = 0.0


cdef class OudinCatchment:
    """
    """
    def __init__(self, subcatchments, double latitude):
        self.latitude = latitude
        self.subcatchments = list(subcatchments)

    cpdef int step(self, int day_of_year, double[:] rainfall, double[:] temperature, double[:] pet,
               double[:, :] percolation, double[:, :] outflow) except -1:
        """ Step the catchment one timestep

        This method overloadds Catchment.step to pre-calculate PET
        """
        cdef int i
        cdef SubCatchment subcatchment
        # Calculate PET first
        pet_oudin(day_of_year, self.latitude, temperature, pet)

        for i, subcatchment in enumerate(self.subcatchments):
            subcatchment.step(rainfall, pet, percolation[i, :], outflow[i, :])

        return 0

    property size:
        def __get__(self):
            return self.subcatchments[0].size



