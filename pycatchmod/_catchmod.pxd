from libc.math cimport exp, atan, tan, sin, acos, cos, fabs, sqrt, M_PI

cdef class Catchment:
    cdef public basestring name
    cdef public list subcatchments

    cpdef int step(self, double[:] rainfall, double[:] pet, double[:, :] percolation, double[:, :] outflow) except -1


cdef class OudinCatchment:
    """
    """
    cdef list subcatchments
    cdef public double latitude

    cpdef int step(self, int day_of_year, double[:] rainfall, double[:] temperature, double[:] pet,
               double[:, :] percolation, double[:, :] outflow) except -1