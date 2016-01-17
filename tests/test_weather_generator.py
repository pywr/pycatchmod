from pycatchmod.weather_generator import AnnualHarmonicModel, RainfallSimulator
import numpy as np
import pytest


def test_annual_harmonic_model_init():
    """
    Test the different methods of initialising AnnualHarmonicModel
    """

    # None of these should raise an error
    AnnualHarmonicModel(5.0)
    AnnualHarmonicModel(5.0, np.array([1.0, ]), np.array([2.5, ]))
    AnnualHarmonicModel(5.0, np.array([1.0, 2.0]), np.array([2.5, 1.0]))

    # Erroneous init arguments ...
    with pytest.raises(ValueError):
        AnnualHarmonicModel(5.0, np.array([1.0, ]), np.array([2.5, 1.0]))
        AnnualHarmonicModel(5.0, np.array([1.0, ]))
        AnnualHarmonicModel(5.0, phase=np.array([1.0, ]))


def test_annual_harmonic_model_values():
    """
    Test the AnnualHarmonicModel.value method

    """

    # This model should return 5.0 for all days_of_year
    m = AnnualHarmonicModel(5.0)
    for i in range(1, 5):
        np.testing.assert_allclose(m.value(float(i)), 5.0)

    # 1st order harmonic with zero phase shift
    m = AnnualHarmonicModel(5.0, np.array([1.0, ]), np.array([0.0, ]))
    for i in range(1, 5):
        np.testing.assert_allclose(m.value(float(i)), 5.0 + np.cos(i*2*np.pi/365))

    # 1st order harmonic with pi/2 phase shift (i.e. now a sine harmonic)
    m = AnnualHarmonicModel(5.0, np.array([1.0, ]), np.array([np.pi/2, ]))
    for i in range(1, 5):
        np.testing.assert_allclose(m.value(float(i)), 5.0 - np.sin(i*2*np.pi/365))


def test_rainfall_simulator():
    """
    Simple test RainfallSimulator to produce a step

    """
    N = 1000
    wet_dry = AnnualHarmonicModel(0.4)  # 40% chance of wet if it was dry
    wet_wet = AnnualHarmonicModel(0.6)  # 60% chance of wet if it was wet
    intensity = AnnualHarmonicModel(1.0)

    rainfall = np.empty(N)

    sim = RainfallSimulator(N, wet_dry, wet_wet, intensity)
    sim.step(1, rainfall)

    # Crude test that the number of wet days is roughly the expected 40%
    np.testing.assert_allclose(0.4, np.count_nonzero(rainfall)/float(N), rtol=0, atol=0.02)
