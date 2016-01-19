from ._weather_generator import AnnualHarmonicModel, RainfallSimulator
import numpy as np
from scipy import stats
import pandas


def estimate_precipitation_parameters(df, wet_threshold=0.2, period_length=14, precip_column='rainfall'):
    """
    Estimate precipitation probability and intensity estimates for wet days given a previous wet
    or dry day.

    This function modifies the incoming DataFrame df by adding various columns to aid with the
    calculating of the parameters.

    The function returns a new DataFrame containing several columns,
     - dayofyear - Julian day of the year for the centre of the period.
     - nsamples - total number of data points in the period.
     - p_wet_given_dry - the probability of wet day given it was dry the day before in this period.
     - p_wet_given_wet - the probability of wet day given it was wet the day before in this period.
     - lambda - the parameter for an exponential distribution of rainfall intensity on wet days.

    :param df: input DataFrame containing a 'rainfall' column and a DatetimeIndex index.
    :param wet_threshold: Threshold of rainfall equal to or above which counts as a wet day.
    :param period: Length of period, in days, to divide the year in to.
    :returns: pandas.DataFrame containing probabilities for each period of the year
    """
    # Julian day of year
    df['dayofyear'] = df.index.dayofyear
    # Flag which days of the year fall in to which period
    df['period'] = (df['dayofyear']-1) // period_length
    # Gather any overflow in to the last but one period
    if 365 % period_length != 0:
        lst_period = 365 / period_length
        df.ix[df.period == lst_period, 'period'] = lst_period - 1
    # Flag wet days
    df['wet'] = df[precip_column] >= wet_threshold
    # Flag whether the day before was wet
    df['wet_yesterday'] = df['wet'].shift(1)
    # Calculate W|D and W|W
    df['wet_given_dry'] = df['wet'][df['wet_yesterday'] == False]
    df['wet_given_wet'] = df['wet'][df['wet_yesterday'] == True]

    # Group by the periods
    grouped = df.groupby('period')

    p_data = []  # Empty container for the probability estimates
    for name, grp in grouped:
        # Calculate P(W|D) and P(W|W) for each group
        p_wet_dry = float(grp['wet_given_dry'].sum()) / np.sum(grp['wet_yesterday'] == False)
        p_wet_wet = float(grp['wet_given_wet'].sum()) / np.sum(grp['wet_yesterday'] == True)

        loc, scale = stats.expon.fit(grp['rainfall'][grp['wet']], floc=0)
        lmbda = 1 / scale

        p_data.append((name, name*period_length+period_length/2, len(grp), p_wet_dry, p_wet_wet, lmbda))

    # Create the new DataFrame with the period data
    p_df = pandas.DataFrame(p_data, columns=['period', 'dayofyear', 'nsamples', 'p_wet_given_dry',
                                             'p_wet_given_wet', 'lambda'])
    p_df = p_df.set_index('dayofyear')
    return p_df


def fit_harmonic_model(df, column, nfreq=2):
    """ Use scipy.optimize.leastsq to fit the data in column of the given pandas.DataFrame
    """
    from scipy.optimize import leastsq

    def fit_func(params, df):
        m = AnnualHarmonicModel(params[0], params[1::2], params[2::2])
        x = np.array([m.value(d) for d in df.index.values])
        return df[column] - x

    x0 = [df[column].iloc[0], ] + nfreq*[0.0, 0.0]
    params, status = leastsq(fit_func, x0, args=(df, ))
    return AnnualHarmonicModel(params[0], params[1::2], params[2::2])