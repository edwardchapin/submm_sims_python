
from scipy import interpolate
import numpy as np

def drawdist1d(pdf, bins, ndraw):
    """
    Draw random numbers given 1d probability density function

    This function calculates the cumulative distribution function
    (CDF) by integrating a PDF. It then uses linear interpolation of
    the CDF to map uniformly distributed random variates to the
    requested PDF.

    Parameters
    ----------
    pdf : array 
        probability density function evaluated in N bins
    bins : array
        N+1 bin boundaries
    ndraw : int
        number of samples to draw from the distribution


    Returns
    -------
    out : array
        Array of ndraw samples drawn from the PDF.

    """

    # Calculate the CDF
    n = pdf.size
    delta = bins[1:] - bins[:-1]
    cdf = np.zeros(n+1)
    cdf[0] = 0

    for i in range(n):
        cdf[i+1] = pdf[i]*delta[i] + cdf[i]
    
    cdf = cdf / cdf.max()

    # create an interpolation table
    cdf_interp = interpolate.interp1d(cdf, bins)

    # use the interpolation table to map uniform random variates to
    # the arbitrary probability density function using the CDF
    return cdf_interp(np.random.random(ndraw))
