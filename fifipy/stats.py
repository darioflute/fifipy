""" Statistical functions. """

def biweightLocation(data,axis=None):
    """ Biweight estimator (location) """
    from astropy.stats.funcs import median_absolute_deviation
    import numpy as np
    np.seterr(divide='ignore', invalid='ignore')
    c = 6.0
    data = np.asanyarray(data)
    M = np.nanmedian(data, axis=axis, keepdims=True)
    # set up the differences
    #d = data - M
    d = data - M
    # set up the weighting
    mad = median_absolute_deviation(data, axis=axis,ignore_nan=True)
    if axis is not None:
        mad = np.expand_dims(mad, axis=axis)
    u = d / (c * mad)
    # now remove the outlier points
    mask = (np.abs(u) >= 1)
    u = (1 - u ** 2) ** 2
    u[mask] = 0
    return M.squeeze() + np.nansum(d * u, axis=axis) / np.nansum(u, axis=axis)
