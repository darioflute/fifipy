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


def biweight(data, axis=None):
    """Biweight estimation of location and scale according to
    Understanding Robust and Exploratory Data Analysis, Hoaghley, Mosteller, & Tukey,
    formula 4. Tuning constant as at page 417. Location on page 421.
    See also Beers, Flynn, & Gebhardt, 1990, AJ 100:32.
    """
    #from astropy.stats.funcs import median_absolute_deviation
    import numpy as np
    np.seterr(divide='ignore', invalid='ignore')
    
    c1 = 6.0
    c2 = 9.0
    data = np.asarray(data)
    M = np.nanmedian(data, axis=axis, keepdims=True)
    S = np.nanmedian(np.abs(data-M), axis=axis, keepdims=True)
        
    if np.nanmean(S) < 0.0001:
        return M.squeeze(), S.squeeze()
    
    u = (data - M) / (c1 * S)
    mask = np.abs(u) > 1.0
    u = (1 - u * u) ** 2
    u[mask] = 0
    s3 = np.nansum((data - M) * u, axis=axis)
    s4 = np.nansum(u, axis=axis)
    if axis is not None:
        s34 = np.expand_dims(s3 / s4, axis=axis)
    else:
        s34 = s3 / s4
    M += s34
    
    u = (data - M ) / (c2 * S)
    mask = np.abs(u) >= 1.
    u = u * u
    u1 = 1 - u
    u1[mask] = 0 
    s1 = np.nansum((data - M)**2 * u1**4, axis=axis)
    s2 = np.nansum((1 - 5 * u) * u1, axis=axis)
    ndata = np.ma.size(data, axis=axis)
    S = np.sqrt( s1 / (ndata-1) ) / ( np.abs(s2) / ndata)
        
    return M.squeeze(), S
