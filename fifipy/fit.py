import numpy as np
from scipy import stats
from dask import delayed, compute


def fitSlope(data):
    """Fit the slope of a series of ramps."""

    saturationLimit = 2.7
    dtime = 1/250.  # Hz
    x = dtime * np.arange(32)
    rshape = np.shape(data)
    ngratings = rshape[0]
    nramps = rshape[1] // 32  # There are 32 readouts per ramp
    slopes = []
    
    for ig in range(ngratings):
        ramps = data[ig,:].reshape(nramps, 32)
        rslopes = []
        for j in [0,2]:  # Consider only the 1st and 3rd ramps
            ramp = ramps[j::4,:]  # One every 4 ramps
            ramp = np.nanmean(ramp, axis = 0)
            # Mask saturated values
            mask = ramp > saturationLimit
            # Mask first readouts and last one
            mask[0:3] = 1
            mask[-1] = 1
            if np.sum(~mask) > 5:  # Compute only if there are at least 5 pts
                slope, intercept, r_value, \
                    p_value, std_err = stats.linregress(x[~mask],ramp[~mask])
                rslopes.append(slope)
            else:
                rslopes.append(np.nan)
        slopes.append(rslopes[0]-rslopes[1])  # On - Off
        
    return slopes


def fitAllSlopes(data):
    
    pixels = [delayed(fitSlope)(data[:,:,i//25,i%25]) for i in range(16*25)]
    specs = compute(*pixels)
    s = np.asarray(specs)
    ns = np.shape(s)
    ng = ns[1]
    
    spectra = np.zeros((ng,16,25))
    for i in range(16*25):
        spectra[:,i//25,i%25] = s[i]
    
    return spectra