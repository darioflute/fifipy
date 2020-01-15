import numpy as np
from scipy import stats
from dask import delayed, compute
# from dask.distributed import Client
# from dask.distributed import LocalCluster
# import dask.multiprocessing

def fitSlope(data, telSim=False):
    """
        Fit the slope of a series of ramps for lab data.
        In this case, data are taken with the internal calibrator.
        A wheel with two holes allows to see the calibrator twice during a cycle.
        Only the 1st and 3rd ramps are retained, the other two being affected by the change in flux.
        The off position corresponds to the wheel position when the calibrator is not visible.
    """

    saturationLimit = 2.7
    dtime = 1/250.  # Hz
    x = dtime * np.arange(32)
    rshape = np.shape(data)
    ngratings = rshape[0]
    nramps = rshape[1] // 32  # There are 32 readouts per ramp
    slopes = []
    if telSim:
        indices = [1,3]
    else:
        indices = [0,2]
    
    for ig in range(ngratings):
        ramps = data[ig,:].reshape(nramps, 32)
        rslopes = []
        
        for j in indices:  # Consider only the 1st and 3rd ramps
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
        slopes.append(rslopes[1]-rslopes[0])  # On - Off
        
    return np.array(slopes)

def fitSlopeSky(data):
    """
        Fit the slope of a series of ramps for atmospheric data.
        In this case, data are taken on the sky and there is no off position for chopping.
        So, all the ramps are treated in the same way.
        The first ramp for each grating position is discarded.
    """

    saturationLimit = 2.7
    dtime = 1/250.  # Hz
    x = dtime * np.arange(32)
    rshape = np.shape(data)
    ngratings = rshape[0]
    nramps = rshape[1] // 32  # There are 32 readouts per ramp
    slopes = []
    
    for ig in range(ngratings):
        ramps = data[ig,:].reshape(nramps, 32)
        ramp = ramps[1:,:]  # Discard first ramp
        ramp = np.nanmean(ramp, axis = 0)
        # Mask saturated values
        mask = ramp > saturationLimit
        # Mask first readouts and last one
        mask[0:3] = 1
        mask[-1] = 1
        if np.sum(~mask) > 5:  # Compute only if there are at least 5 pts
            slope, intercept, r_value, \
                p_value, std_err = stats.linregress(x[~mask],ramp[~mask])
        else:
            slope = np.nan
        slopes.append(slope)  # Mean ramp slope
        
    return np.array(slopes)

def meanSlopeSky(data):
    """
    This routine computes the biweight mean of all the slopes between consecutive readings.
    """
    from fifipy.stats import biweightLocation as biwloc

    saturationLimit = 2.7
    dtime = 1/250.  # Hz
    #x = dtime * np.arange(32)
    rshape = np.shape(data)
    ngratings = rshape[0]
    nramps = rshape[1] // 32  # There are 32 readouts per ramp
    slopes = []
    
    for ig in range(ngratings):
        ramps = data[ig,:].reshape(nramps, 32)
        dr = []
        for ramp in ramps:
            mask = ramp > saturationLimit
            mask[0] = 1 # Discard first and last difference
            mask[-1] = 1
            if np.sum(~mask) > 2:
                ramp = ramp[~mask]
                dramp = ramp[1:]-ramp[:-1]
                ok = np.isfinite(dramp)
                dr.append(dramp[ok])
        if len(dr) > 0:
            dr = np.concatenate(dr)
            if len(dr) > 10:
                slope = biwloc(dr) / dtime
            else:
                slope = np.nan
        else:
            slope = np.nan
        slopes.append(slope)
    return np.array(slopes)

def fitSlopeSpax(data, telSim=False):
    """Fit the slope of the 16 spectral pixels."""
    
    slopes = []
    for i in range(25):
        slope = fitSlope(data[:,:,i], telSim)
        slopes.append(slope)
        
    return np.array(slopes)

def fitAllSlopes(data, telSim=False):

    #client = Client(threads_per_worker=4, n_workers=1)
    #client.cluster

    # Using all the pixels    
    #pixels = [delayed(fitSlope)(data[:,:,i//25,i%25]) for i in range(16*25)]
    #spectra = compute(*pixels, scheduler='processes')
    #spectra = np.asarray(spectra)
    #ns = np.shape(spectra) 
    #spectra = np.reshape(spectra,(ns[1],16,25))
    #return spectra
        
    # Using only spaxels
    spaxels = [delayed(fitSlopeSpax)(data[:,:,i,:], telSim) for i in range(16)]        
    
    #spectra = compute(* spaxels)
    spectra = compute(* spaxels, scheduler='processes')
    spectra = np.asarray(spectra)
    ns = np.shape(spectra)
    ss = [spectra[:,:,i] for i in range(ns[2])]
    return np.array(ss)

def computeSpectra(files, telSim=False):
    from fifipy.io import readData
    
    n=len(files)
    specs = np.zeros((n,16,25))
    gpos = np.zeros(n)
    
    for i,f in enumerate(files):
        if i%10 == 0:
            print (i//10, end='', flush=True)
        else:
            print ('.',end='', flush=True)
        aor, hk, gratpos, flux = readData(f)
        spectra = fitAllSlopes(flux, telSim)
        specs[i,:,:] = spectra[0,:,:]   
        gpos[i] = gratpos[0]
        

    #client.close()
    return gpos, specs


def computeSlopes(i,data):
    slopes = []
    for j in range(25):
        slope = fitSlope(data[:,:,j])
        slopes.append(slope)
        
    return i, np.array(slopes)

def computeSlopesSky(i,data):
    slopes = []
    for j in range(25):
        slope = fitSlopeSky(data[:,:,j])
        #slope = meanSlopeSky(data[:,:,j])  # Instead of fitting, compute the biweight mean of slope
        slopes.append(slope)
        
    return i, np.array(slopes)



def multiSlopes(data, sky=False):
    ''' Compute slopes for each pixel and grating position using multiprocessing '''    
    import multiprocessing as mp

    # To avoid forking error in MAC OS-X
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    with mp.Pool(processes=mp.cpu_count()) as pool:
        if sky:
            res = [pool.apply_async(computeSlopesSky, args=(i,data[:,:,i,:])) for i in range(16)]
        else:
            res = [pool.apply_async(computeSlopes, args=(i,data[:,:,i,:])) for i in range(16)]
        # print('res length is ',len(res))
        results = [p.get() for p in res]
    #pool.terminate() # Kill the pool once terminated (otherwise stays in memory)
    #results.sort()   not needed ...
    
    ds = np.shape(data)
    ng = ds[0]
    #print('number of gratings ',ng)

    spectra = np.zeros((ng,16,25))
    for r in results:
        i = r[0]
        slopes = r[1]
        for ig in range(ng):
            #print(i,np.shape(slopes))
            spectra[ig,i,:] = slopes[:,ig]

    return spectra


def multiSlopesSky(data, sky=True):
    ''' Compute slopes for each pixel and grating position using multiprocessing '''  
    ''' Just to order spectra in the same way as wavelength (ng, 25, 16)'''
    import multiprocessing as mp

    # To avoid forking error in MAC OS-X
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    with mp.Pool(processes=mp.cpu_count()) as pool:
        if sky:
            res = [pool.apply_async(computeSlopesSky, args=(i,data[:,:,i,:])) for i in range(16)]
        else:
            res = [pool.apply_async(computeSlopes, args=(i,data[:,:,i,:])) for i in range(16)]
        # print('res length is ',len(res))
        results = [p.get() for p in res]
    #pool.terminate() # Kill the pool once terminated (otherwise stays in memory)
    #results.sort()   not needed ...
    
    ds = np.shape(data)
    ng = ds[0]
    #print('number of gratings ',ng)

    spectra = np.zeros((ng,25,16))
    for r in results:
        i = r[0]
        slopes = r[1]
        for ig in range(ng):
            #print(i,np.shape(slopes))
            spectra[ig,:,i] = slopes[:,ig]

    return spectra


    
