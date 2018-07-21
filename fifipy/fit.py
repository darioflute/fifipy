import numpy as np
from scipy import stats
from dask import delayed, compute
# from dask.distributed import Client
# from dask.distributed import LocalCluster
# import dask.multiprocessing

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
        
    return np.array(slopes)

def fitSlopeSpax(data):
    """Fit the slope of the 16 spectral pixels."""
    
    slopes = []
    for i in range(25):
        slope = fitSlope(data[:,:,i])
        slopes.append(slope)
        
    return np.array(slopes)

def fitAllSlopes(data):

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
    spaxels = [delayed(fitSlopeSpax)(data[:,:,i,:]) for i in range(16)]        
    
    #spectra = compute(* spaxels)
    spectra = compute(* spaxels, scheduler='processes')
    spectra = np.asarray(spectra)
    ns = np.shape(spectra)
    ss = [spectra[:,:,i] for i in range(ns[2])]
    return np.array(ss)

def computeSpectra(files):
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
        spectra = fitAllSlopes(flux)
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

def multiSlopes(data):
    ''' Compute slopes for each pixel and grating position using multiprocessing '''    
    import multiprocessing as mp

    # To avoid forking error in MAC OS-X
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    with mp.Pool(processes=mp.cpu_count()) as pool:
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