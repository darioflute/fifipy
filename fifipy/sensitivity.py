from dask import delayed, compute
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def fitab(x,y,ey):
    """Slope fitting with error on y."""
    import numpy as np
    S = np.sum(1/ey**2)
    Sx = np.sum(x/ey**2)
    Sy = np.sum(y/ey**2)
    Sxx = np.sum((x/ey)**2)
    Sxy = np.sum(x*y/ey**2)
    Delta = S * Sxx - Sx * Sx
    a = (Sxx * Sy - Sx * Sxy) / Delta
    b = (S * Sxy - Sx * Sy) / Delta
    ea = np.sqrt( Sxx / Delta)
    eb = np.sqrt( S / Delta)
    return a, ea, b, eb


def fitb(y):
    """
    Slope fitting for equidistant x (assuming distance = 1)
    """
    import numpy as np
    # Eliminate NaN (which comes from saturated ramps)
    y = y[np.isfinite(y)]
    n = len(y)
    # Version from 1 to N, but should be x = dt * (i-1) ... not i*dt .. does it matter ?
    #x = np.arange(n) + 1
    #a = 2 * np.sum(y * (2*n+1-3*x)) / (n * (n-1))
    #b = 6 * np.sum(y * (2*x - n - 1)) / (n*(n+1)*(n-1))
    #chi2 = np.sum((a+b*x-y)**2)
    #vb = 12 * (chi2/(n-2)) /(n*(n+1)*(n-1))

    # Version from 0 to N-1
    if n > 5:
        x = np.arange(n)
        a = 2 * np.sum(y * (2*n-1-3*x))/(n*(n+1))
        b = 6 * np.sum(y * (2*x-n+1)) / (n*(n-1)*(n+1))
        chi2 = np.sum((a+b*x-y)**2)
        vb = 12 * chi2 / (n*(n+1)*(n-1)*(n-2))
        eb = np.sqrt(vb)
    else:
        b = np.nan
        eb = np.nan

    #vb = 12 * chi2 / (n * (n**2 + 2) * (n-2)) #Pipeline ?
    return b, eb

def fitCombinedSlopes(flux, ncycles, nodbeam):
    """Compute slopes and errors of sky obs from ramps"""
    import numpy as np
    from fifipy.stats import biweight
    # Select off positions
    mask = np.arange(128*ncycles) // 64 % 2
    if nodbeam == 'A':
        idx = mask == 1.0
    else:
        idx = mask == 0.0
    flux = flux[:,idx,:,:]
    #flux /= 3.63/65536.  # back to IDU (from V) to apply response later
    flux = flux /(3.63/65536.)# - 2**15 
    x = np.arange(64*ncycles) % 32
    xr = np.arange(32)#  * 1/(32*8) # time in seconds
    slopes = []
    eslopes = []
    exptime = []
    for f in flux:
        ramp = []
        eramp = []
        for i in range(32):
            fx = f[x == i,:,:]
            nx = np.sum(x == i)
            # Biweight mean
            r, er = biweight(fx[2:], axis=0)
            ramp.append(r)
            eramp.append(er/np.sqrt(nx-2))
            # Means
            #ramp.append(np.nanmedian(fx[2:], axis=0))  # skip first 2 ramps
            #eramp.append(np.nanstd(fx[2:], axis=0)/np.sqrt(nx-2)) # std of mean
        ramp = np.array(ramp)
        eramp = np.array(eramp)
                
        slope = np.empty((16,25))
        eslope = np.empty((16, 25))
        for i in range(16):
            for j in range(25):
                # Skip first 2 or 3 and last reading 
                a, ea, b, eb = fitab(xr[2:-1],ramp[2:-1,i,j], eramp[2:-1,i,j])
                slope[i, j] = b
                eslope[i, j] = eb
        slopes.append(slope)
        eslopes.append(eslope)
        exptime.append(ncycles*2/8.)  # 2 ramps per cycle with 1/8s integration
        
    slopes = np.array(slopes)
    eslopes = np.array(eslopes)
    exptime = np.array(exptime)
    return slopes, eslopes, exptime


def combineSlopes(b, eb):
    import numpy as np
    
    # Parameters 
    nsigma = 5
    s2n = 7
    
    b = np.array(b)
    eb = np.array(eb)
    bsn = b/eb
    # mask outliers
    med = np.nanmedian(b)
    db = np.abs(b - med)
    mad = 1.4826 * np.nanmedian(db)
    if mad > 0:
        mask = (db < nsigma * mad) & (bsn > s2n)
    else:
        mask = np.ones(len(b), dtype=bool)
    # Compute combined values
    if np.sum(mask) < 5:
        slope = np.nan
        eslope = np.nan
    else:
        w = 1 / eb**2
        #slope, sw = np.ma.average(b[mask], w[mask], returned=True)
        #eslope = 1/np.sqrt(sw)
        var = 1 / np.sum(w[mask])
        slope = np.sum(b[mask] * w[mask]) * var
        eslope = np.sqrt(var)
    return slope, eslope

def fitSpaxb(data):
    # Get slopes and error for each spaxel
    slope = np.empty(25)
    eslope = np.empty(25)
    for j in range(25):
        bb = []
        ebb = []
        ramps = data[:,:,j]
        for ramp in ramps:
            b, eb = fitb(ramp[2:-1])
            bb.append(b)
            ebb.append(eb)
            # Values weighted by the variance
        slope[j], eslope[j] = combineSlopes(bb, ebb)
    return (slope, eslope)


def fitAllb(data):
    # Multithreading here on spexels
    spaxels = [delayed(fitSpaxb)(data[:,:,i,:]) for i in range(16)]
    islope = compute(* spaxels, scheduler='processes')
    slope = [s for s,es in islope]
    eslope = [es for s,es in islope]
    return np.array(slope), np.array(eslope)
    

def fitSlopes(flux, ncycles, nodbeam):
    """Compute slopes and errors of sky obs from ramps"""
    import numpy as np
    # Select off positions
    mask = np.arange(128*ncycles) // 64 % 2
    if nodbeam == 'A':
        idx = mask == 1.0
    else:
        idx = mask == 0.0
    flux = flux[:,idx,:,:]
    #flux /= 3.63/65536.  # back to IDU (from V) to apply response later
    flux = flux /(3.63/65536.)# - 2**15 
    #x = np.arange(64*ncycles) % 32
    #xr = np.arange(32)#  * 1/(32*8) # time in seconds
    slopes = []
    eslopes = []
    exptime = []
    nramps = ncycles * 2
    
    for f in flux:
        data = f.reshape(nramps,32,16,25)
        slope, eslope = fitAllb(data)
        
        # Part multi-threaded
        #slope = np.empty((16,25))
        #eslope = np.empty((16, 25))
        #for i in range(16):
        #    for j in range(25):
        #        bb = []
        #        ebb = []
        #        ramps = f[:,i,j].reshape(nramps, 32)
        #        for ramp in ramps:
        #            b, eb = fitb(ramp[2:-1])
        #            bb.append(b)
        #            ebb.append(eb)
        #        # Values weighted by the variance
        #        slope[i, j], eslope[i, j] = combineSlopes(bb, ebb)
                
        slopes.append(slope)
        eslopes.append(eslope)
        
        exptime.append(nramps/8.)  # ramps per 1/8 s integration
        
    slopes = np.array(slopes)
    eslopes = np.array(eslopes)
    exptime = np.array(exptime)
    return slopes, eslopes, exptime

def fitLabSlopes(flux, ncycles, nodbeam, chop=False):
    """Compute slopes and errors of off ramps from lab data"""
    # Select first of each four ramps
    mask = (np.arange(128*ncycles) // 64 % 2) == 0 
    mask *= (np.arange(128*ncycles) // 32 % 2) == 0 
    idx = mask == 1
    flux /= (3.63/65536.)# - 2**15
    flux_ = flux[:,idx,:,:]
    slopes = []
    eslopes = []
    exptime = []
    nramps = ncycles
    
    for f in flux_:
        data = f.reshape(nramps,32,16,25)
        slope, eslope = fitAllb(data)
        slopes.append(slope)
        eslopes.append(eslope)        
        exptime.append(nramps/8.)  # ramps per 1/8 s integration
    slopes = np.array(slopes)
    eslopes = np.array(eslopes)
        
    if chop:
        mask = (np.arange(128*ncycles) // 64 % 2) == 1
        mask *= (np.arange(128*ncycles) // 32 % 2) == 0 
        idx = mask == 1
        flux_ = flux[:,idx,:,:]
        slopes1 = []
        eslopes1 = []
        for f in flux_:
            data = f.reshape(nramps,32,16,25)
            slope, eslope = fitAllb(data)
            slopes1.append(slope)
            eslopes1.append(eslope)        
        slopes1 = np.array(slopes1)
        eslopes1 = np.array(eslopes1)
        slopes = slopes1 - slopes
        # Compute error for one slope (not the chop)
        eslopes = np.sqrt((eslopes**2 + eslopes1**2)/2.)
        
    exptime = np.array(exptime)
    return slopes, eslopes, exptime


def fitCombinedLabSlopes(flux, ncycles, chop=False):
    """Combining ramps before fitting slope."""
    import numpy as np
    from fifipy.stats import biweight
    # Select off (flux1) and on (flux2) positions
    mask = (np.arange(128*ncycles) // 64 % 2) == 0 
    mask *= (np.arange(128*ncycles) // 32 % 2) == 0 
    idx = mask == 1
    flux1 = flux[:,idx,:,:] / (3.63/65536.)
    if chop:
        mask = (np.arange(128*ncycles) // 64 % 2) == 1 
        mask *= (np.arange(128*ncycles) // 32 % 2) == 0 
        idx = mask == 1
        flux2 = flux[:,idx,:,:]/ (3.63/65536.)
    # x position as function of ramp
    x = np.arange(32*ncycles) % 32
    xr = np.arange(32)#  * 1/(32*8) # time in seconds
    slopes = []
    eslopes = []
    exptime = []
    for f in flux1:
        ramp = []
        eramp = []
        for i in range(32):
            fx = f[x == i,:,:]
            nx = np.sum(x == i)
            # Biweight mean
            r, er = biweight(fx[1:], axis=0) # skip first
            ramp.append(r)
            eramp.append(er/np.sqrt(nx-1))
        ramp = np.array(ramp)
        eramp = np.array(eramp)
                
        slope = np.empty((16,25))
        eslope = np.empty((16, 25))
        for i in range(16):
            for j in range(25):
                # Skip first 2 or 3 and last reading 
                a, ea, b, eb = fitab(xr[2:-1],ramp[2:-1,i,j], eramp[2:-1,i,j])
                slope[i, j] = b
                eslope[i, j] = eb
        slopes.append(slope)
        eslopes.append(eslope)
        exptime.append(ncycles*2/8.)  # 2 ramps per cycle with 1/8s integration
        slopes = np.array(slopes)
        eslopes = np.array(eslopes)
        exptime = np.array(exptime)
    if chop:
        slopes2 = []
        eslopes2 = []
        for f in flux2:
            ramp = []
            eramp = []
            for i in range(32):
                fx = f[x == i,:,:]
                nx = np.sum(x == i)
                # Biweight mean
                r, er = biweight(fx[1:], axis=0) # skip first
                ramp.append(r)
                eramp.append(er/np.sqrt(nx-1))
            ramp = np.array(ramp)
            eramp = np.array(eramp)
                    
            slope = np.empty((16,25))
            eslope = np.empty((16, 25))
            for i in range(16):
                for j in range(25):
                    # Skip first 2 or 3 and last reading 
                    a, ea, b, eb = fitab(xr[2:-1],ramp[2:-1,i,j], eramp[2:-1,i,j])
                    slope[i, j] = b
                    eslope[i, j] = eb
            slopes2.append(slope)
            eslopes2.append(eslope)
        slopes2 = np.array(slopes2)
        eslopes2 = np.array(eslopes2)
        slopes = slopes2 - slopes
        eslopes = np.sqrt((eslopes**2 + eslopes2**2)) # error on combined slopes
        
    return slopes, eslopes, exptime


def fitCombinedLabSlopesNew(flux, ncycles, chop=False):
    """Combining ramps before fitting slope."""
    import numpy as np
    from fifipy.stats import biweight
    # Select off (flux1) and on (flux2) positions
    mask = (np.arange(128*ncycles) // 64 % 2) == 0 
    mask *= (np.arange(128*ncycles) // 32 % 2) == 0 
    idx = mask == 1
    flux1 = flux[:,idx,:,:] / (3.63/65536.)
    if chop:
        mask = (np.arange(128*ncycles) // 64 % 2) == 1 
        mask *= (np.arange(128*ncycles) // 32 % 2) == 0 
        idx = mask == 1
        flux2 = flux[:,idx,:,:]/ (3.63/65536.)
    # x position as function of ramp
    x = np.arange(32*ncycles) % 32
    xr = np.arange(32)#  * 1/(32*8) # time in seconds
    slopes = []
    eslopes = []
    exptime = []
    if chop:
        for f1, f2 in zip(flux1, flux2):
            ramp = []
            eramp = []
            for i in range(32):
                fx1 = f1[x == i,:,:]
                fx2 = f2[x == i,:,:]
                nx = np.sum(x == i)
                # Biweight mean
                r, er = biweight(fx2[1:]-fx1[1:], axis=0) # skip first
                ramp.append(r)
                eramp.append(er/np.sqrt(nx-1))
            ramp = np.array(ramp)
            eramp = np.array(eramp)
            slope = np.empty((16,25))
            eslope = np.empty((16, 25))
            for i in range(16):
                for j in range(25):
                    # Skip first 2 or 3 and last reading 
                    a, ea, b, eb = fitab(xr[2:-1],ramp[2:-1,i,j], eramp[2:-1,i,j])
                    slope[i, j] = b
                    eslope[i, j] = eb
            slopes.append(slope)
            eslopes.append(eslope)
            exptime.append(ncycles*2/8.)  # 2 ramps per cycle with 1/8s integration
    else:
        for f in flux1:
            ramp = []
            eramp = []
            for i in range(32):
                fx = f[x == i,:,:]
                nx = np.sum(x == i)
                # Biweight mean
                r, er = biweight(fx[1:], axis=0) # skip first
                ramp.append(r)
                eramp.append(er/np.sqrt(nx-1))
            ramp = np.array(ramp)
            eramp = np.array(eramp)
                    
            slope = np.empty((16,25))
            eslope = np.empty((16, 25))
            for i in range(16):
                for j in range(25):
                    # Skip first 2 or 3 and last reading 
                    a, ea, b, eb = fitab(xr[2:-1],ramp[2:-1,i,j], eramp[2:-1,i,j])
                    slope[i, j] = b
                    eslope[i, j] = eb
            slopes.append(slope)
            eslopes.append(eslope)
            exptime.append(ncycles*2/8.)  # 2 ramps per cycle with 1/8s integration
            slopes = np.array(slopes)
            eslopes = np.array(eslopes)
            exptime = np.array(exptime)
        
    return slopes, eslopes, exptime


def readRawData(fluxcaldir, direcs, combine=False):
    from fifipy.calib import waveCal
    from glob import glob as gb
    import numpy as np
    from fifipy.io import readData
    waves = []
    dwaves = []
    error = []
    flux = []
    exptime = []
    for direc in direcs:
        files = gb(fluxcaldir+direc+'/input/*.fits')
        nfiles = np.size(files)
        print('\n',direc, nfiles, end='')
        for i, file in enumerate(files):
            if i % 10 == 0:
                print('.', end='')
            aor, hk, gratpos, ramps = readData(file)
            # NaN data which are beyond the saturation threshold of 2.7V
            idx = ramps > 2.7
            ramps[idx] = np.nan
            detchan, order, dichroic, ncycles, nodbeam, filegpid, filenum = aor
            obsdate, telpos, pos, xy, angle, za, alti, wv = hk
            if combine:
                sl, esl, expt = fitCombinedSlopes(ramps, ncycles, nodbeam)                
            else:
                sl, esl, expt = fitSlopes(ramps, ncycles, nodbeam)
            for e in esl:
                error.append(e)
            for f in sl:
                flux.append(f)
            for g in gratpos:
                w,lw = waveCal(gratpos=g, order=order, array=detchan,
                               dichroic=dichroic,obsdate=obsdate)
                waves.append(w)
                dwaves.append(lw)
                exptime.append(expt)
    
    waves = np.array(waves)
    dwaves = np.array(dwaves)
    error = np.array(error)
    flux = np.array(flux)
    exptime = np.array(exptime)
    
    return waves, dwaves, error, flux, np.mean(exptime), obsdate


def readRpData(fitsfile):
    from astropy.io import fits
    import numpy as np

    with fits.open(fitsfile, memmap=False) as hdulist:
        next = len(hdulist)  # primary + grating splits
        header = hdulist[0].header
    
        detchan = header['DETCHAN']
        obsdate = header['DATE-OBS']
        dichroic = header['DICHROIC']
        exptime = header['EXPTIME']/ (next-1)  # Fraction of exptime in one grating

        if detchan == 'RED':
            #ncycles = header['C_CYC_R']
            start = header['G_STRT_R']
            step = header['G_SZUP_R']
            ngrat = header['G_PSUP_R']
            order = 1
        else:
            #ncycles = header['C_CYC_B']
            start = header['G_STRT_B']
            step = header['G_SZUP_B']
            ngrat = header['G_PSUP_B']
            order = header['G_ORD_B']
        gratpos = start+step*np.arange(ngrat)
        del header

        error = []
        flux = []
        for i in range(1, next):
            d = hdulist[i].data
            e = d['STDDEV']
            f = d['DATA']
            error.append(e[0,1:17,:,:])
            flux.append(f[0,1:17,:,:])
            del d,e,f

    return flux,error, gratpos, order, detchan, dichroic, obsdate, exptime


def readRamps(fluxcaldir, direcs):
    from fifipy.calib import waveCal
    from glob import glob as gb
    import numpy as np
    waves = []
    dwaves = []
    error = []
    flux = []
    exptime = []
    for direc in direcs:
        files = gb(fluxcaldir+direc+'/reduced/*RP0*.fits')
        nfiles = np.size(files)
        print('\n',direc, nfiles, end='')
        for i, file in enumerate(files):
            if i % 10 == 0:
                print('.', end='')
            sl,esl,gratpos,order,detchan,dichroic,obsdate,expt = readRpData(file)
            for e in esl:
                error.append(e)
            for f in sl:
                flux.append(f)
            for g in gratpos:
                w,lw = waveCal(gratpos=g, order=order, array=detchan,dichroic=dichroic,obsdate=obsdate)
                waves.append(w)
                dwaves.append(lw)
                exptime.append(expt)
    
    waves = np.array(waves)
    dwaves = np.array(dwaves)
    error = np.array(error)
    flux = np.array(flux)
    exptime = np.array(exptime)
    
    return waves, dwaves, error, flux, exptime, obsdate


def readResponse(array, dichroic, order, obsdate):
    import os, re
    from astropy.io import fits
    from scipy.interpolate import interp1d
    path0, file0 = os.path.split(__file__)
    # Extract month, year, and day from date
    parts = re.split('-|T|:', obsdate)
    odate = int(parts[0]+parts[1]+parts[2])
    if odate <  20180101:
        file = path0+'/data/Response_'+array+'_D'+str(dichroic)+'_Ord'+str(order)+'_20230505v1.fits'
    else:
        file = path0+'/data/Response_'+array+'_D'+str(dichroic)+'_Ord'+str(order)+'_20230505v2.fits'
    hdl = fits.open(file)
    data = hdl['PRIMARY'].data
    hdl.close()
    wr = data[0,:]
    fr = data[1,:]
    response = interp1d(wr,fr,fill_value='extrapolate')
    return response


def applyResponse(waves, fluxes, detchan, order, dichroic, obsdate):
    response = readResponse(detchan, dichroic, order, obsdate)
    for i in range(16):
        for j in range(25):
            fluxes[:,j,i] /= response(waves[:,j,i])    





def computeSensitivity(responseDir, array, order, dichroic, waves, dwaves, 
                       error, exptime, obsdate,raw=True,delregion=True,
                       ymax=4, superflat=True,applyresponse=True):
    from matplotlib import rcParams
    #from scipy.signal import medfilt
    from fifipy.stats import biweight 
    rcParams['font.family']='STIXGeneral'
    rcParams['font.size']=18
    rcParams['mathtext.fontset']='stix'
    rcParams['legend.numpoints']=1
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits
    #from scipy.interpolate import LSQUnivariateSpline
    from fifipy.calib import readFlats

    wflat, specflat, especflat, spatflat= readFlats(array, order, dichroic, 
                                                    obsdate, silent=True)
    #response = readResponse(array, dichroic, order, obsdate)
    error_ = error.copy()
    if applyresponse:
        response = readResponse(array, dichroic, order, obsdate)
        for j in range(25):
            for i in range(16):
                error_[:,i,j] /= response(waves[:,j,i])

    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(1,1,1)

    # Regions to mask
    if array == 'Blue':
        #good = [1,2,3,6,7,8,11,12,13,16,17,20,21,22]
        if order == 1:
            regions = [
                [50.6,50.7],
                [51,51.2],
                [51.4,51.8],
                [53,53.3],
                [54.27,54.52],
                [55,55.2],
                [55.9,57],
                [57.4,58],
                [58.4,59.1],
                [59.9,60.1],
                [70.3,72.75],
                [74.25,76.5],
                [77.6,79.5],
                [80.9,83.6],
                [84.5,85],
                [89.5,90.4],
                [92.5,93],
                [93.2,93.5],
                [93.9,96],
                [98.0,98.8],
                [99,100],
                [100.5,101.5],
                [103.8,104.2]
                ]
        else:
            regions = [
                [50.6,50.7],
                [51,51.2],
                [51.4,51.8],
                [53,53.3],
                [54.2,54.6],
                [54.8,55.3],
                [55.9,57],
                [57.4,58],
                [58.4,59.1],
                [59.8,60.2]
               ]            
    else:
        #good = [0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23]
        if dichroic == 105:
            regions = [
                    [117,117.6],
                    [119.6,120],
                    [120.6,122.5],
                    [124.5,128.5],
                    [131.7,140],
                    #[131.7,131.9],
                    #[132.4,132.5],
                    #[134.5,135.5],
                    #[136,137],
                    #[137.8,139.4],
                    [142.8,143.25],
                    [144,145.5],
                    #[144,145],
                    [146.5,147.5],
                    [148.6,149],
                    [155.5,157],
                    [160,161],
                    [165.3,165.5],
                    [166.7,167.2],
                    [169.8,170.2],
                    [171.1,171.3],
                    [174,175],
                    [178.5,180]  
                    ]
        else:
            regions = [
                    [117,117.6],
                    [119.6,120],
                    #[120.6,122.5],
                    #            [124.5,128.5],
                    #            [131.7,140],
                    [131.7,131.9],
                    [132.4,132.5],
                    [134.5,135.5],
                    [136,137],
                    [137.8,139.4],
                    [142.8,143.25],
                    #            [144,145.5],
                    [144,145],
                    [146.5,147.5],
                    [148.6,149],
                    [155.5,157],
                    [160,161],
                    [165.3,165.5],
                    [166.7,167.2],
                    [169.8,170.2],
                    [171.1,171.3],
                    [174,175],
                    [178.5,180]  
                    ]
            

    good, = np.where((spatflat > 0.8) & (spatflat < 1.2) )
    print('\n Good pixels ', good)
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
    colorst = [colormap(i) for i in np.linspace(0, 0.9,25)]       

    c = 299792458.e+6 # um/s
    wtots = []
    stots = []
    
    #if not delregion:
    #    from fifipy.calib import readWindowTransmission
    #    wwt, twt = readWindowTransmission()
    
    for j in good:
        wt = []
        st = []
        for i in range(16):
            w = waves[:,j,i]
            dw = dwaves[:,j,i]
            dnu = c /w * dw/w
            sf = np.interp(waves[:,j,i], wflat, specflat[:,j,i])
            if raw == True:
                s =  error_[:,i,j]/dnu*np.sqrt(exptime/900)/sf/spatflat[j]
            else:
                s =  error_[:,i,j//5,j%5]/dnu*np.sqrt(exptime/900)/spatflat[j]/sf
            idx = np.ones(np.size(w), dtype = bool)
            if delregion:
                for r  in regions:
                    m = (w < r[0]) | (w > r[1])
                    idx *= m
            wt.append(w[idx])
            st.append(s[idx])
        wt = np.concatenate(wt)
        st = np.concatenate(st)
        wtots.append(wt)
        stots.append(st)
        
    if array == 'Red':
        w1 = 140
        w2 = 160
    else:
        if order == 1:
            w1 = 70
            w2 = 95
        else:
            w1 = 60
            w2 = 68
            
    ax.set_ylim([0,ymax])
    wtot = np.concatenate(wtots)
    stot = np.concatenate(stots)
    
    if superflat:
        # Superflat data
        idx = (wtot > w1) & (wtot < w2)
        stotmed = np.nanmedian(stot[idx])
        # Flats
        #stots2 = []
        for wt, st, j in zip(wtots, stots, good):
            idx = (wt > w1) & (wt < w2)
            stmed = np.nanmedian(st[idx])
            st *= stotmed/stmed
            ax.plot(wt, st , '.', markersize=0.8, color=colorst[j], label=str(j))
        # Recompute the total
        stot = np.concatenate(stots)
    else:
        for wt, st, j in zip(wtots, stots, good):
            ax.plot(wt, st , '.', markersize=0.8, color=colorst[j], label=str(j))

    ax.legend(markerscale=10., scatterpoints=1, fontsize=10)
    
    # Reject NaNs
    idx = np.isfinite(stot)
    wtot = wtot[idx]
    stot = stot[idx]
    idx = np.argsort(wtot)
    wtot = wtot[idx]
    stot = stot[idx]
    u, indices = np.unique(wtot, return_index=True)
    wtot = wtot[indices]
    stot = stot[indices]
    print(np.nanmedian(stot))
    
    # Reject very low values
    if applyresponse:
        idx = stot > 0.01
        wtot = wtot[idx]
        stot = stot[idx]
    x= wtot
    y= stot
    
    if array == 'Blue':
        if order == 1:
            dwr = 0.2
            wr = np.arange(np.nanmin(wtot)+dwr,np.nanmax(wtot)-dwr,dwr)
            ax.set_xlim(60,130)
        else:
            dwr = 0.2
            wr = np.arange(np.nanmin(wtot)+dwr,np.nanmax(wtot)-dwr,dwr)
            ax.set_xlim(48,78)
    else:
        dwr = 0.5
        wr = np.arange(np.nanmin(wtot)+dwr,np.nanmax(wtot)-dwr,dwr)
        ax.set_xlim(110,210)
    # Using biweight ...
    sr  = np.empty(len(wr))
    esr = np.empty(len(wr))
    for i, w in enumerate(wr):
        idx = (np.abs(x-w) < dwr)  # oversampling is 2
        sr[i],esr[i] = biweight(y[idx])

    ax.plot(wr,sr,color='lime')
    ax.plot(wr,sr+esr, color='lime')
    ax.plot(wr,sr-esr, color='lime')
    ax.grid()
    plt.show()

    data = []
    data.append(wr)
    data.append(sr)
    data.append(esr)
    data = np.array(data)
    hdr = fits.Header()
    hdr['DETCHAN'] = array
    hdr['ORDER'] = order
    hdr['DICHOIC'] = dichroic
    hdr['XUNIT'] = 'microns'
    hdr['YUNIT'] = 'Jy'
    hdu = fits.PrimaryHDU(data, header=hdr)
    hdul = fits.HDUList([hdu])
    hdul.writeto(responseDir+'/Sensitivity_'+array+'_D'+str(dichroic)+'_Ord'+str(order)+'.fits',
                 overwrite=True)
    return

# Using lab data
    
def readLabData(fluxcaldir, direcs, chop=False, combine=False):
    from fifipy.calib import waveCal
    from glob import glob as gb
    import numpy as np
    from fifipy.io import readData
    waves = []
    dwaves = []
    error = []
    flux = []
    exptime = []
    for direc in direcs:
        files = gb(fluxcaldir+direc+'/*.fits')
        nfiles = np.size(files)
        print('\n',direc, nfiles, end='')
        for i, file in enumerate(files):
            if i % 10 == 0:
                print('.', end='')
            aor, hk, gratpos, ramps = readData(file)
            # NaN data which are beyond the saturation threshold of 2.7V
            idx = ramps > 2.7
            ramps[idx] = np.nan
            detchan, order, dichroic, ncycles, nodbeam, filegpid, filenum = aor
            obsdate, telpos, pos, xy, angle, za, alti, wv = hk
            if combine:
                sl, esl, expt = fitCombinedLabSlopes(ramps, ncycles, chop)
            else:
                sl, esl, expt = fitLabSlopes(ramps, ncycles, nodbeam, chop)
            for e in esl:
                error.append(e)
            for f in sl:
                flux.append(f)
            for g in gratpos:
                w,lw = waveCal(gratpos=g, order=order, array=detchan,
                               dichroic=dichroic,obsdate=obsdate)
                waves.append(w)
                dwaves.append(lw)
                exptime.append(expt)
    
    waves = np.array(waves)
    dwaves = np.array(dwaves)
    error = np.array(error)
    flux = np.array(flux)
    exptime = np.array(exptime)
    
    return waves, dwaves, error, flux, np.mean(exptime), obsdate


# Routines to remove the correlated noise between ramps of pixels in the same spaxel
def fitramp(x, y):
    """
    Slope fitting for equidistant x (assuming distance = 1)
    """
    import numpy as np
    # Eliminate NaN (which comes from saturated ramps)
    S = len(y)
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxy = np.sum(x*y)
    Sxx = np.sum(x*x)
    D = S * Sxx - Sx * Sx
    a = (Sxx * Sy - Sx * Sxy) / D
    b = (S * Sxy - Sx * Sy) / D
    return a, b


def subCorrNoise(ramp):
    from fifipy.stats import biweight
    import numpy as np
    xr = np.arange(32)
    ramp = np.float32(ramp) / 2**16 + 0.5 # normalize to 0-1
    ramp *= 3.63 # Transform into V
    for spaxel in range(25):
        res = []
        for i in range(18):
            ri = ramp[:,i,spaxel]
            mask = ri < 2.7
            mask[:2] = False
            mask[-1:] = False
            a, b = fitramp(xr[mask], ri[mask])
            res.append(ri - a - b * xr)
        res = np.array(res)
        xsignal, s = biweight(res, axis=0)
        # Get rid of any linear trend or bias
        a, b = fitramp(xr[mask], xsignal[mask])
        xsignal -= a + b * xr        
        for i in range(18):
            ramp[:, i, spaxel] -= xsignal
    ramp *= 65536 / 3.63  # Back to ADU and unsigned integer
    ramp -= 2**15
    # Limit inside unsigned int16 integer limits
    ramp[ramp < -32768] = -32768 
    ramp[ramp > 32767] = 32767
    return ramp.astype(int)

def updateRamps(fitsfile):
    from astropy.io import fits
    from dask import delayed, compute        
    
    with fits.open(fitsfile, mode='update') as hdl:
        scidata = hdl[1].data
        data = scidata.DATA
        header = hdl[0].header
        detchan = header['DETCHAN']
        if detchan == 'RED':
            ncycles = header['C_CYC_R']
            ngrat = header['G_PSUP_R']
        else:
            ncycles = header['C_CYC_B']
            ngrat = header['G_PSUP_B']
        ramps = data.reshape(ngrat*ncycles*4,32,18,26)
        # MultiProcess
        cramps = [delayed(subCorrNoise)(ramp) for ramp in ramps]
        iramp = compute(* cramps, scheduler='processes')
        for i, ramp  in enumerate(iramp):
            ramps[i] = ramp                                
        scidata.DATA = ramps.reshape(ngrat*ncycles*4*32, 18, 26)            

def subReadoutNoise(fitsfile):
    """
    The code subtracts from all pixels the 0-th pixel which
    contains readout noise common to all the pixels.
    Its removal improves the quality of the data reduction.
    
    Parameters
    ----------
    fitsfile : input raw file for FIFI-LS

    Returns
    -------
    None.

    """
    from astropy.io import fits
    import numpy as np
    
    with fits.open(fitsfile, mode='update') as hdl:
        scidata = hdl[1].data
        data = scidata.DATA
        header = hdl[0].header
        header['HISTORY'] = 'Subtracted open pixel from ramps'
        fdata = np.float32(data)
        openpix = fdata[:,0,:25] + 2**15  
        restpix = fdata[:,1:,:25] + 5000
        openpix=np.tile(np.expand_dims(openpix,1),(1,17,1))
        restpix -= openpix
        restpix[restpix<-32768] = -32768
        restpix[restpix>32767] = 32767
        fdata[:,1:,:25] = restpix
        scidata.DATA = np.int16(fdata)
