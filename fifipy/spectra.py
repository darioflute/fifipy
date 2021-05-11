from scipy.interpolate import LSQUnivariateSpline
# from scipy.interpolate import interp1d
import numpy as np

def getResolution(mode, l):
    if mode == 'R':
        return 0.062423 * l*l - 6.6595 * l + 647.65
    elif mode == 'B1':
        return 0.16864 * l*l - 22.831 * l + 1316.6
    elif mode == 'B2':
        #return 1.9163 * l*l - 187.35 * l + 5496.9  # Colditz's values
        return 2.1046 * l*l - 202.72 * l + 5655.8  # new values
    else:
        print('Invalid mode entered. Valid modes are R, B1, and B2')
        return l*0
    
def computeSplineFits(w1, dw1, s1, mode, wmin=None, wmax=None, delta=0.5):
    
    if wmin is None or wmax is None:
        wmin=0
        wmax=250
        for ispex in range(16):
            for ispax in range(25):
                x1 = w1[:,ispax,ispex]
                f1 = s1[:,ispex, ispax]
                idx = np.isfinite(f1)
                if np.sum(idx) > 2:
                    xmin = np.nanmin(x1[idx])
                    xmax = np.nanmax(x1[idx])
                    if xmin > wmin: wmin = xmin
                    if xmax < wmax: wmax = xmax

    # Resolution
    l = (wmin+wmax)*0.5
    R = getResolution(mode, l)
    # print("Resolution at ",l," is: ",R)
    w = np.arange(wmin,wmax,l/R)
    
    # Compute spline fits
    c = 299792458.e+6 # um/s
    spectra = []
    for ispax in range(25):
        for ispex in range(16):
            x = w1[:,ispax,ispex]
            dnu = c/x * dw1[:,ispax,ispex]/x
            y = s1[:,ispex,ispax]/dnu
            # Sort x
            idx = np.argsort(x)
            x = x[idx]
            y = y[idx]
            # Uniq
            u, idx = np.unique(x, return_index=True)
            x = x[idx]
            y = y[idx]

            nknots = int((np.nanmax(x)-np.nanmin(x))//delta)
            if nknots > 10:
                ntot = len(x)-8
                idxk= 4+ntot//(nknots-1)*np.arange(nknots)
                t = x[idxk]
                spectrum = LSQUnivariateSpline(x,y,t)
                s = spectrum(w)
                idx = (w < x[1]) | (w > x[-1])
                s[idx] = np.nan
            else:
                s = np.ones(len(w)) * np.nan
            spectra.append(s)
        
    return w, np.array(spectra)

def computeMedianSpatialSpectra(spectra):
    """Compute spatial median spectra: a spectrum for each spaxel."""
    spatspectra = []
    for ispax in range(25):
        # Discard deviant spectra
        d = []
        for ispex in range(16):
            r = []
            for jspex in range(16):
                r.append(np.nanmedian(spectra[ispax*16+jspex]/spectra[ispax*16+ispex]))
            d.append(np.nanmean(np.array(r)))
        d = np.array(d)
        med = np.nanmedian(d)
        mad = np.nanmedian(np.abs(d-med))
        m = np.abs(d-med) < 3*mad
        # Obtain mean spectrum from non-deviant spectra
        spec = spectra[ispax*16+np.arange(16,dtype='int')]
        mspec = np.nanmean(spec[m],axis=0)
        spatspectra.append(mspec)
    return spatspectra

def computeFlatRed(wwaves,sspectra,sspatspectra,ispax,ispex,minflat=0.5,maxflat=2.3,delta=0.4):
    from scipy.signal import medfilt
    from scipy.ndimage.filters import generic_filter
    # Flats
    w=[];f=[]
    nfiles = np.shape(wwaves)[0]
    for i in range(nfiles):
        wi = wwaves[i]
        fi = sspectra[i][ispax*16+ispex]/sspatspectra[i][ispax]
        m = (fi < minflat) | (fi > maxflat)
        fi[m] = np.nan
        # Mask variable values
        med = medfilt(fi,21)
        mad = medfilt(np.abs(fi-med))
        m = np.abs(fi-med) > 5*mad
        fi[m] = np.nan
        # append
        w.append(wi)
        f.append(fi)
        
    # Discard highly deviant flats
    f = np.array(f)
    w = np.array(w)
    mf = np.nanmedian(f, axis = 1, keepdims=True)
    df = f - mf
    med = np.nanmedian(df, axis = 1, keepdims=True)
    mad = np.nanmedian(np.abs(med-df), axis = 1, keepdims=True)
    mask = np.abs(med - df) < 10*mad
    w = w[mask]
    f = f[mask]


    # Discard highly deviant flats
    #f = np.array(f)
    #w = np.array(w)
    #mf = np.nanmedian(f, axis = 1)
    #med = np.nanmedian(mf)
    #mad = np.nanmedian(np.abs(med-mf))
    #mask = np.abs(med - mf) < 10*mad
    #w = w[mask]
    #f = f[mask]
        
    # concatenate
    #try:
    #    w=np.concatenate(w); f=np.concatenate(f)
    #    m = np.isfinite(f)
    #    w=w[m]
    #    f=f[m]
    #except:
    #    print("There are no arrays to concatenate")
    #    return None
    
    idx = np.argsort(w); w=w[idx]; f=f[idx]
    # making wavelengths unique
    while(True):
        dd = w[1:]-w[:-1]
        idx = np.array(np.where(dd == 0.))  
        if np.sum(idx) == 0:
            break
        else:
            w[idx+1] = w[idx]+0.0001
    
    try:
        ntot = len(w)-8
        nknots = 10
        idxk= 4+ntot//(nknots-1)*np.arange(nknots)
        t = w[idxk]

        flat = LSQUnivariateSpline(w,f,t,k=3,ext=3)
        # Residuals
        for k in range(4):
            res = f - flat(w)
            med = medfilt(res,15)
            mad = medfilt(np.abs(res-med))
            m = (np.abs(res) < 3*mad) 
            x = w[m]
            nknots = int((np.nanmax(x)-np.nanmin(x))//delta)
            #nknots = 100
            ntot = len(x)-8
            idxk= 4+ntot//(nknots-1)*np.arange(nknots)
            t = x[idxk]
            flat = LSQUnivariateSpline(w[m],f[m],t,k=3,ext=3)
        # Compute dispersion of residuals        
        residuals = f[m] - flat(w[m])
        dev  = generic_filter(residuals, np.std, size=30)
        eflat = LSQUnivariateSpline(w[m],dev,t)
        return (flat, eflat)
    except:
        return None
    
def computeFlatBlue(wwaves,sspectra,sspatspectra,ispax,ispex,minflat=0.6,maxflat=1.8,delta=2.0):
    from scipy.signal import medfilt
    from scipy.ndimage.filters import generic_filter
    # Flats
    w=[];f=[]
    nfiles = np.shape(wwaves)[0]
    for i in range(nfiles):
        wi = wwaves[i]
        fi = sspectra[i][ispax*16+ispex]/sspatspectra[i][ispax]
        m = (fi < minflat) | (fi > maxflat)
        fi[m] = np.nan
        # If there is some values
        if np.sum(~np.isnan(fi)) > 0:
            idxm = np.concatenate(np.argwhere(~np.isnan(fi)))
            # Mask variable values
            med = medfilt(fi[idxm],21)
            mad = medfilt(np.abs(fi[idxm]-med))
            m2 = np.abs(fi[idxm]-med) > 5*mad
            if np.sum(m2) > 0:
                idx = np.concatenate(np.argwhere(m2))
                fi[idxm[idx]] = np.nan
        # append
        w.append(wi)
        f.append(fi)
        
    # Discard highly deviant flats
    f = np.array(f)
    w = np.array(w)
    mf = np.nanmedian(f, axis = 1, keepdims=True)
    df = f - mf
    med = np.nanmedian(df, axis = 1, keepdims=True)
    mad = np.nanmedian(np.abs(med-df), axis = 1, keepdims=True)
    mask = np.abs(med - df) < 10*mad
    w = w[mask]
    f = f[mask]


    # Discard highly deviant flats
    #f = np.array(f)
    #w = np.array(w)
    #mf = np.nanmedian(f, axis = 1)
    #med = np.nanmedian(mf)
    #mad = np.nanmedian(np.abs(med-mf))
    #mask = np.abs(med - mf) < 10*mad
    #w = w[mask]
    #f = f[mask]
        
    # concatenate
    #try:
    #    w=np.concatenate(w); f=np.concatenate(f)
    #    m = np.isfinite(f)
    #    w=w[m]
    #    f=f[m]
    #except:
    #    print("There are no arrays to concatenate")
    #    return None
    
    idx = np.argsort(w); w=w[idx]; f=f[idx]
    # making wavelengths unique
    while(True):
        dd = w[1:]-w[:-1]
        idx = np.array(np.where(dd == 0.))  
        if np.sum(idx) == 0:
            break
        else:
            w[idx+1] = w[idx]+0.0001
    
    try:
        # knots based on wavelength resolution
        # Let's difference between orders (limit at 70um)
        nknots1 = (70-np.nanmin(w)-0.2)/(delta/2.) 
        nknots2 = (np.nanmax(w)-70-0.2)/delta
        t = [np.nanmin(w)+0.2+delta/2.*np.arange(nknots1), 70+delta*np.arange(nknots2)]
        t = np.concatenate(t)
        #nknots = (np.nanmax(w)-np.nanmin(w)-0.4)/delta
        #t = np.nanmin(w)+0.2+delta*np.arange(nknots)

        flat = LSQUnivariateSpline(w,f,t,k=3,ext=3)
        # Residuals
        for k in range(4):
            res = f - flat(w)
            med = medfilt(res,15)
            mad = medfilt(np.abs(res-med))
            m = (np.abs(res) < 5*mad) 
            x = w[m]
            #nknots = (np.nanmax(x)-np.nanmin(x)-0.4)/delta
            #t = np.nanmin(x)+0.2+delta*np.arange(nknots)           
            nknots1 = (70-np.nanmin(x)-0.2)/(delta/2.) 
            nknots2 = (np.nanmax(x)-70-0.2)/delta
            t = [np.nanmin(x)+0.2+delta/2.*np.arange(nknots1), 70+delta*np.arange(nknots2)]
            t = np.concatenate(t)
            flat = LSQUnivariateSpline(w[m],f[m],t,k=3,ext=3)
        # Compute dispersion of residuals        
        residuals = f[m] - flat(w[m])
        dev  = generic_filter(residuals, np.std, size=30)
        eflat = LSQUnivariateSpline(w[m],dev,t)
        return (flat, eflat)
    except:
        return None


def computeMedianSpectrum(w, spatspectra):
    """Compute medium spectrum from spatial spectra."""
    # %matplotlib inline
    from matplotlib import rcParams
    rcParams['font.family']='STIXGeneral'
    rcParams['font.size']=18
    rcParams['mathtext.fontset']='stix'
    rcParams['legend.numpoints']=1
    import matplotlib.pyplot as plt

    spatspectra = np.array(spatspectra)
    d = []
    for ispax in range(25):
        r = []
        for jspax in range(25):
            r.append(np.nanmedian(spatspectra[jspax]/spatspectra[ispax]))
        d.append(np.nanmean(np.array(r)))
    d = np.array(d)
    md = np.nanmedian(d)
    mad = np.nanmedian(np.abs(d-md))
    #sd = np.nanstd(d)
    m = np.abs(d-md) < 3*mad
    #medspec = np.nanmedian(spatspectra[m],axis=0)
    medspec = np.nanmean(spatspectra[m],axis=0)

    # Compute flats
    flat = np.ones(25)
    for ispax in range(25):
        flat[ispax] = np.nanmedian(spatspectra[ispax]/medspec)

    # print(flat)
    #fig,ax = plt.subplots(figsize=(16,8))
    #for s in spatspectra:
    #    ax.plot(w,s,color='blue')
    #ax.plot(w,medspec,color='red')
    #plt.show()

    # Check
    fig,ax = plt.subplots(figsize=(16,8))
    i=0
    sflat = []
    for s in spatspectra:
        ax.plot(w,s/flat[i],marker='.',markersize=0.5,linewidth=0,color='blue')
        sflat.append(s/flat[i])
        i+=1
    sflat=np.array(sflat)
    superflat=np.nanmedian(sflat,axis=0)
    ax.plot(w,medspec,color='green')
    ax.plot(w,superflat,color='red')
    plt.show()
    medspec = superflat

    return medspec