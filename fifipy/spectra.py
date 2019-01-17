from scipy.interpolate import LSQUnivariateSpline
# from scipy.interpolate import interp1d
import numpy as np

def getResolution(mode, l):
    if mode == 'R':
        return 0.062423 * l*l - 6.6595 * l + 647.65
    elif mode == 'B1':
        return 0.16864 * l*l - 22.831 * l + 1316.6 
    elif mode == 'B2':
        return 1.9163 * l*l - 187.35 * l + 5496.9
    else:
        print('Invalid mode entered. Valid modes are R, B1, and B2')
        return l*0
    
def computeSplineFits(w1, dw1, s1, mode):
    wmin=0
    wmax=250
    for ispex in range(16):
        for ispax in range(25):
            x1 = w1[:,ispax,ispex]
            xmin = np.nanmin(x1)
            xmax = np.nanmax(x1)
            if xmin > wmin: wmin = xmin
            if xmax < wmax: wmax = xmax

    # Resolution
    l = (wmin+wmax)*0.5
    R = getResolution(mode, l)
    print("Resolution at ",l," is: ",R)
    print("Delta-lambda is: ", l/R)
    w = np.arange(wmin,wmax,l/R)
    

    # Compute spline fits
    c = 299792458.e-6 # um/s
    spectra = []
    for ispex in range(16):
        for ispax in range(25):
            x = w1[:,ispax,ispex]
            #diff = np.nanmax(x1)-np.nanmin(x1)
            #t = np.arange(np.nanmin(x1)+delta,np.nanmax(x1)-delta,diff/30.)
            # In this way I compute F_lambda, I should compute F_nu
            dnu = c/x * dw1[:,ispax,ispex]/x
            y = -s1[:,ispex,ispax]/dnu
            # F_nu
            #c = 299.792458e12 # um/s
            #dnu = c * dw1[:,ispax,ispex]/(x*x)
            #y = -s1[:,ispex,ispax]/dnu
            # Sort x
            idx = np.argsort(x)
            x = x[idx]
            y = y[idx]
            # Uniq
            u, idx = np.unique(x, return_index=True)
            x = x[idx]
            y = y[idx]
            delta = np.nanmedian(x[1:]-x[:-1])
            diff = np.nanmax(x)-np.nanmin(x)-2*delta
            t = np.arange(np.nanmin(x)+delta,np.nanmax(x)-delta,diff/30.)
            spectrum = LSQUnivariateSpline(x,y,t)
            spectra.append(spectrum(w))
        
    return w, np.array(spectra)

def computeMedianSpatialSpectra(spectra):
    """Compute spatial median spectra: a spectrum for each spaxel."""
    spatspectra = []
    for ispax in range(25):
        d = []
        for ispex in range(16):
            r = []
            for jspex in range(16):
                r.append(np.nanmedian(spectra[ispax*16+jspex]/spectra[ispax*16+ispex]))
            d.append(np.nanmean(np.array(r)))
        d = np.array(d)
        md = np.nanmedian(d)
        sd = np.nanstd(d)
        m = np.abs(d-md) < sd
        spec = spectra[ispax*16+np.arange(16,dtype='int')]
        mspec = np.nanmedian(spec[m],axis=0)
        spatspectra.append(mspec)
    return spatspectra


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
    sd = np.nanstd(d)
    m = np.abs(d-md) < sd
    medspec = np.nanmedian(spatspectra[m],axis=0)

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