def reflat(x, y, alpha, p):
    import numpy as np
    n, = np.shape(x)
    ng = n // 16
    pp = p(x)* alpha
    r = y/pp
    r = r.reshape((ng, 16))
    flat = np.nanmedian(r, axis=0)
    flat /= np.nanmedian(flat)
    return flat

def medflat(f, n=16):
    import numpy as np
    ff = f.copy()
    ng = len(ff) // n
    ff = ff[:ng*n].reshape((ng, n))
    for ff_ in ff:
        ff_ /= np.nanmedian(ff_)
    flat = np.nanmedian(ff, axis=0)
    flat /= np.nanmedian(flat)
    return flat

def autoflat(wave,flux,model,ncycle=16):
    # Find flat pattern each 16 pixels and auto-flat
    import numpy as np
    from scipy.signal import medfilt
    data = np.loadtxt('/Users/dfadda/sofia/fluxcal/models/'+model, skiprows=3)
    wmodel = data[:,0]
    fmodel = data[:,1]
    
    # 1st approx
    f = np.array(flux.copy())
    w = np.array(wave.copy())
    n = np.size(f)
    ng = n // ncycle
    ww = np.arange(ncycle)
    ff = np.zeros((ng, ncycle))
    for i in range(ng):
        wi = w[i*ncycle:(i+1)*ncycle]
        fi = f[i*ncycle:(i+1)*ncycle] * np.interp(wi, wmodel,fmodel)
        if np.nanmedian(fi) > 0:
            idx = np.isfinite(fi)
            A = np.vstack([ww[idx], np.ones(len(ww[idx]))]).T
            m, c = np.linalg.lstsq(A, fi[idx], rcond=None)[0]
            ff[i,:]=fi/(m*ww+c)
        else:
            ff[i,:]=1
    aflat = np.nanmedian(ff, axis=0)
    # Chebyshev 
    f = np.array(flux.copy())
    w = np.array(wave.copy())
    x = []
    y = []
    for i in range(ng):
        wi = w[i*ncycle:(i+1)*ncycle] 
        fi = f[i*ncycle:(i+1)*ncycle] * np.interp(wi, wmodel,fmodel)
        x.extend(wi)
        y.extend(fi/aflat)
    x = np.array(x)
    y = np.array(y)
    s = np.argsort(x)
    x = x[s]
    y = y[s]
    # Outlier rejection
    ym = medfilt(y, 31)
    df = y - ym
    idx = np.abs(y - ym )/ym < 0.2
    x = x[idx]
    y = y[idx]
    p = np.polynomial.Chebyshev.fit(x, y, 21)

    # 2nd approx
    f = np.array(flux.copy())
    w = np.array(wave.copy())
    n = np.size(f)
    ng = n // ncycle
    wi = np.arange(ncycle)
    ff = np.zeros((ng, ncycle))
    for i in range(ng):
        wi = w[i*ncycle:(i+1)*ncycle]
        fi = f[i*ncycle:(i+1)*ncycle] * np.interp(wi, wmodel,fmodel)
        if np.nanmedian(fi) > 0:
            ff[i,:]=fi/p(wi)
        else:
            ff[i,:]=1
    aflat = np.nanmedian(ff, axis=0)
    
    return aflat

def readFlfData(path):
    from astropy.io import fits
    import fnmatch, os
    import numpy as np
    #path = '/Users/dfadda/sofia/fluxcal/data/2021/F716/red/'

    nstack = 0
    files = sorted(fnmatch.filter(os.listdir(path),"*FLF*.fits"))
    alt = []
    za  = []
    wvz = []
    dic = []

    for file in files:
        with fits.open(path + file) as hlf:
            header = hlf['PRIMARY'].header
            obslam = header['OBSLAM']
            obsbet = header['OBSBET']
            channel = header['DETCHAN']
            missnid = header['MISSN-ID']
            flightNumber = int((missnid.split('_F'))[-1])
            platscale = header['PLATSCAL']
            wvz.append(header['WVZ_OBS'])
            alt.append(header['ALTI_STA'])
            za.append(header['ZA_START'])
            dic.append(header['DICHROIC'])
            ncycles = (len(hlf)-1) // 9
            for j in range(ncycles):
                xs = hlf['XS_G'+str(j)].data
                ys = hlf['YS_G'+str(j)].data
                if channel == 'RED':
                    pixscale=12
                    pixfactor = (platscale*3/pixscale)**2
                else:
                    pixscale=6
                    pixfactor = (platscale*1.5/pixscale)**2
                fs = hlf['FLUX_G'+str(j)].data / pixfactor   # Normalize for resampling
                ws = hlf['LAMBDA_G'+str(j)].data
                if nstack == 0:
                    x = xs
                    y = ys
                    f = fs
                    w = ws
                else:
                    x = np.vstack((x, xs))
                    y = np.vstack((y, ys))
                    f = np.vstack((f, fs))
                    w = np.vstack((w, ws))
                nstack += 1

    wvz = np.array(wvz)
    alt = np.array(alt)
    za  = np.array(za)
    dic = np.array(dic)
    return x, y, w, f, wvz, alt, za, dic, ncycles

def readScmData(path):
    from astropy.io import fits
    import fnmatch, os
    import numpy as np
    #path = '/Users/dfadda/sofia/fluxcal/data/2021/F716/red/'

    nstack = 0
    files = sorted(fnmatch.filter(os.listdir(path),"*SCM*.fits"))
    alt = []
    za  = []
    wvz = []
    dic = []

    for file in files:
        with fits.open(path + file) as hlf:
            header = hlf['PRIMARY'].header
            obslam = header['OBSLAM']
            obsbet = header['OBSBET']
            channel = header['DETCHAN']
            missnid = header['MISSN-ID']
            flightNumber = int((missnid.split('_F'))[-1])
            xs = hlf['XS'].data
            ys = hlf['YS'].data
            platscale = header['PLATSCAL']
            if channel == 'RED':
                pixscale=12
                pixfactor = (platscale*3/pixscale)**2
            else:
                pixscale=6
                pixfactor = (platscale*1.5/pixscale)**2
            fs = hlf['FLUX'].data / pixfactor   # Normalize for resampling
            ws = hlf['LAMBDA'].data
            ncycle, nspax = np.shape(fs)
            wvz.append(header['WVZ_OBS'])
            alt.append(header['ALTI_STA'])
            za.append(header['ZA_START'])
            dic.append(header['DICHROIC'])
        if nstack == 0:
            x = xs
            y = ys
            f = fs
            w = ws
        else:
            x = np.vstack((x, xs))
            y = np.vstack((y, ys))
            f = np.vstack((f, fs))
            w = np.vstack((w, ws))
        nstack += 1

    wvz = np.array(wvz)
    alt = np.array(alt)
    za  = np.array(za)
    dic = np.array(dic)
    return x, y, w, f, wvz, alt, za, dic, ncycle

def computeResponse(x,y,w,f,alt,wvz,za,dic,wmodel,fmodel,conv130to105,band='R',dichroic='both',plot='True',o3dobson=320,
                    goodpixels = [0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23], atmthreshold=0.5):
    from fifipy.am import callAM, convolveAM
    from fifipy.spectra import getResolution
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Conversion of dichroics
    #conv130to105 = '/Users/dfadda/sofia/FIFI-LS/SpectralFlats/calsource_flat_04_2021/D130onD105_R.fits'
    if dichroic == 'both':
        with fits.open(conv130to105) as hdul:
            w130 = hdul['Wavelength'].data
            f105to130 = hdul['Flux'].data

    if plot:
        fig = plt.figure(figsize=(14,8))
        ax = fig.add_axes([0,0.3,1,.7])
        ax1 = fig.add_axes([0,0,1,.3])
        for i in range(len(alt)):
            ww = np.nanmean(w[i*16:(i+1)*16,goodpixels],axis=1)
            fmod = np.interp(ww, wmodel, fmodel)
            if dic[i] == '105':
                ff = np.nansum(f[i*16:(i+1)*16,goodpixels],axis=1)
                if (dichroic == 'both') | (dichroic == '105'):
                    ax.plot(ww,ff/fmod,color='red',alpha=0.5)
            else:
                if (dichroic == 'both'):
                    fi = f[i*16:(i+1)*16,:]
                    #for j in range(25):
                    #    f2 = np.interp(w[i*16:(i+1)*16,j], w130, f105to130)
                    #    fi [:,j] /= f2
                    ff = np.nansum(fi[:,goodpixels],axis=1)
                    f2 = np.interp(ww, w130, f105to130)
                    ax.plot(ww, ff/fmod/f2,color='purple',alpha=0.5)
                elif dichroic == '130':
                    ff = np.nansum(f[i*16:(i+1)*16,goodpixels],axis=1)
                    ax.plot(ww, ff/fmod,color='purple',alpha=0.5)

    wtot = []
    ftot = []
    for i in range(len(alt)):
        print(i, end=' ')
        #xx = np.nanmean(x[i*16:(i+1)*16,:],axis=0)
        #yy = np.nanmean(y[i*16:(i+1)*16,:],axis=0)
        ww = np.nanmean(w[i*16:(i+1)*16,goodpixels],axis=1)
        ff = np.nansum(f[i*16:(i+1)*16,goodpixels],axis=1)
        wmin = np.nanmin(ww)
        wmax = np.nanmax(ww)
        dw = wmin/getResolution(band, wmin)
        wt,trans = callAM(alt[i],  wvz[i], wmin-2*dw, wmax+2*dw, o3dobson=o3dobson)
        # Sort in wt
        s = np.argsort(wt)
        wt = wt[s]
        trans = trans[s]
        wc, tc = convolveAM(wt, trans, band)
        ti = np.interp(ww, wc, tc)
        # Correction for zenithal angle
        angle = za[i] * np.pi/180.
        cos_angle = np.cos(angle)
        depth = 1. / cos_angle  # Flat Earth approximation
        ti = ti**depth
        idx = (ff < 0) | (ti < atmthreshold)
        ff[idx] = np.nan
        fmod = np.interp(ww, wmodel, fmodel)
        if dic[i] == 105:
            if (dichroic == 'both') | (dichroic == '105'):
                ftot.extend(ff/ti/fmod)
                wtot.extend(ww)
                if plot:
                    ax.plot(ww,ff/ti/fmod,color='red')
                    ax1.plot(ww, ti)
        else:
            if (dichroic == 'both'):            
                #fi = f[i*16:(i+1)*16,:]
                #wi = w[i*16:(i+1)*16,:]
                #for j in range(25):
                #    f2 = np.interp(wi[:,j], w130, f105to130)
                #    fi [:,j] /= f2
                #ff = np.nansum(fi[:,goodpixels],axis=1)
                f2 = np.interp(ww, w130, f105to130)
                ftot.extend(ff/ti/fmod/f2)
                wtot.extend(ww)
                if plot:
                    ax.plot(ww, ff/ti/fmod/f2,color='purple')
                    ax1.plot(ww, ti)
            elif dichroic == '130':
                ftot.extend(ff/ti/fmod)
                wtot.extend(ww)
                if plot:
                    ax.plot(ww, ff/ti/fmod,color='purple')
                    ax1.plot(ww, ti)
    if plot:
        ax.grid()
        ax1.grid()
        plt.show()

    return np.array(wtot), np.array(ftot)
