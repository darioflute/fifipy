def sortAtmFiles(files):
    import re
    # Regular expression taking out the time in the name (sequence of 6 digits surrounded by _)
    r = re.compile(r'\_(\d\d\d\d\d\d)\_')
    files = sorted(files, key=lambda x:r.search(x).group(1))
    return files


def computeFluxes(group):
    from fifipy.io import readData
    from fifipy.fit import multiSlopes
    from fifipy.calib import waveCal, applyFlats
    from fifipy.sensitivy import applyResponse
    import numpy as np
    from numpy import transpose
    c = 299792458.e+6 # um/s
    waves =[]
    fluxes=[]
    #dtime = 1./250.
    #factor = 3.63/65536./dtime # Factor to go back from V/s to pipeline scale

    for file in group:
        # Read data
        aor, hk, gratpos, voltage = readData(file)
        # Compute slopes
        spectra = multiSlopes(voltage, sky=True)
        # Get wavelengths
        detchan, order, dichroic, ncycles, nodbeam, filegpid, filenum = aor
        obsdate, telcoords, coords, offset, angle, za, altitude, wv = hk
        wave = []
        dw  = []
        for gp in gratpos:
            l,lw = waveCal(gratpos=gp, order=order, array=detchan,dichroic=dichroic,obsdate=obsdate)
            wave.append(transpose(l))
            dw.append(transpose(lw))
        
        # Compute flux
        dw = np.array(dw)
        wave = np.array(wave)
        dnu = c/wave * dw/wave
        flux = spectra / dnu
        for w,f in zip(wave,flux):
            waves.append(transpose(w))
            fluxes.append(transpose(f))
    
    # Apply flats
    waves = np.array(waves)
    fluxes = np.array(fluxes)
    # one obsdate is OK, since data are taken during the same day
    fluxes = applyFlats(waves, fluxes, detchan, order, dichroic, obsdate)
    # Apply response
    fluxes = applyResponse(waves, fluxes, detchan, order, dichroic, obsdate)
    
    return waves, fluxes, detchan, order, za[0], altitude[0]

def fitSlopesFile(file):
    from fifipy.io import readData
    from fifipy.stats import biweightLocation
    import numpy as np
    np.warnings.filterwarnings('ignore')
    saturationLimit = 2.7
    dtime = 1/250.  # Hz
    aor, hk, gratpos, voltage = readData(file)
    spectra = []
    # Readouts not considered: first two and the last one
    last = -2
    first = 2
    for v0 in voltage:
        # Reshape in ramps
        nz,ny,nx = np.shape(v0)
        nr = nz // 32
        v0 = v0.reshape(nr,32, nx*ny)
        v0 = np.transpose(v0, (1, 0, 2))
        m = v0 > saturationLimit
        v0[m] = np.nan
        # Compute intermediate slopes up to a difference of 16 readouts (halp ramp)
        # The first ramp is not considered
        dv0=[]
        for i in range(0,16-first):
            vv = (v0[first+i:last,1:,:]-v0[first:last-i,1:,:])/i
            dv0.append(vv)
        dv0 = np.concatenate(dv0)
        n1,n2,n3 = np.shape(dv0)
        dv0 = dv0.reshape(n1*n2, n3)
        # Compute the mean slope
        dvm = biweightLocation(dv0, axis=0) / dtime
        spectra.append(dvm)
    return spectra, gratpos


def computeMeanFlux(group, multi=True, subtractzero=True):
    from fifipy.io import readData
    #from fifipy.stats import biweightLocation
    from fifipy.calib import mwaveCal, applyFlats
    import numpy as np
    from dask import delayed, compute
    c = 299792458.e+6 # um/s
    np.warnings.filterwarnings('ignore')

    if multi:
        slopefit = [delayed(fitSlopesFile)(file) for file in group]        
        spectrafit = compute(* slopefit, scheduler='processes')
        spectra = []
        gpos = []
        for s in spectrafit:
            ss, gg = s
            spectra.append(ss)
            gpos.append(gg)
    else:
        spectra = []
        gpos = []
        for file in group:
            ss, gg = fitSlopesFile(file)
            spectra.append(ss)
            gpos.append(gg)     
           
    gpos = np.concatenate(gpos)
    spectra = np.array(spectra)

    aor, hk, gratpos, voltage = readData(group[0],subtractzero)
    detchan, order, dichroic, ncycles, nodbeam, filegpid, filenum = aor
    obsdate, telcoords, coords, offset, angle, za, altitude, wv = hk
    wave,dw = mwaveCal(gratpos=gpos, order=order, array=detchan,dichroic=dichroic,obsdate=obsdate)
    telra, teldec = telcoords
    #if addbaryshift:
    #    zb = baryshift(obsdate, telra, teldec)
    #    print('Baryshift ', zb)
    #    if zb < 0:
    #        zb = 0
    #    wave /= (1+zb)
        

    ng = len(gpos)
    spectra = spectra.reshape(ng, 16, 25)

    # Compute flux
    dnu = c/wave * dw/wave
    flux = spectra / dnu
    wave = np.transpose(wave, (0, 2, 1))  # Transpose the 2 last axes
    flux = np.transpose(flux, (0, 2, 1))
    flux = applyFlats(wave, flux, detchan, order, dichroic, obsdate) 
    return wave, flux, detchan, order, za[0], altitude[0]

def alphabeta(x,y):
    """Slope fitting."""
    import numpy as np
    idx = np.isfinite(y)
    if np.sum(idx) > 10:
        x = x[idx]
        y = y[idx]
        S = len(x)
        Sx = np.sum(x)
        Sy = np.sum(y)
        Sxx = np.sum(x*x)
        Sxy = np.sum(x*y)
        Delta = S * Sxx - Sx * Sx
        a = (Sxx * Sy - Sx * Sxy) / Delta
        b = (S * Sxy - Sx * Sy) / Delta
    else:
        a = 1
        b = 0
    return a, b

def reflat(waves, fluxes, good, computeAlpha=True):
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.signal import medfilt    
    import statsmodels.api as sm
    
    wtot = np.ravel(waves[:,good,:])
    ftot = np.ravel(fluxes[:,good,:])
    idx = np.isfinite(ftot)
    wtot = wtot[idx]
    ftot = ftot[idx]
    idx = np.argsort(wtot)
    wtot = wtot[idx]
    ftot = ftot[idx]
    fm = medfilt(ftot, 31)
    di = ftot - fm
    medi = np.nanmedian(di)
    madi = np.nanmedian(np.abs(di - medi))
    idx = np.abs(di) < 5 * madi
    wtot = wtot[idx]
    ftot = ftot[idx]

    if computeAlpha:   
        lspec = sm.nonparametric.lowess(ftot,wtot, frac=0.03)
        wlow = lspec[:,0]
        flow = lspec[:,1]
        f_ = interp1d(wlow, flow, fill_value='extrapolate')
        alpha = np.empty(25) 
        beta = np.empty(25)
        for i in good:
            wi = np.ravel(waves[:,i,:])
            fi = np.ravel(fluxes[:,i,:])
            # Exclude outliers
            idx = np.isfinite(fi)
            wi = wi[idx]
            fi = fi[idx]
            li = sm.nonparametric.lowess(fi, wi, frac=0.03)
            di = fi - li[:,1]
            medi = np.median(di)
            madi = np.median(np.abs(di - medi))
            idx = np.abs(di - medi) < 5 * madi
            wi = wi[idx]
            fi = fi[idx]
            alpha[i], beta[i] = alphabeta(wi, f_(wi)/fi)
            #alpha[i] = np.nansum(fi*f_(wi))/np.nansum(fi*fi)
    
        # Recompute
        for i in good:
            #print(i, alpha[i], beta[i])
            fluxes[:,i,:] *= alpha[i] + beta[i] * waves[:,i,:]
        wtot = np.ravel(waves[:,good,:])
        ftot = np.ravel(fluxes[:,good,:])
        idx = np.isfinite(ftot)
        wtot = wtot[idx]
        ftot = ftot[idx]
        # Get rid of outliers
        fm = medfilt(ftot, 31)
        di = ftot - fm
        medi = np.nanmedian(ftot - fm)
        madi = np.nanmedian(np.abs(ftot - medi))
        idx = np.abs(di) < 5 * madi
        wtot = wtot[idx]
        ftot = ftot[idx]
        
    idx = np.argsort(wtot)
    wtot = wtot[idx]
    ftot = ftot[idx]
    return wtot, ftot
    
def computeXcorr(wtot, ftot, wt, at):
    """Compute cross-correlation between ATRAN template and spectrum."""
    import numpy as np
    import statsmodels.api as sm

    lspec = sm.nonparametric.lowess(ftot,wtot, frac=0.03)
    wsky = lspec[:,0]
    fsky = lspec[:,1]
    wmin = np.min(wsky)
    wmax = np.max(wsky)
    
    # grid
    corr = []
    zs = np.arange(-51,51)*1e-5
    for z in zs:
        fi = np.interp(wt, wsky/(1+z), fsky)
        fi[wt < wmin] = 0
        fi[wt > wmax] = 0
        n = np.sum(fi > 0)
        corr.append(np.sum(fi*at))
        
    i = np.argmax(corr)
    zmin = zs[i]
    
    speedoflight = 300000.
    print('cz ', speedoflight * zmin)
    return zmin    
    

def computeAtran(waves, fluxes, detchan, order, za, altitude, 
                 atrandata=None, computeAlpha=True, plot=True, xcorr=True):
    import matplotlib.pyplot as plt
    from fifipy.calib import readAtran
    import numpy as np
    #import time
    #from astropy import units as u
    #from astropy.modeling.blackbody import blackbody_nu

    
    #temperature = -30 * u.Celsius
    #angle = 36. * (180/np.pi*3600)**2
    #Jy = 1.e-23

    #for i in range(16):
    #    for j in range(25):
    #        wbb = waves[:,j,i]*u.micron
    #        fbb_nu = blackbody_nu(wbb, temperature)
    #        if detchan == 'RED':
    #            fbb_nu = fbb_nu.value*angle*Jy*1.e12
    #        else:
    #            fbb_nu = fbb_nu.value*angle*Jy*2.e11
    #        fluxes[:,j,i] -= fbb_nu

    if detchan == 'RED':  
        good = [1,2,3,5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23]
        #good = [0,6,7,8,10,11,13,14,15,16,19,23]
    else:
        #good = [1,2,6,7,8,11,12,13,16,17,20,21,22]
        good = [0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23]
    # compute better spatial flats
    wtot, ftot = reflat(waves, fluxes, good, computeAlpha=computeAlpha)
    
    if atrandata is None:
        atrandata = readAtran(detchan, order)
        wt, atran, altitudes, wvs = atrandata
    else:
        wt, atran, altitudes, wvs = atrandata
    imin = np.argmin(np.abs(altitudes-altitude))
    at = atran[imin]
    angle = za * np.pi/180.
    a = 6371 + 12.5  # Earth radius + altitude (~12.5 km)
    c = 100           # Rest of stratosphere (~ 38 km)
    b = a + c      
    alpha = np.arcsin(a/b * np.sin(angle)) # From law of sinus
    dx = np.sqrt(a*a + b*b - 2*a*b*np.cos(angle-alpha)) # From law of cosinus
    depth = dx / c
    
    
    if detchan == 'BLUE':
        # Subtract BB (interpolation between T ~ 1)
        #lc1 = [61.55,61.70]
        #lc2 = [62.00,62.10]
        #id1 = ((wtot > lc1[0]) & (wtot < lc1[1])) 
        #id2 = ((wtot > lc2[0]) & (wtot < lc2[1]))
        #f1 = np.nanmean(ftot[id1])
        #f2 = np.nanmean(ftot[id2])
        #w1 = 0.5 * (lc1[0] + lc1[1])
        #w2 = 0.5 * (lc2[0] + lc2[1])
        #ftot -= np.interp(wtot, np.array([w1,w2]), np.array([f1,f2]))
        # Compute the normalized curve
        lc = [61.55,61.70]
        lm = [63.30,63.34]
        idmin = ((wtot > lc[0]) & (wtot < lc[1])) #| ((wtot > lc1[0]) & (wtot < lc1[1]))
        idmax = (wtot > lm[0]) & (wtot < lm[1])
        fmin = np.nanmean(ftot[idmin])
        fmax = np.nanmean(ftot[idmax])
        df = fmax - fmin
        ftotabs = 1-(ftot-fmin)/df
        # Normalize at the same way the ATRAN models
        diff = []
        idmin = (wt > lc[0]) & (wt < lc[1])
        idmax = (wt > lm[0]) & (wt < lm[1])
        for t, wv in zip(at, wvs):
            t = t**depth  # Apply the ZA
            tmax = np.nanmean(t[idmin])
            tmin = np.nanmean(t[idmax])
            t = (t-tmin)/(tmax-tmin)  # Normalize
            ti  = np.interp(wtot,wt,t)
            idx = ftotabs < 1.1
            diff.append(np.nansum((ti[idx] - ftotabs[idx])**2))
        diff = np.array(diff)
        imin = np.argmin(diff)
        wvmin = wvs[imin]
        # Recompute minimum using a parabola through the lowest three points
        try:
            x1=wvs[imin-1]
            x2=wvs[imin]
            x3=wvs[imin+1]
            y1=diff[imin-1]
            y2=diff[imin]   
            y3=diff[imin+1]
            b = np.array([[x1*x1,y1,1],[x2*x2,y2,1],[x3*x3,y3,1]])
            a = np.array([[y1,x1,1],[y2,x2,1],[y3,x3,1]])
            wvmin = -0.5 * np.linalg.det(b)/np.linalg.det(a)
        except:
            pass
            
        t = at[imin]**depth
        tmax = np.nanmean(t[idmin])
        tmin = np.nanmean(t[idmax])
        t = (t-tmin)/(tmax-tmin) # Normalize
        
        # We can cross-correlate here
        if xcorr:
            zcorr = computeXcorr(wtot, ftotabs, wt, t)
            waves /= 1 + zcorr

        if plot:
            fig,axes = plt.subplots(1, 3, figsize=(16,5), sharey=True,
                                    gridspec_kw = {'width_ratios': [2,3,3]})
            ax=axes[0]
            ax.set_title('WVZ')
            ax.plot(wvs,diff/np.nanmax(diff))#, 'o')
            ax.grid()
            ax=axes[1]
            for j in good:
                ax.plot(np.ravel(waves[:,j,:]),1-(np.ravel(fluxes[:,j,:])-fmin)/df,'.')
            ax.set_xlim(61.45,62.19)
            ax.set_ylim(-0.3,1.2)
            ax.plot( wt,t,color='orange',linewidth=2)
            ax.grid()
            ax=axes[2]
            for j in good:
                ax.plot(np.ravel(waves[:,j,:]),1-(np.ravel(fluxes[:,j,:])-fmin)/df,'.')
            ax.set_xlim(63.01,63.75)
            ax.set_ylim(-0.3,1.2)
            ax.grid()
            plt.subplots_adjust(wspace=0)
            ax.plot( wt,t,color='orange',linewidth=2)
            fig.suptitle(' Channel: '+str(detchan)+ ', Alt: '+str(altitude)+ ', ZA: '+'{:5.2f}'.format(za)+
                         ', WVZ: ' + '{:5.2f}'.format(wvmin), size=20)
            fig.subplots_adjust(top=0.9) 
            plt.show()
    else:
        # Subtract BB (interpolation between T ~ 1)
        #lc1 = [145.65, 145.85]
        #lc2 = [149.3, 149.6]
        #id1 = ((wtot > lc1[0]) & (wtot < lc1[1])) 
        #id2 = ((wtot > lc2[0]) & (wtot < lc2[1]))
        #f1 = np.nanmean(ftot[id1])
        #f2 = np.nanmean(ftot[id2])
        #w1 = 0.5 * (lc1[0] + lc1[1])
        #w2 = 0.5 * (lc2[0] + lc2[1])
        #ftot -= np.interp(wtot, np.array([w1,w2]), np.array([f1,f2]))
        lc = [148.0,148.3]
        #lc = [149.3, 149.6]
        lm = [146.85,147]
        idmin = (wtot > lc[0]) & (wtot < lc[1])
        idmax = (wtot > lm[0]) & (wtot < lm[1])
        fmin = np.nanmean(ftot[idmin])
        fmax = np.nanmean(ftot[idmax])
        df = fmax - fmin
        ftotabs = 1-(ftot-fmin)/df
        # Normalize at the same way the ATRAN models
        diff = []     
        idmin = (wt > lc[0]) & (wt < lc[1])
        idmax = (wt > lm[0]) & (wt < lm[1])
        for t,wv in zip(at,wvs):
            t = t**depth  # Apply the ZA
            tmax = np.nanmean(t[idmin])
            tmin = np.nanmean(t[idmax])
            t = (t-tmin)/(tmax-tmin)  # Normalize
            ti  = np.interp(wtot,wt,t)
            idx = ftotabs < 1.2
            diff.append(np.nansum((ti[idx] - ftotabs[idx])**2))
        diff = np.array(diff)
        imin = np.argmin(diff)
        wvmin = wvs[imin]
        try:
            x1=wvs[imin-1]
            x2=wvs[imin]
            x3=wvs[imin+1]
            y1=diff[imin-1]
            y2=diff[imin]   
            y3=diff[imin+1]
            b = np.array([[x1*x1,y1,1],[x2*x2,y2,1],[x3*x3,y3,1]])
            a = np.array([[y1,x1,1],[y2,x2,1],[y3,x3,1]])
            wvmin = -0.5 * np.linalg.det(b)/np.linalg.det(a)
        except:
            pass

        t = at[imin]**depth
        tmax = np.nanmean(t[idmin])
        tmin = np.nanmean(t[idmax])
        t = (t-tmin)/(tmax-tmin) # Normalize
        # We can cross-correlate here
        if xcorr:
            zcorr = computeXcorr(wtot, ftotabs, wt, t)
            waves /= 1 + zcorr

        if plot:
            
            
            fig,axes = plt.subplots(1,2,figsize=(16,5),sharey=True, 
                                    gridspec_kw = {'width_ratios': [2,6]})
            ax=axes[0]
            ax.set_title('WVZ')
            ax.plot(wvs,diff/np.nanmax(diff))#,'o')
            ax.grid()
            ax=axes[1]
            for j in good:
                ax.plot(np.ravel(waves[:,j,:]),1-(np.ravel(fluxes[:,j,:])-fmin)/df,'.',label=str(j))
            ax.set_ylim(-0.3,1.2)
            ax.set_xlim(np.nanmin(wtot),np.nanmax(wtot))
            ax.plot( wt,t,color='orange',linewidth=2)
            ax.grid()
            fig.suptitle(' Channel: '+str(detchan)+ ', Alt: '+str(altitude)+ ', ZA: '+'{:5.2f}'.format(za)+ 
                         ', WVZ: ' + '{:5.2f}'.format(wvmin), size=20)
            plt.subplots_adjust(wspace=0)
            fig.subplots_adjust(top=0.9) 
            plt.show()
    return wvmin, alpha, wtot, ftotabs

def computeAtranTot(wred, fred, wblue, fblue, za, altitude, atran1, atran2):
    import numpy as np
    import matplotlib.pyplot as plt
        
    wtr, atranr, altitudes, wvs = atran1
    wtb, atranb, altitudes, wvs = atran2
    imin = np.argmin(np.abs(altitudes-altitude))
    atr = atranr[imin]
    atb = atranb[imin]
        
    angle = za * np.pi/180.
    a = 6371 + 12.5  # Earth radius + altitude (~12.5 km)
    c = 100          # Rest of stratosphere (~ 38 km)
    b = a + c      
    alpha = np.arcsin(a/b * np.sin(angle)) # From law of sinus
    dx = np.sqrt(a*a + b*b - 2*a*b*np.cos(angle-alpha)) # From law of cosinus
    depth = dx / c    
    
    # Blue
    lc = [61.55,61.70]
    lm = [63.30,63.34]
    idbmin = (wtb > lc[0]) & (wtb < lc[1])
    idbmax = (wtb > lm[0]) & (wtb < lm[1])
    
    #lc = [149.3, 149.6]
    #lm = [146.8,147.0]
    lc = [148.0,148.3]
    lm = [146.85,147]
    idrmin = (wtr > lc[0]) & (wtr < lc[1])
    idrmax = (wtr > lm[0]) & (wtr < lm[1])
    
    diff = []
    for tr, tb, wv in zip(atr, atb, wvs):
        tb = tb**depth  # Apply the ZA
        tmax = np.nanmean(tb[idbmin])
        tmin = np.nanmean(tb[idbmax])
        tb = (tb-tmin)/(tmax-tmin)  # Normalize
        tbi  = np.interp(wblue,wtb,tb)
        idxb = fblue < 1.1
        tr = tr**depth  # Apply the ZA
        tmax = np.nanmean(tr[idrmin])
        tmin = np.nanmean(tr[idrmax])
        tr = (tr-tmin)/(tmax-tmin)  # Normalize
        tri  = np.interp(wred,wtr,tr)
        idxr = fred < 1.1
        diff.append(np.nanmean((tbi[idxb] - fblue[idxb])**2) +
                    np.nanmean((tri[idxr] - fred[idxr])**2))
            
    diff = np.array(diff)
    imin = np.argmin(diff)
    wvmin = wvs[imin]
    
    try:
        x1=wvs[imin-1]
        x2=wvs[imin]
        x3=wvs[imin+1]
        y1=diff[imin-1]
        y2=diff[imin]   
        y3=diff[imin+1]
        b = np.array([[x1*x1,y1,1],[x2*x2,y2,1],[x3*x3,y3,1]])
        a = np.array([[y1,x1,1],[y2,x2,1],[y3,x3,1]])
        wvmin = -0.5 * np.linalg.det(b)/np.linalg.det(a)
    except:
        pass
        print('parabola not fitted')

    

    tb = atb[imin]**depth
    tmax = np.nanmean(tb[idbmin])
    tmin = np.nanmean(tb[idbmax])
    tb = (tb-tmin)/(tmax-tmin) # Normalize
    tr = atr[imin]**depth
    tmax = np.nanmean(tr[idrmin])
    tmin = np.nanmean(tr[idrmax])
    tr = (tr-tmin)/(tmax-tmin) # Normalize


    fig,axes = plt.subplots(1, 4, figsize=(16,5), sharey=True,
                                gridspec_kw = {'width_ratios': [1,1,1,2]})
    ax=axes[0]
    ax.set_title('WVZ')
    ax.plot(wvs,diff/np.nanmax(diff))#, 'o')
    ax.grid()
    ax=axes[1]
    ax.plot(wblue, fblue,'.')
    ax.set_xlim(61.45,62.19)
    ax.set_ylim(-0.3,1.2)
    ax.plot( wtb,tb,color='orange',linewidth=2)
    ax.grid()
    ax=axes[2]
    ax.plot(wblue, fblue,'.')
    ax.set_xlim(63.01,63.75)
    ax.set_ylim(-0.3,1.2)
    ax.grid()
    plt.subplots_adjust(wspace=0)
    ax.plot( wtb,tb,color='orange',linewidth=2)
    ax=axes[3]
    ax.plot(wred, fred,'.',color='red')
    ax.set_xlim(145.6, 149.9)
    ax.set_ylim(-0.3,1.2)
    ax.grid()
    plt.subplots_adjust(wspace=0)
    ax.plot( wtr,tr,color='orange',linewidth=2)
        
    fig.suptitle('Alt: '+str(altitude)+ ', ZA: '+'{:5.2f}'.format(za)+
                     ', WVZ: ' + '{:5.2f}'.format(wvmin), size=20)
    fig.subplots_adjust(top=0.9) 
    plt.show()
    
    return wvmin


def getGroups(wvzdir, flight):
    from glob import glob as gb
    path = wvzdir + '/' + flight
    
    groups = []
    for w in ['sw','lw']:
        afiles = sortAtmFiles(gb(path+'/*a_atm*'+w+'.fits'))
        bfiles = sortAtmFiles(gb(path+'/*b_atm*'+w+'.fits'))
        cfiles = sortAtmFiles(gb(path+'/*c_atm*'+w+'.fits'))
        dfiles = sortAtmFiles(gb(path+'/*d_atm*'+w+'.fits'))
        efiles = sortAtmFiles(gb(path+'/*e_atm*'+w+'.fits'))
        group = [ [a,b,c,d,e] for a,b,c,d,e in zip(afiles,bfiles,cfiles,dfiles,efiles)]
        groups.append(group)
        
    #print(afiles)
        
    return groups

def flightPlots(lwgroups, alt, wblue, wred, wtot, title, monitor=True):
    from matplotlib import rcParams
    import matplotlib.pyplot as plt
    rcParams['font.family']='STIXGeneral'
    rcParams['font.size']=18
    rcParams['mathtext.fontset']='stix'
    rcParams['legend.numpoints']=1
    from astropy.io.fits import getheader
    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
    import numpy as np
    import re

    temp = []
    date = []
    wmon = []
    time = []
    for group in lwgroups:
        file = group[0]
        header = getheader(file)
        #temp.append(header['TEMPPRI1'])
        temp.append(header['TEMP_OUT'])
        date.append(header['DATE-OBS'])
        wmon.append(header['WVZ_STA'])
        t = re.findall(r'\_(\d{6})\_',file)
        t = t[0]
        time.append(int(t[0:2])+int(t[2:4])/60.+int(t[4:6])/3600.)
    temp = np.array(temp)
    date = np.array(date)
    wmon = np.array(wmon)
    time = np.array(time)  
    
    try:
        mintime = np.nanmin(time)
    except:
        mintime = 0
    
    time = time - mintime + 1.0
    fig1,axes = plt.subplots(3,1,figsize=(14,12),sharex=True, gridspec_kw = {'height_ratios': [1,1,2]})
    ax = axes[0]
    #ax.axis([1,11,37500,44500])
    ax.semilogx(time, alt,'o')
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    ax.grid(which='both')
    ax.set_ylabel('Altitude (ft)')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax = axes[1]
    ax.semilogx(time, temp, 'o')
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    ax.set_ylabel('External Temp ($^o$C)')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.grid(which='both')
    ax = axes[2]
    ax.loglog(time,wblue, 'o', color='blue',label='blue')
    ax.loglog(time,wred, 'o', color='red',label='red')
    ax.loglog(time,wtot, 'o', color='black',label='red + blue',
              markerfacecolor='None', markersize=15)
    if monitor:
        ax.loglog(time,wmon, 'o', color='green',label='monitor')
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    ax.set_xlabel('Hours')
    ax.set_ylabel('Water Vapor Zenith [$\mu m$]')
    ax.grid(which='both')
    if monitor:
        ax.set_ylim(0.9,100)
    else:
        ax.set_ylim(0.9, 30)
    ax.set_xlim(0.9,10)
    plt.legend(loc = 'upper right')
    fig1.suptitle(title, fontsize=20)
    plt.subplots_adjust(hspace=0)
    plt.show()

    fig2,axes = plt.subplots(1,2,figsize=(15,5))
    idx = temp < -5
    ax = axes[0]
    ax.plot(alt[idx], wblue[idx], 'o',color='blue',label='blue')
    ax.plot(alt[idx], wred[idx], 'o',color='red',label='red')
    if monitor:
        ax.plot(alt[idx], wmon[idx], 'o',color='green',label='monitor')
    ax.set_xlabel('Altitude')
    ax.set_ylabel('Water Vapor Zenith [$\mu m$]')
    ax.legend()
    ax = axes[1]
    ax.plot(wmon[idx], wblue[idx]/wmon[idx],'o',color='blue')
    ax.plot(wmon[idx], wred[idx]/wmon[idx],'o',color='red')
    ax.set_xlabel('WVZ monitor')
    ax.set_ylabel('WVZ fit/monitor')
    plt.show()
    
    return date, time, temp, wmon, fig1

def baryshift(obsdate, ra, dec, equinox='J2000'):
    """
    Compute the redshift due to sun and earth movements
    Parameters
    ----------
    obsdate : observational date
    ra : right ascension of line of sight (direction of telescope)
    dec : declination of line of sight (direction of telescope)
    equinox : Equinox. The default is 2000.0.

    Returns
    -------
    redshift due to sun/earth movement along the line of sight
    """

    import astropy.constants as const
    from astropy.coordinates import (UnitSphericalRepresentation, FK5, 
                                     solar_system, CartesianRepresentation)
    from astropy.io import fits
    from astropy.time import Time
    import astropy.units as u

    # Convert to Julian date
    try:
        time = Time(obsdate)
    except:
        print('Invalidate observational date')
        return
    
    sc = FK5(ra * u.hourangle, dec * u.deg, equinox=equinox)
    sc_cartesian = sc.represent_as(UnitSphericalRepresentation).\
            represent_as(CartesianRepresentation)
    _, ev = solar_system.get_body_barycentric_posvel('earth', time)
    helio_vel = sc_cartesian.dot(ev).to(u.km / u.s)
    # Compute solar velocity wrt LSR
    sunpos = FK5(18 * u.hourangle, 30 * u.deg, equinox='J1900')
    # Precess to current equinox
    sunpos = sunpos.transform_to(FK5(equinox=equinox))
    sun_v0 = 20 * u.km / u.s
    sun_cartesian = sunpos.represent_as(UnitSphericalRepresentation).\
        represent_as(CartesianRepresentation)
    sun_vel = sc_cartesian.dot(sun_cartesian) * sun_v0
    vlsr = helio_vel + sun_vel
    speed_of_light = const.c.to(vlsr.unit)
    result = vlsr / speed_of_light
   
    return result.value