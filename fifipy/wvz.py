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
        obsdate, coords, offset, angle, za, altitude, wv = hk
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
    fluxes = applyFlats(waves, fluxes, detchan, obsdate)  
    return waves, fluxes, detchan, order, za[0], altitude[0]

def computeAtran(waves, fluxes, detchan, order, za, altitude, computeAlpha=True):
    import matplotlib.pyplot as plt
    from fifipy.calib import readAtran
    import statsmodels.api as sm
    import numpy as np
    
    good = [0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23]
    wtot = np.ravel(waves[:,good,:])
    ftot = np.ravel(fluxes[:,good,:])
    idx = (ftot > 0.1e-8)
    wtot = wtot[idx]
    ftot = ftot[idx]
    lspec = sm.nonparametric.lowess(ftot,wtot, frac=0.03)
    wlow = lspec[:,0]
    flow = lspec[:,1]
    
    # compute better spatial flats
    alpha = np.ones(25)    
    if computeAlpha:
        for i in range(25):
            wi = np.ravel(waves[:,i,:])
            f_ = np.interp(wi, wlow, flow)
            fi = np.ravel(fluxes[:,i,:])
            alpha[i] = np.nansum(fi*f_)/np.nansum(fi*fi)
        
        # Recompute
        for i in range(25):
            fluxes[:,i,:] *= alpha[i]
        wtot = np.ravel(waves[:,good,:])
        ftot = np.ravel(fluxes[:,good,:])
        idx = (ftot > 0.1e-8)
        wtot = wtot[idx]
        ftot = ftot[idx]
        lspec = sm.nonparametric.lowess(ftot,wtot, frac=0.03)
        wlow = lspec[:,0]
        flow = lspec[:,1]
    
    
    wt, atran, altitudes, wvs = readAtran(detchan, order)
    #altitudes = 38000+np.arange(13)*500.
    imin = np.argmin(np.abs(altitudes-altitude))
    at = atran[imin]
    #wvs = 1. + np.arange(40)*0.25
    
    print('Order ',order,' Channel ',detchan, 'Alt ', altitude, 'ZA ',za)
    angle = za * np.pi/180.
    cos_angle = np.cos(angle)
    #depth = 1. / cos_angle  # Approximation
    r = 6383.5/50.  # assuming r_earth = 6371 km, altitude = 12.5 km, and 50 km of more stratosphere
    rcos = r * cos_angle
    depth = -rcos + np.sqrt(rcos * rcos + 1 + 2 * r) # taking into account Earth sphere
    
    if detchan == 'BLUE':
        # Compute the normalized curve
        lc = [62.00,62.10]
        lm = [63.31,63.33]
        idmin = (wtot > lc[0]) & (wtot < lc[1])
        idmax = (wtot > lm[0]) & (wtot < lm[1])
        fmin = np.nanmean(ftot[idmin])
        fmax = np.nanmean(ftot[idmax])
        df = fmax - fmin
        fabs = 1-(flow-fmin)/df
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
            idx = ftotabs < 1.1
            diff.append(np.nansum((ti[idx] - ftotabs[idx])**2))
        diff = np.array(diff)
        imin = np.argmin(diff)
        wvmin = wvs[imin]
        print('Min WV is ', wvmin)
        t = at[imin]**depth
        tmax = np.nanmean(t[idmin])
        tmin = np.nanmean(t[idmax])
        t = (t-tmin)/(tmax-tmin) # Normalize
        fig,ax = plt.subplots(figsize=(10,4))
        plt.plot(wvs,diff,'o')
        plt.grid()
        plt.show()
    
        fig,axes = plt.subplots(1,2,figsize=(16,6),sharey=True)
        ax=axes[0]
        for i in range(len(waves)):
            for j in good:
                ax.plot(waves[i,j,:],1-(fluxes[i,j,:]-fmin)/df,'.')
        ax.set_xlim(61.45,62.19)
        ax.set_ylim(-0.2,1.2)
        ax.plot( wlow,fabs)
        ax.plot( wt,t,color='orange')
        ax.grid()
        ax=axes[1]
        for i in range(len(waves)):
            for j in good:
                ax.plot(waves[i,j,:],1-(fluxes[i,j,:]-fmin)/df,'.')
        ax.set_xlim(63.01,63.75)
        ax.set_ylim(-0.2,1.2)
        ax.grid()
        plt.subplots_adjust(wspace=0)
        ax.plot( wlow,fabs)
        ax.plot( wt,t,color='orange')
        plt.show()
    else:
        lc = [148.2,148.4]
        lm = [146.8,146.9]
        idmin = (wtot > lc[0]) & (wtot < lc[1])
        idmax = (wtot > lm[0]) & (wtot < lm[1])
        fmin = np.nanmean(ftot[idmin])
        fmax = np.nanmean(ftot[idmax])
        df = fmax - fmin
        fabs = 1-(flow-fmin)/df
        ftotabs = 1-(ftot-fmin)/df
        fig,ax = plt.subplots(figsize=(10,4))
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
        print('Min WV is ', wvmin)
        t = at[imin]**depth
        tmax = np.nanmean(t[idmin])
        tmin = np.nanmean(t[idmax])
        t = (t-tmin)/(tmax-tmin) # Normalize
        plt.grid()
        plt.plot(wvs,diff,'o')
        plt.show()
        fig,ax = plt.subplots(figsize=(16,6))
        for i in range(len(waves)):
            for j in good:
                ax.plot(waves[i,j,:],1-(fluxes[i,j,:]-fmin)/df,'.')
        ax.plot( wlow,fabs)
        ax.set_ylim(-0.2,1.2)
        ax.set_xlim(np.nanmin(wtot),np.nanmax(wtot))
        ax.plot( wt,t,color='orange')
        ax.grid()
        plt.show()

    return wvmin, alpha

def getGroups(wvzdir, flight):
    from glob import glob as gb
    #wvzdir = '/Users/dfadda/sofia/FIFI-LS/WaterVapor/'
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

def flightPlots(lwgroups, alt, wblue, wred, title):
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
        temp.append(header['TEMPPRI1'])
        date.append(header['DATE-OBS'])
        wmon.append(header['WVZ_STA'])
        t = re.findall(r'\_(\d{6})\_',file)
        t = t[0]
        time.append(int(t[0:2])+int(t[2:4])/60.+int(t[4:6])/3600.)
    temp = np.array(temp)
    date = np.array(date)
    wmon = np.array(wmon)
    time = np.array(time)    
    
    time = time - np.nanmin(time) + 1.0
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
    ax.set_ylabel('Mirror Temp ($^o$C)')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.grid(which='both')
    ax = axes[2]
    ax.loglog(time,wblue, 'o', color='blue',label='blue')
    ax.loglog(time,wred, 'o', color='red',label='red')
    ax.loglog(time,wmon, 'o', color='green',label='monitor')
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    ax.set_xlabel('Hours')
    ax.set_ylabel('Water Vapor Zenith [$\mu m$]')
    ax.grid(which='both')
    ax.set_ylim(0.9,100)
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
    
    return time, temp, wmon, fig1