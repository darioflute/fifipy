def readRpData(fitsfile):
    from astropy.io import fits
    import numpy as np
    #import gc


    with fits.open(fitsfile, memmap=False) as hdulist:
        #data = []
        next = len(hdulist)
        header = hdulist[0].header
    
        detchan = header['DETCHAN']
        obsdate = header['DATE-OBS']
        dichroic = header['DICHROIC']
        exptime = header['EXPTIME']/next  # Fraction of exptime in one grating

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
        for i in range(1,next):
            d = hdulist[i].data
            e = d['STDDEV']
            f = d['DATA']
            error.append(e[0,1:17,:,:])
            flux.append(f[0,1:17,:,:])
            del d,e,f

    #gc.collect()
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
        print('\n',fluxcaldir+direc+'/reduced/')
        nfiles = np.size(files)
        print('number of files ',nfiles)
        i = nfiles
        for file in files:
            i -= 1
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
        file = path0+'/data/Response_'+array+'_D'+str(dichroic)+'_Ord'+str(order)+'_20190705v1.fits'
    else:
        file = path0+'/data/Response_'+array+'_D'+str(dichroic)+'_Ord'+str(order)+'_20190705v2.fits'
    hdl = fits.open(file)
    data = hdl['PRIMARY'].data
    hdl.close()
    wr = data[0,:]
    fr = data[1,:]
    response = interp1d(wr,fr,fill_value='extrapolate')
    return response


def computeSensitivity(responseDir, array, order, dichroic, waves, dwaves, error, exptime, obsdate):
    from matplotlib import rcParams
    rcParams['font.family']='STIXGeneral'
    rcParams['font.size']=18
    rcParams['mathtext.fontset']='stix'
    rcParams['legend.numpoints']=1
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits
    from scipy.interpolate import LSQUnivariateSpline
    from fifipy.calib import readFlats

    #spatflat = readSpatFlats(array, obsdate, silent=False)
    wflat, specflat, especflat, spatflat= readFlats(array, order, dichroic, obsdate, silent=True)
    response = readResponse(array, dichroic, order, obsdate)

    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(1,1,1)
    wtot = []
    stot = []

    # Regions to mask
    if array == 'Blue':
        regions = [
            [53,53.25],
            [55,55.2],
            [56.2,56.5],
            [57.5,57.75],
            [58.5,59.0],
            [59.9,60.1],
            [70.25,72.75],
            #[71.75,72.25],
            [74.25,76.5],
            [78,79.5],
            [81.25,82.25],
            [83,83.5],
            [89.5,90.25],
            [95.4,95.9] ,
            [99,100],
            [100.5,101]
        ]
    else:
        regions = [
            [117,117.6],
            [119.6,120],
            [120.6,122.5],
            [124.5,128.5],
            [131.7,140],
            [134.2,137],
            [137.8,139.6],
            [142.8,143.25],
            [144,145.5],
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

    good, = np.where((spatflat > 0.7) & (spatflat < 1.3) )
    print('Good pixels ', good)
    c = 299792458.e+6 # um/s
    for j in good:
        for i in range(16):
            w = waves[:,j,i]
            #print(np.nanmax(w))
            dw = dwaves[:,j,i]
            dnu = c /w * dw/w
            sf = np.interp(waves[:,j,i], wflat, specflat[:,j,i])
            #s =  error[:,i,j//5,j%5]/dnu/response(waves[:,j,i])*np.sqrt(0.12/900)/spatflat[j]
            s =  error[:,i,j//5,j%5]/dnu/response(waves[:,j,i])*np.sqrt(exptime/900)/spatflat[j]/sf
            idx = np.ones(np.size(w), dtype = bool)
            for r  in regions:
                m = (w < r[0]) | (w > r[1])
                idx *= m
            m = (s > 0.01) & (s < 10); idx *= m
            #m = (w < 71) & (s < 0.6); idx[m]=0
            #m = (w > 73) & (w<110) & (s > 0.6); idx[m]=0
            #m = (w > 71.5)  & (s > 1.0); idx[m]=0
            #m = (s > 1.5); idx[m]=0
            wtot.append(w[idx])
            stot.append(s[idx])
            ax.plot(w[idx],s[idx],'.',markersize=0.8)
            ax.plot(w[~idx],s[~idx],'.',markersize=0.5,color='red')
    

    ax.set_ylim([0,4])
    wtot = np.concatenate(wtot)
    stot = np.concatenate(stot)
    print('limits in wavelength ',np.nanmin(wtot),np.nanmax(wtot))
    idx = np.argsort(wtot)
    wtot=wtot[idx]
    stot=stot[idx]
    u, indices = np.unique(wtot, return_index=True)
    wtot = wtot[indices]
    stot = stot[indices]

    x= wtot
    y= stot
    if array == 'Blue':
        if order == 1:
            t = np.array([68.0, 70.0, 71.5, 72.5,  74.,  77.7,  81.3,   84.9, 88.5,  
                          92.1,  95.7,  99.3, 102.9, 106.5, 110.1, 113.6, 117.2, 120.8])
            wr = np.arange(np.nanmin(wtot),np.nanmax(wtot),.3)
            ax.set_xlim(60,130)
        else:
            delta = np.nanmedian(wtot[1:]-wtot[:-1])
            diff = np.nanmax(wtot)-np.nanmin(wtot)
            t = np.arange(np.nanmin(wtot)+1,np.nanmax(wtot)-delta,diff/15.)
            wr = np.arange(np.nanmin(wtot),np.nanmax(wtot),.2)
            ax.set_xlim(48,78)
    else:
        t = np.array([118,120,123,130,140,142.5,146,148,150,152,
                      155,161,165,170,175,180,185,190,195,200,202,205])
        wr = np.arange(np.nanmin(wtot),np.nanmax(wtot),1)
        ax.set_xlim(110,210)
    idx = (t > np.nanmin(wtot)) & (t < np.nanmax(wtot))
    t = t[idx]
    #print('t ',t)
    sensitivity = LSQUnivariateSpline(x,y,t)
    

    # Rejecting outliers
    for k in range(3):
        residual = y - sensitivity(x)
        med = np.nanmedian(residual)
        mad = np.nanmedian(np.abs(residual - med))
        idx = np.abs(residual-med) <  3*mad
        sensitivity = LSQUnivariateSpline(x[idx],y[idx],t)
        #ax.plot(x[~idx],y[~idx],'.',color='black')
        x = x[idx]; y = y[idx]

    sr = sensitivity(wr)
    ax.plot(wr,sr,color='lime')
    #ax.set_xlim(np.nanmin(wtot),np.nanmax(wtot))
    # ax.set_ylim([0,1.5])
    ax.grid()
    plt.show()
    print(np.nanmean(sr))

    data = []
    data.append(wr)
    data.append(sr)
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
