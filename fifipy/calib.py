def waveCalOld(gratpos, dichroic, obsdate, array, order):
    import numpy as np
    import pandas as pd
    import os

    '''
    Usage:
    l,lw = waveCal( gratpos=1496600, order=1, array='RED',dichroic=105,
                   obsdate='2015-03-12T04:41:33')
    '''

    if array == 'RED':
        channel = 'R'
    else:
        if order == '1':
            channel = 'B1'
        else:
            channel = 'B2'

    # Extract month and year from date
    year = obsdate.split('-')[0]
    month = obsdate.split('-')[1]
    odate = int(year[2:] + month)

    path0, file0 = os.path.split(__file__)
    wvdf = pd.read_csv(path0 + '/data/CalibrationResults.csv', header=[0, 1])
    ndates = (len(wvdf.columns) - 2) // 4
    dates = np.zeros(ndates)
    for i in range(ndates):
        dates[i] = wvdf.columns[2 + i * 4][0]
    # print('Dates: ', dates)

    # Select correct date
    for i, date in enumerate(dates):
        if date < odate:
            pass
        else:
            break
    cols = range(2 + 4 * i , 2 + 4 * i + 4)
    # print('Selected columns are: ', cols)
    w1 = wvdf[wvdf.columns[cols]].copy()
    if channel == 'R':
        if dichroic == 105:
            co = w1.columns[0]
        else:
            co = w1.columns[1]
    elif channel == 'B1':
        co = w1.columns[2]
    else:
        co = w1.columns[3]
    g0 = w1.iloc[0][co]
    NP = w1.iloc[1][co]
    a = w1.iloc[2][co]
    ISF = w1.iloc[3][co]
    gamma = w1.iloc[4][co]
    PS = w1.iloc[5][co]
    QOFF = w1.iloc[6][co]
    QS = w1.iloc[7][co]
    ISOFF = w1.iloc[8:][co].values

    pix = np.arange(16) + 1.
    result = np.zeros((25, 16))
    result_dwdp = np.zeros((25, 16))
    for module in range(25):
        phi = 2. * np.pi * ISF * (gratpos + ISOFF[module]) / 2.0 ** 24
        sign = np.sign(pix - QOFF)
        delta = (pix - 8.5) * PS + sign * (pix - QOFF) ** 2 * QS
        slitPos = 25 - 6 * (module // 5) + module % 5
        g = g0 * np.cos(np.arctan2(slitPos - NP, a))  # Careful with arctan
        lambd = 1000. * (g / order) * (np.sin(phi + gamma + delta) +
                                       np.sin(phi - gamma))
        dwdp = 1000. * (g / order) * (PS + 2. * sign * QS *
                                      (pix - QOFF)) * np.cos(phi + gamma + delta)
        result[module, :] = lambd
        result_dwdp[module, :] = dwdp

    return result, result_dwdp

def waveCal(gratpos, dichroic, obsdate, array, order):
    import numpy as np
    import pandas as pd
    import os

    '''
    Usage:
    l,lw = waveCal( gratpos=1496600, order=1, array='RED',dichroic=105,
                   obsdate='2015-03-12T04:41:33')
    '''

    if array == 'RED':
        channel = 'R'
    else:
        if order == '1':
            channel = 'B1'
        else:
            channel = 'B2'

    # Extract month and year from date
    obsdate = obsdate.replace('T', '-')
    year = obsdate.split('-')[0]
    month = obsdate.split('-')[1]
    day = obsdate.split('-')[2]
    odate = year+month+day

    path0, file0 = os.path.split(__file__)
    header_list = ["Date", "ch", "g0","NP","a","PS","QOFF","QS",
               "I1","I2","I3","I4","I5","I6","I7","I8","I9","I10",
               "I11","I12","I13","I14","I15","I16","I17","I18","I19","I20",
               "I21","I22","I23","I24","I25"]

    wvdf = pd.read_csv(os.path.join(path0, 'data' ,'FIFI_LS_WaveCal_Coeffs.txt'), 
                   comment='#', delimiter='\s+', names=header_list)
    # Fixed variables
    ISF = 1
    if array == 'RED':
        gamma = 0.0167200
    else:
        gamma = 0.0089008


    # Select calibration date
    dates = np.unique(wvdf['Date'])
    #print('dates ', dates)
    for i, date in enumerate(dates):
        if date < int(odate):
            pass
        else:
            break
        
    idx = dates < int(odate)
    #print('idx ', idx)
    caldate = np.max(dates[idx])   


    # Select line in calibration file with caldate and channel
    idx = (wvdf["Date"] == caldate) & (wvdf["ch"] == channel)
    wcal = wvdf.loc[idx]

    g0 = wcal['g0'].values[0]
    NP = wcal['NP'].values[0]
    a = wcal['a'].values[0]
    PS = wcal['PS'].values[0]
    QOFF = wcal['QOFF'].values[0]
    QS = wcal['QS'].values[0]
    ISOFF = wcal.iloc[0][8:].values

    pix = np.arange(16) + 1.
    result = np.zeros((25, 16))
    result_dwdp = np.zeros((25, 16))
    for module in range(25):
        phi = 2. * np.pi * ISF * (gratpos + ISOFF[module]) / 2.0 ** 24
        sign = np.sign(pix - QOFF)
        delta = (pix - 8.5) * PS + sign * (pix - QOFF) ** 2 * QS
        slitPos = 25 - 6 * (module // 5) + module % 5
        g = g0 * np.cos(np.arctan2(slitPos - NP, a))  # Careful with arctan
        lambd = 1000. * (g / order) * (np.sin(phi + gamma + delta) +
                                       np.sin(phi - gamma))
        dwdp = 1000. * (g / order) * (PS + 2. * sign * QS *
                                      (pix - QOFF)) * np.cos(phi + gamma + delta)
        result[module, :] = lambd
        result_dwdp[module, :] = dwdp

    return result, result_dwdp


def mwaveCal(gratpos, dichroic, obsdate, array, order):
    import numpy as np
    import pandas as pd
    import os

    '''
    Usage:
    l,lw = waveCal( gratpos=1496600, order=1, array='RED',dichroic=105,
                   obsdate='2015-03-12T04:41:33')
    '''

    if array == 'RED':
        channel = 'R'
    else:
        if order == '1':
            channel = 'B1'
        else:
            channel = 'B2'

    # Extract month and year from date
    year = obsdate.split('-')[0]
    month = obsdate.split('-')[1]
    odate = int(year[2:] + month)

    path0, file0 = os.path.split(__file__)
    wvdf = pd.read_csv( path0 + '/data/CalibrationResults.csv', header=[0, 1])
    ndates = (len(wvdf.columns) - 2) // 4
    dates = np.zeros(ndates)
    for i in range(ndates):
        dates[i] = wvdf.columns[2 + i * 4][0]

    # Select correct date
    for i, date in enumerate(dates):
        if date < odate:
            pass
        else:
            break
    cols = range(2 + 4 * i, 2 + 4 * i  + 4)
    w1 = wvdf[wvdf.columns[cols]].copy()
    if channel == 'R':
        if dichroic == 105:
            co = w1.columns[0]
        else:
            co = w1.columns[1]
    elif channel == 'B1':
        co = w1.columns[2]
    else:
        co = w1.columns[3]
    g0 = w1.loc[0][co]
    NP = w1.loc[1][co]
    a = w1.loc[2][co]
    ISF = w1.loc[3][co]
    gamma = w1.loc[4][co]
    PS = w1.loc[5][co]
    QOFF = w1.loc[6][co]
    QS = w1.loc[7][co]
    ISOFF = w1.loc[8:][co].values
        
    ng = len(gratpos)
    pix = np.arange(16) + 1.
    result = np.zeros((ng, 16, 25))
    result_dwdp = np.zeros((ng, 16, 25))
    for ig, gp in enumerate(gratpos):
        for module in range(25):
            phi = 2. * np.pi * ISF * (gp + ISOFF[module]) / 2.0 ** 24
            sign = np.sign(pix - QOFF)
            delta = (pix - 8.5) * PS + sign * (pix - QOFF) ** 2 * QS
            slitPos = 25 - 6 * (module // 5) + module % 5
            g = g0 * np.cos(np.arctan2(slitPos - NP, a))  # Careful with arctan
            lambd = 1000. * (g / order) * (np.sin(phi + gamma + delta) +
                                           np.sin(phi - gamma))
            dwdp = 1000. * (g / order) * (PS + 2. * sign * QS *
                                          (pix - QOFF)) * np.cos(phi + gamma + delta)
            result[ig, :, module] = lambd
            result_dwdp[ig, :, module] = dwdp

    return result, result_dwdp


def computeAllWaves(gpos, dichroic, obsdate, detchan, order):
    from dask import delayed, compute
    import numpy as np
    wavegpos = [delayed(waveCal)(g, dichroic, obsdate, detchan, order) for g in gpos]        
    waves = compute(* wavegpos, scheduler='processes')
    wave = []
    dwave = []
    for w in waves:
        wave.append(w[0])
        dwave.append(w[1])
    return np.array(wave), np.array(dwave)

def readWindowTransmission():
    '''Read window transmission data.'''
    import os
    import numpy as np
    path0, file0 = os.path.split(__file__)
    data = np.loadtxt(path0+'/data/windowTrans.txt')    
    wt = data[:,0]
    t = data[:,1]
    return wt, t

def readFlats(channel, order, dichroic, obsdate, silent=False):
    ''' Read flats '''
    wflat, specflat, especflat = readSpecFlats(channel, order, dichroic, silent=silent)
    spatflat = readSpatFlats(channel, order, obsdate, silent=silent)
    return wflat, specflat, especflat, spatflat

def readSpecFlats(channel, order, dichroic, silent=False):
    ''' Read flats '''
    import os
    from astropy.io import fits
    path0, file0 = os.path.split(__file__)
    if channel == 'RED':
        infile = path0 + '/data/spectralFlatsR1D'+str(dichroic)+'.fits.gz'
    else:
        infile = path0 + '/data/spectralFlatsB'+str(order)+'D'+str(dichroic)+'.fits.gz'
    hdl = fits.open(infile)
    if silent == False:
        hdl.info()
    wflat = hdl['WAVE'].data
    specflat = hdl['SPECFLAT'].data
    especflat = hdl['ESPECFLAT'].data
    hdl.close()
    return wflat, specflat, especflat

def readSpatFlats(channel, order, obsdate, silent=False):
    ''' Read spatial flats.'''
    import os, re
    import numpy as np
    path0, file0 = os.path.split(__file__)
    if channel == 'RED':
        infile = path0 + '/data/spatialFlatR.txt'
    else:
        if order == 1:
            infile = path0 + '/data/spatialFlatB1.txt'
        else:
            infile = path0 + '/data/spatialFlatB2.txt'
    data = np.genfromtxt(infile,dtype='str',skip_header=1)
    dates = data[:,0].astype(int)
    spatflats = data[:, 1:].astype(float)
    # Extract month, year, and day from date
    parts = re.split('-|T|:', obsdate)
    odate = int(parts[0]+parts[1]+parts[2])
    # Select correct date
    for date, spatflat in zip(dates, spatflats):
        if date < odate:
            pass
        else:
            return spatflat
        


def readSpatFlatsOld(channel, obsdate, silent=False):
    ''' Read flats '''
    import os, re
    from astropy.io import fits
    path0, file0 = os.path.split(__file__)
    if channel == 'RED':
        infile = path0 + '/data/spatialFlatR.fits'
    else:
        infile = path0 + '/data/spatialFlatB.fits'
    hdl = fits.open(infile)
    if silent == False:
        hdl.info()
    dates = hdl['DATES'].data
    spatflats = hdl['SPATFLAT'].data
    hdl.close()
       
    # Extract month, year, and day from date
    parts = re.split('-|T|:', obsdate)
    odate = int(parts[0]+parts[1]+parts[2])
    # Select correct date
    for date, spatflat in zip(dates, spatflats):
        if date < odate:
            pass
        else:
            return spatflat


def applyFlats(waves, fluxes, channel, order, dichroic, obsdate):
    ''' Apply flats to fluxes '''
    import numpy as np
    
    wflat, specflat, especflat, spatflat= readFlats(channel, order, dichroic, obsdate, silent=True)
    for i in range(16):
        for j in range(25):
            sf = np.interp(waves[:,j,i], wflat, specflat[:,j,i])
            fluxes[:,j,i] /= sf
    for j in range(25):
        fluxes[:,j,:] /= spatflat[j]
        
    # Apply bad pixel mask
    import os
    path0, file0 = os.path.split(__file__)
    if channel == 'RED':
        bads = np.loadtxt(path0 + '/data/badpixels_202104_r.txt')
    else:
        bads = np.loadtxt(path0 + '/data/badpixels_202104_b.txt')
    for bad in bads:
        j,i = bad
        fluxes[:,np.int(j)-1,np.int(i)-1] = np.nan
    return fluxes


def readAtran(detchan, order):
    import os
    from astropy.io import fits
    path0, file0 = os.path.split(__file__)
    if detchan == 'BLUE':
        file = 'AtranBlue'+str(order)+'.fits.gz'
    else:
        file = 'AtranRed.fits.gz'
    path = path0+'/data/'
    hdl = fits.open(path+file)
    wt = hdl['WAVELENGTH'].data
    atran = hdl['ATRAN'].data
    altitudes = hdl['ALTITUDE'].data
    wvzs = hdl['WVZ'].data
    hdl.close()
    return (wt, atran, altitudes, wvzs)


# Routines to compute (with spline fitting) and save the response in a file
def computeResponse(wtot, rtot, nknots = 50, t = None):
    from scipy.interpolate import LSQUnivariateSpline
    from scipy.ndimage.filters import generic_filter
    import numpy as np
    # Order wtot
    idx = np.argsort(wtot)
    wtot = wtot[idx]
    rtot = rtot[idx]
    if t is None:
        nknots1=50
        ntot = len(wtot)-8
        idxk = 4+ntot//(nknots1-1)*np.arange(nknots1)
        t = wtot[idxk]
        tset = False
    else:
        tset = True
    x = wtot; y = rtot
    response = LSQUnivariateSpline(x,y,t)
    # Rejecting outliers
    residual = rtot - response(wtot)
    med = np.nanmedian(residual)
    mad = np.nanmedian(np.abs(residual - med))
    idx = np.abs(residual-med) <  5*mad
    if tset == False:
        xx = x[idx]
        ntot = len(xx)-8
        idxk= 4+ntot//(nknots-1)*np.arange(nknots)
        t = xx[idxk]
    response = LSQUnivariateSpline(x[idx],y[idx],t)
    
    residuals = y[idx] - response(x[idx])
    dev  = generic_filter(residuals, np.std, size=30)
    eresponse = LSQUnivariateSpline(x[idx],dev,t)
    
    return t, response, eresponse

