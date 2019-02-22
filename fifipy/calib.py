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
    year = obsdate.split('-')[0]
    month = obsdate.split('-')[1]
    odate = int(year[2:] + month)

    path0, file0 = os.path.split(__file__)
    wvdf = pd.read_csv(path0 + '/data/CalibrationResults.csv', header=[0, 1])
    ndates = (len(wvdf.columns) - 2) // 5
    dates = np.zeros(ndates)
    for i in range(ndates):
        dates[i] = wvdf.columns[2 + i * 5][0]

    # Select correct date
    i = 0
    for date in dates:
        if date < odate:
            i += 1
        else:
            pass
    cols = range(2 + 5 * (i - 1), 2 + 5 * (i - 1) + 5)
    w1 = wvdf[wvdf.columns[cols]].copy()
    if channel == 'R':
        if dichroic == 105:
            co = w1.columns[0]
        else:
            co = w1.columns[1]
    elif channel == 'B1':
        co = w1.columns[2]
    else:
        if dichroic == 105:
            co = w1.columns[3]
        else:
            co = w1.columns[4]
    g0 = w1.ix[0][co]
    NP = w1.ix[1][co]
    a = w1.ix[2][co]
    ISF = w1.ix[3][co]
    gamma = w1.ix[4][co]
    PS = w1.ix[5][co]
    QOFF = w1.ix[6][co]
    QS = w1.ix[7][co]
    ISOFF = w1.ix[8:][co].values

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


def readFlats(channel, silent=False):
    ''' Read flats '''
    import os
    from astropy.io import fits
    path0, file0 = os.path.split(__file__)
    if channel == 'RED':
        infile = path0 + '/data/RedFlats.fits.gz'
    else:
        infile = path0 + '/data/BlueFlats.fits.gz'
    hdl = fits.open(infile)
    if silent == False:
        hdl.info()
    wflat = hdl['WAVE'].data
    specflat = hdl['SPECFLAT'].data
    especflat = hdl['ESPECFLAT'].data
    spatflat = hdl['SPATFLAT'].data
    hdl.close()
    return wflat, specflat, especflat, spatflat


def applyFlats(waves, fluxes, channel):
    ''' Apply flats to fluxes '''
    import numpy as np
    wflat, specflat, especflat, spatflat = readFlats(channel, silent=True)
    for i in range(16):
        for j in range(25):
            sf = np.interp(waves[:,j,i], wflat[:,j,i], specflat[:,j,i])
            fluxes[:,j,i] /= sf
    for j in range(25):
        fluxes[:,j,:] /= spatflat[j]
        
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
    hdl.close()
    return wt, atran
