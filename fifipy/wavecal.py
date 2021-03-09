# -*- coding: utf-8 -*-

def reduceData(rootdir, names=None):
    """
    Reduction of wavelength calibration data taken in the lab

    Parameters
    ----------
    rootdir : TYPE string
        DESCRIPTION. directory where raw data exists

    Returns
    -------
    None.

    """

    from glob import glob as gb
    from fifipy.fit import computeSpectra
    from fifipy.calib import computeAllWaves
    from fifipy.io import saveSlopeFits, readData
    import numpy as np
    import os
    import re
    
    # Create new directory to save reduced data
    if not os.path.exists(rootdir+'Reduced/'):
        os.makedirs(rootdir+'Reduced/')
    
    if names == None:
        names = np.arange(1,50)
    for channel in ['sw','lw']:
        for name in names:
            filenames = '*GC'+str(name)+'-*'+channel+'.fits'
            files = sorted(gb(os.path.join(rootdir, '**', filenames)))
            if len(files) > 0:
                # Check sequence of grating positions
                gratpos = [int(re.search('STRT_B-(.+?)_', file).group(1)) for file in files]
                gratpos = np.array(gratpos)
                dgrat = gratpos[1:]-gratpos[:-1]
                mask = dgrat > np.nanmedian(dgrat)*3
                # Break into pieces
                if np.sum(mask) > 0:
                    idx, = np.argwhere(dgrat > np.nanmedian(dgrat)*5)+1
                    idx = np.append(idx, len(files))
                else:
                    idx = [len(files)]
                id0 = 0
                for kf, idi in enumerate(idx):
                    ifiles = files[id0:idi]
                    id0 = idi
                    print('\nIn GC', name, kf, ' there are ', len(ifiles), channel+' files')
                    gpos, specs = computeSpectra(ifiles, telSim=True)
                    aor, hk, gratpos, flux = readData(ifiles[0])
                    obsdate,telpos,pos,xy,an,za,alti,wv = hk
                    detchan, order, dichroic, ncycles, nodbeam, filegpid, filenum = aor
                    wave, dwave = computeAllWaves(gpos, dichroic, obsdate, detchan, order)
                    outname = rootdir+'Reduced/'
                    if channel == 'sw':
                        outname += 'B'
                    else:
                        outname += 'R'
                    sname = '{0:02d}'.format(name)
                    outname += str(order)+'_'+str(dichroic)+'_GC'+sname+'_'+str(kf)+'.fits'
                    saveSlopeFits(gpos, dichroic, obsdate, detchan, order, specs, wave, dwave, outname)                            

def mergeFiles(file1,file2,fileout):
    """
    Merge two reduced files if the grating is contiguous

    Parameters
    ----------
    file1 : TYPE
        DESCRIPTION. 1st input file
    file2 : TYPE
        DESCRIPTION. 2nd input file
    fileout : TYPE
        DESCRIPTION. output file

    Returns
    -------
    None.

    """
    import numpy as np
    from astropy.io import fits
    from fifipy.io import saveSlopeFits
    try:
        with fits.open(file1) as hdl:
            g1 = hdl['Grating Position'].data
            w1 = hdl['WAVE'].data
            dw1 = hdl['DWAVE'].data
            specs1 = hdl['SPECS'].data
            header = hdl[0].header
            channel1 = header['CHANNEL']
            order1 = header['ORDER']
            dichroic1 = header['DICHROIC']
            obsdate1 = header['OBSDATE']
    except:
        print('Check if '+file1+' exists')
        return
    
    try:
        with fits.open(file2) as hdl:
            g2 = hdl['Grating Position'].data
            w2 = hdl['WAVE'].data
            dw2 = hdl['DWAVE'].data
            specs2 = hdl['SPECS'].data
            header = hdl[0].header
            channel2 = header['CHANNEL']
            order2 = header['ORDER']
            dichroic2 = header['DICHROIC']
            obsdate2 = header['OBSDATE']
    except:
        print('Check if '+file2+' exists')
        return
    
    if (order1 == order2) & (dichroic1 == dichroic2) & (channel1 == channel2):
        g = np.concatenate((g1,g2), axis=0)
        w = np.concatenate((w1,w2), axis=0)
        dw = np.concatenate((dw1,dw2), axis=0)
        specs = np.concatenate((specs1,specs2), axis=0)
        saveSlopeFits(g, dichroic1, obsdate1, channel1, order1, specs, w, dw, fileout) 
    return
    

def wlimits(w, i=None):
    """
    Parameters
    ----------
    w : TYPE: array
        DESCRIPTION: wavelength of the spectral cube

    Returns
    -------
    wcommon : TYPE array
        DESCRIPTION: min and max of common wavelength range
    wrange : TYPE array
        DESCRIPTION: min and max of total wavelength range

    """
    import numpy as np

    # 1) Find common wavelength range
    if i == None:
        minw = np.empty((16,25))
        maxw = np.empty((16,25))
        for i in range(16):
            for j in range(25):
                minw[i,j] = np.nanmin(w[:,j,i])
                maxw[i,j] = np.nanmax(w[:,j,i])

        wcommon = [np.nanmax(minw), np.nanmin(maxw)]

        # 2) Find total wavelength range
        wrange = [np.nanmin(w), np.nanmax(w)]
    else:
        wrange = [np.nanmin(w[:,:,i].ravel()), np.nanmax(w[:,:,i].ravel())]
        wcommon = wrange
    
    return wcommon, wrange

def lineclean(x, y):
    """
    The code remove outliers from a positive spectrum

    Parameters
    ----------
    x : TYPE array
        DESCRIPTION. grating positions
    y : TYPE array
        DESCRIPTION. flux

    Returns
    -------
    None.

    """
    #from scipy.signal import medfilt
    import numpy as np
    #ysmooth = y.copy()
    # Filter out sudden variation of flux
    #for k in range(2):
    #    dy = ysmooth - medfilt(ysmooth,3)
    #    med = np.nanmedian(dy)
    #    mad = np.nanmedian(np.abs(dy - med))
    #    y3 = np.abs(dy) > 5*mad
    #    ysmooth[y3] = np.nan 
    # Eliminate values under the median of the lowest 50% 
    med = np.nanmedian(y)
    med50 = np.nanmedian(y[y < med])
    y3 = (y < (med50 * 0.5)) | (y < 0)
    y[y3] = np.nan 
    #ysmooth[y3] = np.nan
    # Interpolate spectrum
    idnan = np.isfinite(y)
    if np.sum(~idnan) > 0:
        ispec = np.interp(x[~idnan], x[idnan],y[idnan])
        y[~idnan] = ispec
    #idnan = np.isfinite(ysmooth)
    #if np.sum(~idnan) > 0:
    #    ispec = np.interp(x[~idnan], x[idnan],ysmooth[idnan])
    #    ysmooth[~idnan] = ispec
    #return ysmooth
        
        
def searchtop(g,f,cen):
    import numpy as np
    
    nk = 100
    ng = len(g)
    imax = np.argmin(np.abs(g-cen))
    fmax = f[imax]
    for k in range(nk):
        i1 = imax-3
        i2 = imax+3
        delta = 0
        #if i1 < 0:
        #    i1 = 0
        #if i2 > ng-1:
        #    i2 = ng-1
        if (i1 >= 0) & (i2 < ng):
            if imax > i1+1:
                if np.nanmedian(f[i1:imax]) > fmax:
                    delta = -1
                    fmax = np.nanmedian(f[i1:imax])
            if imax < i2-1:
                if np.nanmedian(f[imax+1:i2]) > fmax:
                    delta = +1
                    fmax = np.nanmedian(f[imax+1:i2])
        if delta == 0:
            return imax
        else:
            imax += delta
        
    return imax
    


def fitLines(wlines, fwhms, g, w, specs, i, j):
    
    import numpy as np
    from fifipy.wavecal import lineclean, searchtop
    from lmfit.models import PseudoVoigtModel, QuadraticModel
    
    interpg = np.interp(wlines, w[:,j,i], g)
    #interpg_ = np.interp(wlines-fwhms, w[:,j,i], g)
    #dg = np.abs(interpg_-interpg)
    # Fit with new guess values
    gmedian = np.nanmedian(g)
    cen = interpg - gmedian
    wid = fwhms * 0.5
    x = g.copy() - gmedian
    y = specs[:,i,j].copy()
    lineclean(x, y)
    # Estimate of intercept and slope
    mask = np.isfinite(y)
    for c,d in zip(cen,wid):
        mask &= np.abs(x-c) > 3*d
    xx = x[mask]
    yy = y[mask]
    intercept = np.nanmedian(yy)
    slope = np.nanmedian((yy[1:]-yy[:-1])/(xx[1:]-xx[:-1]))
    # Continuum
    model = QuadraticModel(prefix='q_')
    params = model.make_params()
    params['q_a'].set(0, vary=True)
    params['q_b'].set(slope, vary=True)
    params['q_c'].set(intercept, vary=True)
    result = model.fit(yy, params, x=xx, method='leastsq')
    a = result.params['q_a'].value
    b = result.params['q_b'].value
    c = result.params['q_c'].value
    continuum = a * x**2 + b * x + c  
    
    # Define the model
    y -= continuum
    model = PseudoVoigtModel(prefix='l0_')
    if len(cen) > 1:
        for k in range(1, len(cen)):
            model += PseudoVoigtModel(prefix='l' + str(k) + '_')

    params = model.make_params()
    ncen = len(cen)
    for k, (c,d) in enumerate(zip(cen,wid)):
        li = 'l' + str(k) + '_'
        if (c > np.min(x)+3*d) & (c < np.max(x)-3*d):
            imax = searchtop(x, y, c)
            if np.abs(c - x[imax]) < d:
                #print('in ', c, x[imax])
                c = x[imax]
            #else:
                #print('out ', c)
            if ncen > 1:
                if k > 0:
                    distance = cen[k]-cen[k-1]
                    if distance < d*2:
                        d = distance/3
                if k < ncen-1:
                    distance = cen[k+1]-cen[k]
                    if distance < d*2:
                        d = distance/3
            a = y[imax] # - (intercept + slope * c)
            a *= np.sqrt(2*np.pi) * d/2.355
            clow = c - d/2
            chigh = c + d/2
            params[li + 'center'].set(c, min=clow, max=chigh)
            params[li + 'sigma'].set(d, min=d*0.5, max=d*2)
            params[li + 'fraction'].set(0.0, vary=True)
            params[li + 'amplitude'].set(a, min=a*0.5, max=a*3)
        else:
            a = 1
            params[li + 'center'].set(c, vary=False)
            params[li + 'sigma'].set(d, vary=False)
            params[li + 'fraction'].set(0.0, vary=False)
            params[li + 'amplitude'].set(a)
        

    result = model.fit(y, params, x=x, method='leastsq')

    centers = []
    fwhms = []
    amplitudes = []
    cerrors = []
    fractions = []
    for k in range(len(cen)):
        li = 'l' + str(k) + '_'
        centers.append(result.params[li+'center'].value + gmedian)
        cerrors.append(result.params[li+'center'].stderr)
        fwhms.append(result.params[li+'fwhm'].value)
        amplitudes.append(result.params[li+'amplitude'].value)
        fractions.append(result.params[li+'fraction'].value)
        
    # Save also fractions and use them as starting point !
    return j, centers, cerrors, fwhms, amplitudes, fractions, result.best_fit+continuum
    

def fitData(datafile, plot=True, multi=True):
    """
    Fit data from a reduced file using lines from the database

    Parameters
    ----------
    datafile : TYPE string
        DESCRIPTION. file containing reduced data

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from astropy.io import fits
    import pandas as pd
    import numpy as np
    import os
    from lmfit.models import PseudoVoigtModel, QuadraticModel
    from fifipy.wavecal import wlimits, lineclean
    from dask import delayed, compute
    import time
    
    # Read data
    try:
        with fits.open(datafile) as hdl:
            g = hdl['Grating Position'].data
            w = hdl['WAVE'].data
            #dw = hdl['DWAVE'].data
            specs = hdl['SPECS'].data
            header = hdl[0].header
            detchan = header['CHANNEL']
            order = header['ORDER']
            #dichroic = header['DICHROIC']
            #obsdate = header['OBSDATE']
        fileroot = os.path.splitext(datafile)[0]
    except:
        print ('There is not such a file: ', datafile)
        
    # Read appropriate line database
    print(detchan, order, '  ', end='')
    path0, file0 = os.path.split(__file__)
    linedata = os.path.join(path0, 'data', 'water'+detchan+str(order)+'.csv')
    lines = pd.read_csv(linedata, delimiter=',',header=0,
                    names=['wave','fwhm_air','fwhm_h2o','good','fwhm_isu'])

    #wcommon, wrange = wlimits(w)
    # Lines in the range
    wlines = lines.wave.values
    good = lines.good.values
    fwhms  = lines.fwhm_isu.values
    #idx =  (wrange[0] < wlines) & (wlines < wrange[1])
    
    #idxcommon = (wcommon[0] < wlines) & (wlines < wcommon[1])
    # Here we check if there are lines common to all the pixels
    # Maybe we should be less restrictive and accept observations 
    # with good lines in the total range
    #if np.sum(idxcommon) == 0:
    #    print('No common lines for all the pixels')
        #return None, None
    #nlines = np.sum(idxcommon)
    
    # Variables to store wavelength and grating positions
    #wavepos = np.empty((nlines, 25, 16))
    #wavegood = np.empty((nlines, 25,16))
    #gratpos = np.empty((nlines, 25, 16))
    #gerrpos = np.empty((nlines, 25, 16))
    #gratamp = np.empty((nlines, 25, 16))
    #outfile = open(fileroot+'.csv', mode='w')
    
    with PdfPages(fileroot+'.pdf') as pdf:
        with open(fileroot+'.csv', mode='w') as outfile:
            if plot:
                print('Beginning of plots')
    
            # Save fitted data in csv file
            outformat = '\n{0:d},{1:d},{2:.4f},{3:.1f},{4:.1f},{5:.1f},{6:.4f},{7:.4f},{8:d}'
            gmin = np.nanmin(g)
            gmax = np.nanmax(g)
            
            for i in range(16):
                if plot:
                    print('Spexel ', i)
                else:
                    print('.', end='')
                wcommon, wrange = wlimits(w, i)
                idx = (wlines > wrange[0]-0.5) & (wlines < wrange[1]+0.5)
                nidx = np.sum(idx)
                # Skip if no lines are present
                if nidx > 0:
                    wpos = wlines[idx]
                    wgood = good[idx]
                    fwhm = fwhms[idx]
                    # Comment: the dead pixels should be skipped !
                    # Comment: to make it faster, we could pass all the 25x16 pixels ? 
                    # In this case we have to postpone the plots, maybe it would be much more efficient..
                    # print('FWHM ', fwhm)
                    if multi == True:
                        linefit = [delayed(fitLines)(wpos, fwhm, g, w, specs, i, j) for j in range(25)]        
                        linesfit = compute(* linefit, scheduler='processes')
                    else:
                        linesfit = []
                        for j in range(25):
                            lfit = fitLines(wpos, fwhm, g, w, specs, i, j)
                            linesfit.append(lfit)          
                    # Unravel and plot/save
                    fig,axs = plt.subplots(5,5, figsize=(13,13),sharex=True,sharey=True)
                    for lfit in linesfit:
                        j, centers, errors, ofwhms, amplitudes, fractions, bestfit = lfit
                        ax = axs[j//5][j%5]
                        ax.plot(g, specs[:,i,j], '.')
                        for c in centers:
                            ax.axvline(c, linestyle=':',color='grey')
                        ax.plot(g, bestfit)
                        ax.set_xlim(gmin, gmax)
                        ax.grid()
                        ax.text(0.7,0.7,'['+str(i)+','+str(j)+']', transform=ax.transAxes)
                        # Save this for later to avoid excessive garbage collection
                        for wp, gp, ge, gf, ga, wg, fr in zip(wpos, centers, errors, ofwhms, amplitudes, wgood, fractions):
                            # Write a line if inside the limits by at least one sigma
                            if (ge is not None):
                                if (gp > gmin+gf) & (gp < gmax-gf) & (ga > 0):
                                    ax.axvline(gp, linestyle=':', color='skyblue')
                                    outfile.write(outformat.format(j,i, wp,gp,ge,gf,ga,fr,int(wg)))
        
                    plt.subplots_adjust(wspace=0, hspace=0)
                    # Plot or save to a pdf file
                    if plot:
                        plt.show()
                    pdf.savefig(fig)
                    plt.close(fig)
            if plot:
                print('End of plots.') 
            # Trick to flash the memory, otherwise memory leaks kill the code
            plt.show(block=False)
            time.sleep(5)
            plt.close('all')
    return


                                  
def gratingModel1(p, gratpos, pixel, data):
    """
    Function to minimize to estimate the grating formula parameters

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    xdata : TYPE  tuple
        DESCRIPTION. Contains grating position, pixel, module, and order
    data : TYPE
        DESCRIPTION. Expected wavelengths

    Returns
    -------
    TYPE
        DESCRIPTION. Difference between modeled and expected wavelength

    """
    import numpy as np
    # Parameters
    g, ISOFF, gamma = p['g'], p['ISOFF'], p['gamma']
    PS, QOFF, QS = p['PS'], p['QOFF'], p['QS']
    pix = pixel + 1
    
    # Model (Add one to pixel, since they are counted 1,16)
    phi = 2. * np.pi * (gratpos + ISOFF) / 2.0 ** 24
    sign = np.sign(pix - QOFF)
    delta = PS  * (pix - 8.5) + sign * QS * (pix - QOFF)**2
    #delta = PS  * (pixel + 1 - 8.5) + QS * (pix - QOFF) ** 3
    alpha = phi + gamma + delta
    beta = phi - gamma
    model = 1000. * g * (np.sin(alpha) + np.sin(beta))
    
    return model - data

def gratingModel2(p, gratpos, pixel, data):
    """
    Function to minimize to estimate the grating formula parameters

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    xdata : TYPE  tuple
        DESCRIPTION. Contains grating position, pixel, module, and order
    data : TYPE
        DESCRIPTION. Expected wavelengths

    Returns
    -------
    TYPE
        DESCRIPTION. Difference between modeled and expected wavelength

    """
    import numpy as np
    # Parameters
    g, ISOFF, gamma = p['g'], p['ISOFF'], p['gamma']
    PS, QOFF, QS = p['PS'], p['QOFF'], p['QS']
    order = 2
    
    # Model (Add one to pixel, since they are counted 1,16)
    phi = 2. * np.pi * (gratpos + ISOFF) / 2.0 ** 24
    sign = np.sign(pixel + 1 - QOFF)
    delta = PS  * (pixel + 1 - 8.5) + sign * QS * (pixel + 1 - QOFF) ** 2
    #delta = PS  * (pixel + 1 - 8.5) +  QS * (pixel + 1 - QOFF) **3
    alpha = phi + gamma + delta
    beta = phi - gamma
    model = 1000. * g / order * (np.sin(alpha) + np.sin(beta))
    
    return model - data

def dGratingModel1(p, gratpos, pixel, data):
    import numpy as np
    # Parameters 
    g, ISOFF, gamma = p['g'], p['ISOFF'], p['gamma']
    PS, QOFF, QS = p['PS'], p['QOFF'], p['QS']

    f = 2 * np.pi / 2**24
    phi = f * (gratpos + ISOFF)
    sign = np.sign(pixel + 1 - QOFF)
    delta = (pixel + 1 - 8.5) * PS + sign * (pixel + 1 - QOFF) ** 2 * QS
    #delta = (pixel + 1 - 8.5) * PS +  (pixel + 1 - QOFF) ** 3 * QS
    alpha = phi + gamma + delta
    beta = phi - gamma
    
    d_g = 1000. * (np.sin(alpha) + np.sin(beta))
    d_ISOFF = 1000. * g * f * (np.cos(alpha) + np.cos(beta))
    d_gamma = 1000. * g * (-np.cos(alpha) + np.cos(beta))
    d_PS = 1000 * g  * (pixel + 1 - 8.5) * np.cos(alpha)
    d_delta = -2 * QS * sign * (pixel + 1 - QOFF)
    #d_delta = -3 * QS *  (pixel + 1 - QOFF)**2
    d_QOFF = 1000 * g * np.cos(alpha) * d_delta
    d_QS = 1000 * g * sign * (pixel + 1 - QOFF)**2 * np.cos(alpha)
    
    deriv = []
    if p['g'].vary:
        deriv.append(d_g)
    if p['ISOFF'].vary:
        deriv.append(d_ISOFF)
    if p['gamma'].vary:
        deriv.append(d_gamma)
    if p['PS'].vary:
        deriv.append(d_PS)
    if p['QOFF'].vary:
        deriv.append(d_QOFF)
    if p['QS'].vary:
        deriv.append(d_QS)
    deriv = np.array(deriv)
    
    return deriv

def dGratingModel2(p, gratpos, pixel, data):
    import numpy as np
    # Parameters 
    g, ISOFF, gamma = p['g'], p['ISOFF'], p['gamma']
    PS, QOFF, QS = p['PS'], p['QOFF'], p['QS']

    f = 2 * np.pi / 2**24
    phi = f * (gratpos + ISOFF)
    sign = np.sign(pixel + 1 - QOFF)
    delta = (pixel + 1 - 8.5) * PS + sign * (pixel + 1 - QOFF) ** 2 * QS
    #delta = (pixel + 1 - 8.5) * PS +  (pixel + 1 - QOFF) ** 3 * QS
    alpha = phi + gamma + delta
    beta = phi - gamma
    
    d_g = 500. * (np.sin(alpha) + np.sin(beta))
    d_ISOFF = 500. * g * f * (np.cos(alpha) + np.cos(beta))
    d_gamma = 500. * g * (-np.cos(alpha) + np.cos(beta))
    d_PS = 500 * g  * (pixel + 1 - 8.5) * np.cos(alpha)
    d_delta = -2 * QS * sign * (pixel + 1 - QOFF)
    #d_delta = -3 * QS *  (pixel + 1 - QOFF)**2
    d_QOFF = 500 * g * np.cos(alpha) * d_delta
    d_QS = 500 * g * sign * (pixel + 1 - QOFF)**2 * np.cos(alpha)
    
    deriv = []
    if p['g'].vary:
        deriv.append(d_g)
    if p['ISOFF'].vary:
        deriv.append(d_ISOFF)
    if p['gamma'].vary:
        deriv.append(d_gamma)
    if p['PS'].vary:
        deriv.append(d_PS)
    if p['QOFF'].vary:
        deriv.append(d_QOFF)
    if p['QS'].vary:
        deriv.append(d_QS)
    deriv = np.array(deriv)
    
    return deriv



def computeWavCal(pixel, module, wavepos, gratpos, channel, order,
                  fixPS=False, fixQS=False, fixQOFF=False, fixgamma=False,
                  fixg=None, fixISOFF=None):
    """
    Estimate parameters for wavelength calibration from measurement of grating
    positions of known water vapor lines.

    Parameters
    ----------
    pixel : TYPE
        DESCRIPTION.
    module : TYPE
        DESCRIPTION.
    wavepos : TYPE
        DESCRIPTION.
    gratpos : TYPE
        DESCRIPTION.
    channel : TYPE
        DESCRIPTION.
    order : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from lmfit import Parameters, Minimizer
    import numpy as np
    import pandas as pd
    import os
    # Select a module
    g = []
    gamma = []
    ISOFF = []
    PS = []
    QOFF = []
    QS = []
    path0, file0 = os.path.split(__file__)
    if channel == 'R':
        deadpixelfile = os.path.join(path0, 'data', 'deadRED.csv')
    else:
        deadpixelfile = os.path.join(path0, 'data', 'deadBLUE.csv')
    dead = pd.read_csv(deadpixelfile, delimiter=',',header=0,
                    names=['spaxel','spexel'])      
    spaxels = dead.spaxel.values
    spexels = dead.spexel.values

    for j in range(25):
        slitPos = 25 - 6 * (j // 5) + j % 5
    
        fit_params = Parameters()
        if channel == 'R':
            a = 424.6828
            NP = 14.2926
            g0 = 0.117155
            g_est = g0 * np.cos(np.arctan2(slitPos - NP, a)) 
            if fixg is None:
                fit_params.add('g', value=0.11765)
                #fit_params.add('g', value=g_est)
            else:
                fit_params.add('g', value=fixg[j], vary=False)
            if fixISOFF is None:
                fit_params.add('ISOFF', value=1150258.0)
            else:
                fit_params.add('ISOFF', value=fixISOFF[j], vary=False)
            if fixgamma == False:
                #fit_params.add('gamma', value=0.0167200, min=0)#min=0.0160,max=0.0170) # 0.0185828
                fit_params.add('gamma', value=0.0167200, min=0.016, max=0.017)#min=0.0160,max=0.0170) # 0.0185828
            else:
                fit_params.add('gamma', value=fixgamma, vary=False)
            #fit_params.add('QOFF', value=6.150454)
            if fixPS == False:
                fit_params.add('PS', value=0.0006)
            else:
                fit_params.add('PS', value=fixPS, vary=False)
            if fixQOFF == False:
                fit_params.add('QOFF', value=8.5,min=0.5,max=15.5)
            else:
                fit_params.add('QOFF', value=fixQOFF, vary=False)
            if fixQS == False:
                fit_params.add('QS', value=1.6e-06)
            else:
                fit_params.add('QS', value=fixQS, vary=False)
        else:
            a = 13.9389
            NP = 13.9389
            g0 = 0.082577
            g_est = g0 * np.cos(np.arctan2(slitPos - NP, a))
            if fixg is None:
                fit_params.add('g', value=0.083333)
                #fit_params.add('g', value=g_est)
            else:
                fit_params.add('g', value=fixg[j], vary=False)
            if fixISOFF is None:
                fit_params.add('ISOFF', value=1075019.0)
            else:
                fit_params.add('ISOFF', value=fixISOFF[j], vary=False)
            if fixgamma == False:
                #fit_params.add('gamma', value=0.0089008, min=0)#min=0.0088,max=0.0090) # 0.011126
                fit_params.add('gamma', value=0.0089008, min=0.008, max=0.01)#min=0.0088,max=0.0090) # 0.011126
            else:
                fit_params.add('gamma', value=fixgamma, vary=False)
            if fixPS == False:            
                fit_params.add('PS', value=0.0006)
            else:
                fit_params.add('PS', value=fixPS, vary=False)
            if fixQOFF == False:
                fit_params.add('QOFF', value=8.5)
            else:
                fit_params.add('QOFF', value=fixQOFF, vary=False)
            if fixQS == False:
                fit_params.add('QS', value=9.5e-06)
            else:
                fit_params.add('QS', value=fixQS, vary=False)

        if j in spaxels:
            # remove also dead pixels
            spex = spexels[spaxels == j]
            idx = module == j
            for s in spex:
                idx &= (pixel != s)
        else:
            idx = module == j
        
        if order == 1:
            min1 = Minimizer(gratingModel1, fit_params, 
                             fcn_args=(gratpos[idx], pixel[idx]), 
                             fcn_kws={'data':  wavepos[idx]})
            out = min1.leastsq(Dfun=dGratingModel1, col_deriv=True)
            #out = minimize(gratingModel1, fit_params, 
            #           args=(gratpos[idx], pixel[idx]), kws=kws, method='leastsq')
        else:
            #kws = {'data': wavepos[idx]}
            #out = minimize(gratingModel2, fit_params, 
            #           args=(gratpos[idx], pixel[idx]), kws=kws, method='leastsq')
            min2 = Minimizer(gratingModel2, fit_params, 
                             fcn_args=(gratpos[idx], pixel[idx]), 
                             fcn_kws={'data':  wavepos[idx]})
            out = min2.leastsq(Dfun=dGratingModel2, col_deriv=True)
       
        outpar = out.params
        g.append(outpar['g'].value)
        gamma.append(outpar['gamma'].value)
        QOFF.append(outpar['QOFF'].value)
        PS.append(outpar['PS'].value)
        QS.append(outpar['QS'].value)
        ISOFF.append(outpar['ISOFF'].value)
    
    g = np.array(g)
    gamma = np.array(gamma)
    QOFF = np.array(QOFF)
    PS = np.array(PS)
    ISOFF = np.array(ISOFF)
    QS = np.array(QS)
    
    return g,gamma,QOFF, PS,QS, ISOFF

def computeWavelength(spexel, spaxel, order, coeffs, gratpos):
    """
    Computes the wavelength from a set of coefficients

    Parameters
    ----------
    spexel : TYPE
        DESCRIPTION. spectral pixel
    spaxel : TYPE
        DESCRIPTION. spatial module
    order : TYPE
        DESCRIPTION. spectral order
    coeffs : TYPE
        DESCRIPTION. g0,NP,a,ISF,gamma,PS,QOFF,QS,ISOFF 
    gratpos : TYPE
        DESCRIPTION. grating position

    Returns
    -------
    w : TYPE
        DESCRIPTION. wavelength
    dw : TYPE
        DESCRIPTION. wavelength interval

    """
    import numpy as np
    
    g0,NP,a,ISF,gamma,PS,QOFF,QS,ISOFF = coeffs
    pix = spexel + 1.
    module = spaxel
    phi = 2. * np.pi * ISF * (gratpos + ISOFF[module]) / 2.0 ** 24
    sign = np.sign(pix - QOFF)
    delta = (pix - 8.5) * PS + sign * (pix - QOFF) ** 2 * QS
    slitPos = 25 - 6 * (module // 5) + module % 5
    # g = g0 * np.cos(np.arctan2(slitPos - NP, a)) 
    
    g = g0 * (1 - 0.5 * ((slitPos - NP)/a)**2)
    
    w = 1000. * (g / order) * (np.sin(phi + gamma + delta) + np.sin(phi - gamma))
    dw = 1000. * (g / order) * (PS + 2. * sign * QS * (pix - QOFF)) * np.cos(phi + gamma + delta)

    return w, dw

