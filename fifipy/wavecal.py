# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')

def reduceData(rootdir, names=None, channels=['sw','lw'],telSim=True):
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
    
    if names is None:
        names = np.arange(1,50)
    for channel in channels:
        for name in names:
            filenames = '*_GC'+str(name)+'-*'+channel+'.fits'
            files = sorted(gb(os.path.join(rootdir, '**', filenames), recursive=True))
            filenames2 = '*_GC'+str(name)+'_?-*'+channel+'.fits'
            files2 = sorted(gb(os.path.join(rootdir, '**', filenames2), recursive=True))
            files += files2
            if len(files) > 0:
                # Check sequence of grating positions
                gratpos = [int(re.search('STRT_B-(.+?)_', file).group(1)) for file in files]
                gratpos = np.array(gratpos)
                dgrat = gratpos[1:]-gratpos[:-1]
                mask = dgrat > np.nanmedian(dgrat)*3
                # Break into pieces
                if np.sum(mask) > 0:
                    idx = np.argwhere(dgrat > np.nanmedian(dgrat)*5)[:,0]+1
                    idx = np.append(idx, len(files))
                else:
                    idx = [len(files)]
                id0 = 0
                for kf, idi in enumerate(idx):
                    ifiles = files[id0:idi]
                    id0 = idi
                    print('\nIn GC', name, kf, ' there are ', len(ifiles), channel+' files')
                    gpos, specs = computeSpectra(ifiles, telSim=telSim)
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
    
    # Case of negative fluxes 
    #ymax = np.nanmax(y)
    #if ymax < 0:
    #    y = -y
    
    med = np.nanmedian(y)
    if med < 0:
        y -= med
        med = 0
    med50 = np.nanmedian(y[y < med])
    y3 = (y < (med50 * 0.5)) | (y < 0)
    y[y3] = np.nan 
    #ysmooth[y3] = np.nan
    # Interpolate spectrum
    idnan = np.isfinite(y)
    #if ymax < 0:
    #    y[~idnan] = med
    #else:
    if np.sum(idnan) == 0:
        return
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
    #print('finite ',np.sum(mask))
    for c,d in zip(cen,wid):
        mask &= np.abs(x-c) > 3*d
        
        
    centers = []
    fwhms = []
    amplitudes = []
    cerrors = []
    fractions = []

    if np.sum(mask) < 5:
        intercept = np.nan
        slope = np.nan        
        for k in range(len(cen)):
            li = 'l' + str(k) + '_'
            centers.append(np.nan)
            cerrors.append(0)
            fwhms.append(np.nan)
            amplitudes.append(0)
            fractions.append(0)
        model = np.ones(len(g)) * np.nan
    else:
        xx = x[mask]
        yy = y[mask]
        intercept = np.nanmedian(yy)
        dx = xx[1:]-xx[:-1]
        dy = yy[1:]-yy[:-1]
        # Exclude repetitions in the data
        idx = dx != 0
        slope = np.nanmedian(dy[idx]/dx[idx])
        #slope = np.nanmedian((yy[1:]-yy[:-1])/(xx[1:]-xx[:-1]))
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
                params[li + 'sigma'].set(d, min=d*0.5)
                params[li + 'fraction'].set(0.0, vary=True)
                if a == 0:
                    params[li + 'amplitude'].set(a)
                else:
                    params[li + 'amplitude'].set(a, min=a*0.5, max=a*3)
            else:
                a = 1
                params[li + 'center'].set(c, vary=False)
                params[li + 'sigma'].set(d, vary=False)
                params[li + 'fraction'].set(0.0, vary=False)
                params[li + 'amplitude'].set(a)
            
    
        result = model.fit(y, params, x=x, method='leastsq')
    
        for k in range(len(cen)):
            li = 'l' + str(k) + '_'
            centers.append(result.params[li+'center'].value + gmedian)
            cerrors.append(result.params[li+'center'].stderr)
            fwhms.append(result.params[li+'fwhm'].value)
            amplitudes.append(result.params[li+'amplitude'].value)
            fractions.append(result.params[li+'fraction'].value) 
        model = result.best_fit + continuum
        
    # Save also fractions and use them as starting point !
    return j, centers, cerrors, fwhms, amplitudes, fractions, model

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
                            print(i,j)
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


                                  
def gratingModel1(p, gratpos, pixel, data, error=None):
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
    alpha = phi + gamma + delta
    beta = phi - gamma
    model = 1000. * g * (np.sin(alpha) + np.sin(beta))
    
    if error is None:
        return model - data
    else:
        return (model - data) / error


def gratingModel2(p, gratpos, pixel, data, error=None):
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
    
    if error is None:
        return model - data
        return model - data
    else:
        return (model - data) / error

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
                  fixPS=False, fixQS=False, fixQOFF=False, fixgamma=True,
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
    from lmfit import Parameters, Minimizer, minimize
    import numpy as np
    import pandas as pd
    import os
    from fifipy.spectra import getResolution
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
                fit_params.add('gamma', value=0.0167200, vary=False)
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
                fit_params.add('gamma', value=0.0089008, vary=False)
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
        
        if channel == 'R':
            channelorder = 'R'
        else:
            if order == 1:
                channelorder = 'B1'
            else:
                channelorder = 'B2'
        
        # Add error as R/lambda
        if np.sum(idx) > 6:
            if order == 1:
                #min1 = Minimizer(gratingModel1, fit_params, 
                #                 fcn_args=(gratpos[idx], pixel[idx]), 
                #                 fcn_kws={'data': wavepos[idx]})
                #out = min1.leastsq(Dfun=dGratingModel1, col_deriv=True)
                R = getResolution(channelorder, wavepos[idx])
                kws = {'data': wavepos[idx], 'error': wavepos[idx]/R}
                kws = {'data': wavepos[idx]}
                out = minimize(gratingModel1, fit_params, 
                               args=(gratpos[idx], pixel[idx]), kws=kws, method='leastsq')
            else:
                R = getResolution(channelorder, wavepos[idx])
                kws = {'data': wavepos[idx], 'error': wavepos[idx]/R}
                out = minimize(gratingModel2, fit_params, 
                           args=(gratpos[idx], pixel[idx]), kws=kws, method='leastsq')
                #min2 = Minimizer(gratingModel2, fit_params, 
                #                 fcn_args=(gratpos[idx], pixel[idx]), 
                #                 fcn_kws={'data': wavepos[idx]})
                #out = min2.leastsq(Dfun=dGratingModel2, col_deriv=True)
           
            outpar = out.params
            g.append(outpar['g'].value)
            gamma.append(outpar['gamma'].value)
            QOFF.append(outpar['QOFF'].value)
            PS.append(outpar['PS'].value)
            QS.append(outpar['QS'].value)
            ISOFF.append(outpar['ISOFF'].value)
        else:
            g.append(np.nan)
            gamma.append(np.nan)
            QOFF.append(np.nan)
            PS.append(np.nan)
            QS.append(np.nan)
            ISOFF.append(np.nan)
    
    g = np.array(g)
    gamma = np.array(gamma)
    QOFF = np.array(QOFF)
    PS = np.array(PS)
    ISOFF = np.array(ISOFF)
    QS = np.array(QS)
    
    return g,gamma,QOFF, PS,QS, ISOFF

def gratingModelR105(p, gratpos, pixel, module, data, error=None):
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
    ai,bi,ci = p['ai'],p['bi'],p['ci']
    g0, NP, a = p['g0'], p['NP'], p['a']
    PS, QOFF, QS = p['PS'], p['QOFF'], p['QS']
    pix = pixel + 1
    slitPos = 25 - 6 * (module // 5) + module % 5
    gamma=0.0167200
    red105 = np.array( [ -7.82734595,    6.48103643,   15.37344189,   47.39558183,   54.25017651,
              -87.78073561,  -57.87672198,  -57.02387395,  -30.75647953,  -82.13171852,
               38.17407445,   53.9293801 ,   62.15816713,   82.60265586,  -51.04419029,
               -6.0626937,   36.28682384,   42.49162215,   70.33355788, -148.78530207,
              -52.04256692 , -29.12922045,   -4.73520485,   20.72545992, -268.51481606])
    
    g = g0 * (1 - 0.5 * ((slitPos - NP)/a)**2)
    ISOFF = ai * slitPos**2 + bi * slitPos + ci - red105[module]
    # Model (Add one to pixel, since they are counted 1,16)
    phi = 2. * np.pi * (gratpos + ISOFF) / 2.0 ** 24
    sign = np.sign(pix - QOFF)
    delta = PS  * (pix - 8.5) + sign * QS * (pix - QOFF)**2
    alpha = phi + gamma + delta
    beta = phi - gamma
    model = 1000. * g * (np.sin(alpha) + np.sin(beta))
    
    if error is None:
        return model - data
    else:
        return (model - data) / error
    
def gratingModelR130(p, gratpos, pixel, module, data, error=None):
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
    ai,bi,ci = p['ai'],p['bi'],p['ci']
    g0, NP, a = p['g0'], p['NP'], p['a']
    PS, QOFF, QS = p['PS'], p['QOFF'], p['QS']
    pix = pixel + 1
    slitPos = 25 - 6 * (module // 5) + module % 5
    gamma=0.0167200
    red130 = np.array([ -12.70859072,    7.50024661,   18.53167461,   41.46400465,   52.7757175,
              -95.78015715,  -56.53938436,  -54.24399594,  -33.75992799,  -68.99733959,
               31.27967525,   53.60554151,   58.10103624,   71.69960587,  -22.11761283,
               -4.64846212 ,  38.77585613,   42.34325365,   60.40053434, -118.02749666,
              -47.8753654 ,  -24.45939546,   -4.54977914,    8.74871326, -223.38722927])
    
    g = g0 * (1 - 0.5 * ((slitPos - NP)/a)**2)
    ISOFF = ai * slitPos**2 + bi * slitPos + ci - red130[module]
    # Model (Add one to pixel, since they are counted 1,16)
    phi = 2. * np.pi * (gratpos + ISOFF) / 2.0 ** 24
    sign = np.sign(pix - QOFF)
    delta = PS  * (pix - 8.5) + sign * QS * (pix - QOFF)**2
    alpha = phi + gamma + delta
    beta = phi - gamma
    model = 1000. * g * (np.sin(alpha) + np.sin(beta))
    
    if error is None:
        return model - data
    else:
        return (model - data) / error
    
def gratingModelB1(p, gratpos, pixel, module, data, error=None):
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
    ai,bi,ci = p['ai'],p['bi'],p['ci']
    g0, NP, a = p['g0'], p['NP'], p['a']
    PS, QOFF, QS = p['PS'], p['QOFF'], p['QS']
    pix = pixel + 1
    slitPos = 25 - 6 * (module // 5) + module % 5
    gamma=0.0089008
    blue1 = np.array([-263.92944121,  -53.59084654,    1.16697799,   51.19513828,  422.65026353,
             -189.63033763,  -33.17725668,  -19.96267952,   26.01302266,  307.31828786,
             -156.31979898,  -37.76920495,   14.25657713,    9.02851029,  216.42404114,
              -75.57154681,   28.56399698,   33.54483603,   24.91445915,  215.17805003,
             -108.48468372,  -12.59286879,    6.90170244,  -10.74710888,  175.93175233])
    
    g = g0 * (1 - 0.5 * ((slitPos - NP)/a)**2)
    ISOFF = ai * slitPos**2 + bi * slitPos + ci - blue1[module]
    # Model (Add one to pixel, since they are counted 1,16)
    phi = 2. * np.pi * (gratpos + ISOFF) / 2.0 ** 24
    sign = np.sign(pix - QOFF)
    delta = PS  * (pix - 8.5) + sign * QS * (pix - QOFF)**2
    alpha = phi + gamma + delta
    beta = phi - gamma
    model = 1000. * g * (np.sin(alpha) + np.sin(beta))
    
    if error is None:
        return model - data
    else:
        return (model - data) / error
    
def gratingModelB2(p, gratpos, pixel, module, data, error=None):
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
    ai,bi,ci = p['ai'],p['bi'],p['ci']
    g0, NP, a = p['g0'], p['NP'], p['a']
    PS, QOFF, QS = p['PS'], p['QOFF'], p['QS']
    pix = pixel + 1
    slitPos = 25 - 6 * (module // 5) + module % 5
    gamma=0.0089008
    blue2 = np.array([-1.80111492e+02, -4.09611668e+01,  1.78797557e-02,  5.33911505e+01,
              4.51898768e+02, -1.28648267e+02, -3.41402874e+01, -2.58367960e+01,
              1.51806221e+01,  3.40600043e+02, -1.00297089e+02, -2.52445624e+01,
              4.35994998e+00,  3.34233424e+00,  2.48134145e+02, -3.43214702e+01,
              2.64531668e+01,  2.99021981e+01,  4.11197888e+01,  2.59380351e+02,
             -6.88399816e+01, -1.68668733e-01,  1.23190431e+01,  3.38400050e+00,
              2.28956503e+02])
    
    g = g0 * (1 - 0.5 * ((slitPos - NP)/a)**2)
    ISOFF = ai * slitPos**2 + bi * slitPos + ci - blue2[module]
    # Model (Add one to pixel, since they are counted 1,16)
    phi = 2. * np.pi * (gratpos + ISOFF) / 2.0 ** 24
    sign = np.sign(pix - QOFF)
    delta = PS  * (pix - 8.5) + sign * QS * (pix - QOFF)**2
    alpha = phi + gamma + delta
    beta = phi - gamma
    model = 1000. * g/2. * (np.sin(alpha) + np.sin(beta))
    
    if error is None:
        return model - data
    else:
        return (model - data) / error
    
def computeWavCalTot(pixel, module, wavepos, gratpos, channel, order, dichroic):
    from lmfit import Parameters, Minimizer, minimize
    import numpy as np
    import pandas as pd
    import os
    from fifipy.spectra import getResolution
    
    path0, file0 = os.path.split(__file__)

    if channel == 'R':
        deadpixelfile = os.path.join(path0, 'data', 'deadRED.csv')
        channelorder = 'R'
        g0 =  0.11716182038086806
        NP =  14.20296497802675
        a  =  422.5360507587561
        ai =  6.416939412313303
        bi =  -162.00403731250026
        ci =  1150612.1286321995
        PS = 0.0006
        QOFF = 8.5
        QS = 1.6e-06
    else:
        deadpixelfile = os.path.join(path0, 'data', 'deadBLUE.csv')
        PS = 0.0006
        QOFF = 8.5
        QS = 9.5e-06
        g0 =  0.1171490870070836
        NP =  14.355714704737624
        a  =  426.73530005633415
        ai =  6.228562535851191
        bi =  -157.68826192900738
        ci =  1150951.9213831278
        if order == 1:
            channelorder = 'B1'
        else:
            channelorder = 'B2'

    fit_params = Parameters()
    fit_params.add('g0', value=g0)
    fit_params.add('NP', value=NP)
    fit_params.add('a', value=a)
    fit_params.add('ai', value=ai)
    fit_params.add('bi', value=bi)
    fit_params.add('ci', value=ci)
    fit_params.add('PS', value=PS)
    fit_params.add('QOFF', value=QOFF)
    fit_params.add('QS', value=QS)
    
    

    dead = pd.read_csv(deadpixelfile, delimiter=',',header=0,
                    names=['spaxel','spexel'])      
    spaxels = dead.spaxel.values
    spexels = dead.spexel.values
    idx = module < 0
    for i,j in zip(spaxels, spexels):
        idx |= (module == i) & (pixel == j) 
    idx = ~idx
    
    
    R = getResolution(channelorder, wavepos[idx])
    kws = {'data': wavepos[idx], 'error': wavepos[idx]/R}
    if (channel == 'R') & (dichroic == 105):
        out = minimize(gratingModelR105, fit_params, 
                          args=(gratpos[idx], pixel[idx], module[idx]),
                          kws=kws, method='leastsq')
    elif (channel == 'R') & (dichroic == 130):
        out = minimize(gratingModelR130, fit_params, 
                          args=(gratpos[idx], pixel[idx], module[idx]),
                          kws=kws, method='leastsq')
    elif (channel == 'B') & (order == 1):
        out = minimize(gratingModelB1, fit_params, 
                          args=(gratpos[idx], pixel[idx], module[idx]),
                          kws=kws, method='leastsq')
    elif (channel == 'B') & (order == 2):
        out = minimize(gratingModelB2, fit_params, 
                          args=(gratpos[idx], pixel[idx], module[idx]),
                          kws=kws, method='leastsq')
    else:
        print('This case is not contemplated')
    outpar = out.params
    g0 = outpar['g0'].value
    NP = outpar['NP'].value
    a = outpar['a'].value
    ai = outpar['ai'].value
    bi = outpar['bi'].value
    ci = outpar['ci'].value
    PS = outpar['PS'].value
    QOFF = outpar['QOFF'].value
    QS = outpar['QS'].value
    
    return g0,NP,a,ai,bi,ci,PS,QOFF,QS


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

def selectFiles(rootdir, channel, order, dichroic):
    """
    select files to compute the wavelength calibration factors

    Parameters
    ----------
    rootdir : TYPE
        DESCRIPTION.
    channel : TYPE
        DESCRIPTION.
    order : TYPE
        DESCRIPTION.
    dichroic : TYPE
        DESCRIPTION.

    Returns
    -------
    modules : TYPE
        DESCRIPTION.
    pixel : TYPE
        DESCRIPTION.
    wavepos : TYPE
        DESCRIPTION.
    gerrpos : TYPE
        DESCRIPTION.
    gratpos : TYPE
        DESCRIPTION.
    gratamp : TYPE
        DESCRIPTION.
    waveok : TYPE
        DESCRIPTION.
    nfile : TYPE
        DESCRIPTION.

    """
    from glob import glob as gb
    import pandas as pd
    import numpy as np
    import os

    if channel == 'R':
        infiles = gb(os.path.join(rootdir, channel+'1_'+dichroic+'_*.csv'))
    else:
        infiles = gb(os.path.join(rootdir, channel+str(order)+'_*_*.csv'))

    module = []
    pixel = []
    wavepos = []
    gratpos = []
    gerrpos = []
    gratfwhm = []
    gratamp = []
    waveok = []
    fractions = []
    nfile = []
    print('Number of files ', len(infiles))
    for nf, infile in enumerate(infiles):  
        lines = pd.read_csv(infile, delimiter=',',header=0,
                    names=['module','pixel','wavepos','gratpos', 'gerrpos', 'gratfwhm','gratamp','fractions','ok'])
        module.extend(lines.module.values)
        pixel.extend(lines.pixel.values)
        wavepos.extend(lines.wavepos.values)
        gratpos.extend(lines.gratpos.values)
        gerrpos.extend(lines.gerrpos.values)
        gratfwhm.extend(lines.gratfwhm.values)
        gratamp.extend(lines.gratamp.values)
        waveok.extend(lines.ok.values)
        fractions.extend(lines.fractions.values)
        nfile.extend([nf]*len(lines.module.values))
    
    modules = np.array(module)
    pixel = np.array(pixel)
    wavepos = np.array(wavepos)
    gerrpos = np.array(gerrpos)
    gratpos = np.array(gratpos)
    gratfwhm = np.array(gratfwhm)
    gratamp = np.array(gratamp)
    fractions = np.array(fractions)
    waveok = np.array(waveok)
    nfile = np.array(nfile)
    
    return modules, pixel, wavepos, gerrpos, gratpos, gratamp, waveok, nfile

def fitISOFF(ISOFF, channel, dichroic, order):
    """
    Fit the ISOFF values of each spatial module 
    to find parameters for the parabola.

    Parameters
    ----------
    ISOFF : TYPE
        DESCRIPTION.
    channel : TYPE
        DESCRIPTION.

    Returns
    -------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.

    """
    from lmfit.models import QuadraticModel
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    
    red = [  -8.20328893,    6.35303121,   15.15850744,   47.13169968,   53.19085482,
           -87.37379178,  -58.5477522,   -56.42734186,  -30.89791712,  -77.46628125,
           38.68317738,   53.59124765,   61.66651196,   82.1684345,   -38.09540496,
           -6.24125859,   35.99663217 ,  43.09176554,   72.24014254 ,-131.23373814,
           -53.15045713,  -28.70681729 ,  -4.15794671,   20.91594107, -251.65783885]

    blue = [-258.04881172,  -53.44382751 ,   0.9087238 ,   50.50108697 , 425.55698219,
            -185.70929074,  -33.69497542 , -18.19496335,   26.4389202,   316.63904619,
            -143.17299604,  -39.18047949 ,  15.21162265 ,   9.6569573 ,  213.22748723,
            -86.86089338 ,  26.81683061 ,  32.44337077 ,  24.67524844 , 216.62538125,
            -106.91685921 , -16.26743503 ,   6.44892054,  -13.76693703 , 172.77801033]
    
    
    red105 = [ -7.82734595,    6.48103643,   15.37344189,   47.39558183,   54.25017651,
              -87.78073561,  -57.87672198,  -57.02387395,  -30.75647953,  -82.13171852,
               38.17407445,   53.9293801 ,   62.15816713,   82.60265586,  -51.04419029,
               -6.0626937,   36.28682384,   42.49162215,   70.33355788, -148.78530207,
              -52.04256692 , -29.12922045,   -4.73520485,   20.72545992, -268.51481606]
    
    red130 = [ -12.70859072,    7.50024661,   18.53167461,   41.46400465,   52.7757175,
              -95.78015715,  -56.53938436,  -54.24399594,  -33.75992799,  -68.99733959,
               31.27967525,   53.60554151,   58.10103624,   71.69960587,  -22.11761283,
               -4.64846212 ,  38.77585613,   42.34325365,   60.40053434, -118.02749666,
              -47.8753654 ,  -24.45939546,   -4.54977914,    8.74871326, -223.38722927]
    
    blue1 = [-263.92944121,  -53.59084654,    1.16697799,   51.19513828,  422.65026353,
             -189.63033763,  -33.17725668,  -19.96267952,   26.01302266,  307.31828786,
             -156.31979898,  -37.76920495,   14.25657713,    9.02851029,  216.42404114,
              -75.57154681,   28.56399698,   33.54483603,   24.91445915,  215.17805003,
             -108.48468372,  -12.59286879,    6.90170244,  -10.74710888,  175.93175233]
    
    blue2 = [-1.80111492e+02, -4.09611668e+01,  1.78797557e-02,  5.33911505e+01,
              4.51898768e+02, -1.28648267e+02, -3.41402874e+01, -2.58367960e+01,
              1.51806221e+01,  3.40600043e+02, -1.00297089e+02, -2.52445624e+01,
              4.35994998e+00,  3.34233424e+00,  2.48134145e+02, -3.43214702e+01,
              2.64531668e+01,  2.99021981e+01,  4.11197888e+01,  2.59380351e+02,
             -6.88399816e+01, -1.68668733e-01,  1.23190431e+01,  3.38400050e+00,
              2.28956503e+02]
    
    module = np.arange(25)
    slitPos = 25 - 6 * (module // 5) + module % 5
    x = slitPos.copy()
    y = ISOFF.copy()

    if channel == 'R':
        if dichroic == '105':
            y += red105
        else:
            y += red130
        pass
    else:
        if order == 1:
            y += blue1
            pass
        else:
            y += blue2
            pass

    plt.plot(x, y,'o')
    plt.plot(x, ISOFF, '.')
    mod = QuadraticModel()
    idx = np.isfinite(y)
    pars = mod.guess(y[idx], x=x[idx])
    out = mod.fit(y[idx], pars, x=x[idx])
    a,b,c = out.params['a'].value,out.params['b'].value,out.params['c'].value
    x_=np.arange(0, 30, 0.5)
    plt.plot(x_, a*x_**2+b*x_+c, color='blue')
    plt.title('ISOFF')

    # Reject outliers 
    for k in range(3):
        res = y - (a*x**2+b*x+c)
        med = np.nanmedian(res)
        mad = np.nanmedian(np.abs(res - med))
        id3 = np.abs(res-med) < 3*mad
        pars = mod.guess(y[id3], x=x[id3])
        out = mod.fit(y[id3], pars, x=x[id3])
        a, b, c = out.params['a'].value,out.params['b'].value,out.params['c'].value
        x = x[id3]
        y = y[id3]
        plt.plot(x,y,'x',color='red')
        plt.plot(x_, a*x_**2+b*x_+c, color='red')
        #print(out.params)
    plt.title('ISOFF')
    plt.show()

    print('ai = ',a)
    print('bi = ',b)
    print('ci = ',c)

    isoff = a * slitPos**2 + b * slitPos + c
    if channel == 'R':
        if dichroic == '105':
            isoff -= red105
        else:
            isoff -= red130
        pass
    else:
        if order == 1:
            isoff -= blue1
            pass
        else:
            isoff -= blue2
            pass

    print("ISOFF = [", end='')
    for i, isof in enumerate(isoff):
        if i < 24:
            if i % 4 == 0:
                print('{0:.3f},'.format(isof))
            else:
                print('{0:.3f},'.format(isof),end='')
        else:
            print('{0:.3f}]'.format(isof))

    return a, b, c


def fitg(g):
    """
    Fit values of g computed for every spaxel to a parabola.

    Parameters
    ----------
    g : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    from lmfit.models import QuadraticModel
    import numpy as np
    import matplotlib.pyplot as plt

    module = np.arange(25)
    slitPos = 25 - 6 * (module // 5) + module % 5

    plt.plot(slitPos, g,'o')
    x = slitPos.copy()
    y = g.copy()
    mod = QuadraticModel()
    idx = np.isfinite(y)
    pars = mod.guess(y[idx], x=x[idx])
    out = mod.fit(y[idx], pars, x=x[idx])
    a,b,c = out.params['a'].value,out.params['b'].value,out.params['c'].value
    x_=np.arange(0,30,0.5)
    plt.plot(x_, a*x_**2+b*x_+c,color='blue')
    plt.title('g')

    # Reject outliers 
    for k in range(10):
        res = y - (a*x**2+b*x+c)
        med = np.nanmedian(res)
        mad = np.nanmedian(np.abs(res - med))
        id3 = np.abs(res-med) < 3*mad
        pars = mod.guess(y[id3], x=x[id3])
        out = mod.fit(y[id3], pars, x=x[id3])
        a,b,c = out.params['a'].value,out.params['b'].value,out.params['c'].value
        x = x[id3]
        y = y[id3]
        plt.plot(x,y,'x',color='red')
        plt.plot(x_, a*x_**2+b*x_+c, color='red')
        print('g0 ', c - b**2/(4*a))
    #print(out.params)
    plt.title('g')
    plt.show()

    a = out.params['a'].value
    b = out.params['b'].value
    c = out.params['c'].value
    
    print('g = [')
    for ii, sp in enumerate(slitPos):
        if ii < 24:
            if ii % 4 == 0:
                print('{0:.8f},'.format(a*sp**2+b*sp+c))
            else:
                print('{0:.8f},'.format(a*sp**2+b*sp+c), end='')
        else:
            print('{0:.8f}]'.format(a*sp**2+b*sp+c))
    
#g = a*slitPos**2 + b*slitPos + c
#print('central value ', -b/(2*a))
#print('vertex ', c - b * b/ (4 *a))
#print('Amplitude ', a)
    # Sebastian's expression
    g0 = c - b**2/(4*a)
    NP = - 0.5 * b / a
    a_ = np.sqrt(-0.5 * g0/ a)
    print('g0 = ', g0)
    print('NP = ', NP)
    print('a  = ', a_)

    return g0, NP, a_

def plotLines(rootdir, channel, order,i=8,j=12,files=None):
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from glob import glob as gb
    import os
    import re 
    from matplotlib.ticker import ScalarFormatter
    import numpy as np
    order = str(order)
    if files is None:
        filenames = channel+order+'*.fits'
        files = sorted(gb(os.path.join(rootdir, 'Reduced', filenames)))
    # Split in two plots if R
    if channel == 'B':
        fig,ax = plt.subplots(figsize=(18,6))
        wmin, wmax = 200, 0
        for file in files:
            with fits.open(file) as hdl:
                gc = re.findall(r'GC(\d+_\d)', file)[0]
                g = hdl['Grating Position'].data
                specs = hdl['SPECS'].data
                w = hdl['WAVE'].data
                header = hdl[0].header
                if np.nanmin(w) < wmin:
                    wmin = np.nanmin(w)
                if np.nanmax(w) > wmax:
                    wmax = np.nanmax(w)
                dichroic = header['DICHROIC']
                if dichroic == 130:
                    ax.plot(g, specs[:,i,j], linestyle='-', label=gc)
                else:
                    ax.plot(g, specs[:,i,j], linestyle='--', label=gc)
        #ax.set_ylim(0,12)
        ax.set_ylim(ymin=0)
        wax = ax.twiny()
        wax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        wax.set_xlim(wmin, wmax)
        wax.set_xlabel("Wavelength [$\mu$m]")
        ax.set_xlabel("Grating position [ISU]")
        ax.grid()
        ax.legend()
        ax.set_title(channel+' '+order)
    else:
        fig, ax = plt.subplots(nrows=4,figsize=(18,24))
        wmin1, wmax1 = 200, 0
        wmin2, wmax2 = 200, 0
        wmin3, wmax3 = 200, 0
        wmin4, wmax4 = 200, 0
        ax = np.array(ax)
        for file in files:
            with fits.open(file) as hdl:
                gc = re.findall(r'GC(\d+_\d)', file)[0]
                g = hdl['Grating Position'].data
                specs = hdl['SPECS'].data
                w = hdl['WAVE'].data
                header = hdl[0].header
                dichroic = header['DICHROIC']
                wmin, wmax = np.nanmin(w), np.nanmax(w)
                if wmax < 130:
                    idplot = 0
                elif (wmin > 125) & (wmax < 145):
                    idplot = 1
                elif (wmin > 140) & (wmax < 155):
                    idplot = 2
                else:
                    idplot = 3   
                if idplot == 0:
                    if wmin < wmin1:
                        wmin1 = wmin
                    if wmax > wmax1:
                        wmax1 = wmax
                elif idplot == 1:
                    if wmin < wmin2:
                        wmin2 = wmin
                    if wmax > wmax2:
                        wmax2 = wmax
                elif idplot == 2:
                    if wmin < wmin3:
                        wmin3 = wmin
                    if wmax > wmax3:
                        wmax3 = wmax
                else:
                    if wmin < wmin4:
                        wmin4 = wmin
                    if wmax > wmax4:
                        wmax4 = wmax
                if dichroic == 130:
                    ax[idplot].plot(g, specs[:,i,j], linestyle='-', label=gc)
                else:
                    ax[idplot].plot(g, specs[:,i,j], linestyle='--', label=gc)

        wax1 = ax[0].twiny()
        wax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        wax1.set_xlim(wmin1, wmax1)
        wax1.set_xlabel("Wavelength [$\mu$m]")
        wax2 = ax[1].twiny()
        wax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        wax2.set_xlim(wmin2, wmax2)
        wax2.set_xlabel("Wavelength [$\mu$m]")
        wax3 = ax[2].twiny()
        wax3.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        wax3.set_xlim(wmin3, wmax3)
        wax3.set_xlabel("Wavelength [$\mu$m]")
        wax4 = ax[3].twiny()
        wax4.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        wax4.set_xlim(wmin4, wmax4)
        wax4.set_xlabel("Wavelength [$\mu$m]")
        for i in [0,1,2,3]:
            ax[i].set_ylim(ymin=0)
            ax[i].xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax[i].set_xlabel("Grating position [ISU]")
            ax[i].grid()
            ax[i].legend()
            ax[i].set_title(channel+' '+order)
    plt.show()    


def plotQualityFit(rootdir, channelorder, dichroic, g0, NP, a, ai, bi, ci, PS, QS, QOFF, ISOFF=None, comparison=None):
    import matplotlib.pyplot as plt
    from fifipy.wavecal import computeWavelength
    from fifipy.spectra import getResolution
    import numpy as np
    from fifipy.wavecal import selectFiles
        
    if channelorder == 'R':
        channel = 'R'
        order = 1
    elif channelorder == 'B1':
        channel = 'B'
        order = 1
    elif channelorder == 'B2':
        channel = 'B'
        order = 2
    else:
        print(channelorder ,' is an invalid channelorder')
        return
    
    modules, pixel, wavepos, gerrpos, gratpos, gratamp, waveok, nfile = selectFiles(rootdir,channel,order,dichroic)
    module = np.arange(25)
    slitPos = 25 - 6 * (module // 5) + module % 5
    red105 = [ -7.82734595,    6.48103643,   15.37344189,   47.39558183,   54.25017651,
              -87.78073561,  -57.87672198,  -57.02387395,  -30.75647953,  -82.13171852,
               38.17407445,   53.9293801 ,   62.15816713,   82.60265586,  -51.04419029,
               -6.0626937,   36.28682384,   42.49162215,   70.33355788, -148.78530207,
              -52.04256692 , -29.12922045,   -4.73520485,   20.72545992, -268.51481606]
    
    red130 = [ -12.70859072,    7.50024661,   18.53167461,   41.46400465,   52.7757175,
              -95.78015715,  -56.53938436,  -54.24399594,  -33.75992799,  -68.99733959,
               31.27967525,   53.60554151,   58.10103624,   71.69960587,  -22.11761283,
               -4.64846212 ,  38.77585613,   42.34325365,   60.40053434, -118.02749666,
              -47.8753654 ,  -24.45939546,   -4.54977914,    8.74871326, -223.38722927]
    
    blue1 = [-263.92944121,  -53.59084654,    1.16697799,   51.19513828,  422.65026353,
             -189.63033763,  -33.17725668,  -19.96267952,   26.01302266,  307.31828786,
             -156.31979898,  -37.76920495,   14.25657713,    9.02851029,  216.42404114,
              -75.57154681,   28.56399698,   33.54483603,   24.91445915,  215.17805003,
             -108.48468372,  -12.59286879,    6.90170244,  -10.74710888,  175.93175233]
    
    blue2 = [-1.80111492e+02, -4.09611668e+01,  1.78797557e-02,  5.33911505e+01,
              4.51898768e+02, -1.28648267e+02, -3.41402874e+01, -2.58367960e+01,
              1.51806221e+01,  3.40600043e+02, -1.00297089e+02, -2.52445624e+01,
              4.35994998e+00,  3.34233424e+00,  2.48134145e+02, -3.43214702e+01,
              2.64531668e+01,  2.99021981e+01,  4.11197888e+01,  2.59380351e+02,
             -6.88399816e+01, -1.68668733e-01,  1.23190431e+01,  3.38400050e+00,
              2.28956503e+02]
    if ISOFF is None:
        if channelorder == 'B1':
                ISOFF = ai*slitPos**2 + bi*slitPos + ci - blue1
        elif channelorder == 'B2':
                ISOFF = ai*slitPos**2 + bi*slitPos + ci - blue2
        elif channelorder == 'R':
            if dichroic == '105':
                ISOFF = ai*slitPos**2 + bi*slitPos + ci - red105
            else:
                ISOFF = ai*slitPos**2 + bi*slitPos + ci - red130
           
    if channelorder == 'R':
        gamma = 0.0167200
    else:
        gamma = 0.0089008

    ISF=1
    coeffs = [g0, NP, a, ISF, gamma, PS, QOFF, QS, ISOFF]
    
    w_est = []
    resol = []
    modul = []
    idx = (gratamp > 100) & (gerrpos < 100) & (waveok ==1)
    for pix, mo, gp, wp in zip(pixel[idx], modules[idx], gratpos[idx], wavepos[idx]):
        w,dw = computeWavelength(pix, mo, order, coeffs, gp)
        resol.append(getResolution(channelorder, wp))
        w_est.append(w)
        modul.append(mo)
    w_est = np.array(w_est)
    R = np.array(resol)
    modul = np.array(modul)
    
    fig,ax = plt.subplots(figsize=(14,6))
    waveposx = wavepos[idx]
    for i in range(25):
        mi = modul == i
        plt.plot(waveposx[mi],(1 - w_est[mi]/waveposx[mi])*R[mi],'.',color='skyblue')

    if comparison is not None:
        w_comp = []
        idx = (gratamp > 100) & (gerrpos < 100) & (waveok ==1)
        resol = []
        for pix, mo, gp, wp in zip(pixel[idx], modules[idx], gratpos[idx], wavepos[idx]):
            w,dw = computeWavelength(pix, mo, order, comparison, gp)
            resol.append(getResolution(channelorder, wp))
            w_comp.append(w)
        w_comp = np.array(w_comp)
        R = np.array(resol)
        if channelorder == 'B2':
            xshift = 0.15
        else:
            xshift = 0.35
        plt.plot(wavepos[idx]+xshift, (1-w_comp/wavepos[idx])*R,'.',label='comp',color='orange')
        plt.legend()

        

    plt.ylabel ( '$(1 - \lambda_{est}/\lambda )R$')
    plt.ylim(-0.3,0.3)
    plt.grid()
    plt.show()
    if comparison is not None:
        fig,ax = plt.subplots(figsize=(14,6))
        plt.plot(wavepos[idx], (w_est-w_comp)/wavepos[idx]*R,'.',color='skyblue')
        #plt.plot(w_comp, (w_est-w_comp)/w_comp*R,'.',color='skyblue')
        plt.ylabel('$(\lambda_{est}-\lambda_{comp})R/\lambda$')
        plt.grid()
        plt.show()

def plotComparisonFit(rootdir, channelorder, dichroic, g0, NP, a, ai, bi, ci, PS, QS, QOFF, ISOFF=None, comparison=None):
    import matplotlib.pyplot as plt
    from fifipy.wavecal import computeWavelength
    from fifipy.spectra import getResolution
    import numpy as np
    from fifipy.wavecal import selectFiles
        
    if channelorder == 'R':
        channel = 'R'
        order = 1
    elif channelorder == 'B1':
        channel = 'B'
        order = 1
    elif channelorder == 'B2':
        channel = 'B'
        order = 2
    else:
        print(channelorder ,' is an invalid channelorder')
        return
    
    modules, pixel, wavepos, gerrpos, gratpos, gratamp, waveok, nfile = selectFiles(rootdir,channel,order,dichroic)
    module = np.arange(25)
    slitPos = 25 - 6 * (module // 5) + module % 5
    red105 = [ -7.82734595,    6.48103643,   15.37344189,   47.39558183,   54.25017651,
              -87.78073561,  -57.87672198,  -57.02387395,  -30.75647953,  -82.13171852,
               38.17407445,   53.9293801 ,   62.15816713,   82.60265586,  -51.04419029,
               -6.0626937,   36.28682384,   42.49162215,   70.33355788, -148.78530207,
              -52.04256692 , -29.12922045,   -4.73520485,   20.72545992, -268.51481606]
    
    red130 = [ -12.70859072,    7.50024661,   18.53167461,   41.46400465,   52.7757175,
              -95.78015715,  -56.53938436,  -54.24399594,  -33.75992799,  -68.99733959,
               31.27967525,   53.60554151,   58.10103624,   71.69960587,  -22.11761283,
               -4.64846212 ,  38.77585613,   42.34325365,   60.40053434, -118.02749666,
              -47.8753654 ,  -24.45939546,   -4.54977914,    8.74871326, -223.38722927]
    
    blue1 = [-263.92944121,  -53.59084654,    1.16697799,   51.19513828,  422.65026353,
             -189.63033763,  -33.17725668,  -19.96267952,   26.01302266,  307.31828786,
             -156.31979898,  -37.76920495,   14.25657713,    9.02851029,  216.42404114,
              -75.57154681,   28.56399698,   33.54483603,   24.91445915,  215.17805003,
             -108.48468372,  -12.59286879,    6.90170244,  -10.74710888,  175.93175233]
    
    blue2 = [-1.80111492e+02, -4.09611668e+01,  1.78797557e-02,  5.33911505e+01,
              4.51898768e+02, -1.28648267e+02, -3.41402874e+01, -2.58367960e+01,
              1.51806221e+01,  3.40600043e+02, -1.00297089e+02, -2.52445624e+01,
              4.35994998e+00,  3.34233424e+00,  2.48134145e+02, -3.43214702e+01,
              2.64531668e+01,  2.99021981e+01,  4.11197888e+01,  2.59380351e+02,
             -6.88399816e+01, -1.68668733e-01,  1.23190431e+01,  3.38400050e+00,
              2.28956503e+02]
    if ISOFF is None:
        if channelorder == 'B1':
                ISOFF = ai*slitPos**2 + bi*slitPos + ci - blue1
        elif channelorder == 'B2':
                ISOFF = ai*slitPos**2 + bi*slitPos + ci - blue2
        elif channelorder == 'R':
            if dichroic == '105':
                ISOFF = ai*slitPos**2 + bi*slitPos + ci - red105
            else:
                ISOFF = ai*slitPos**2 + bi*slitPos + ci - red130
           
    if channelorder == 'R':
        gamma = 0.0167200
    else:
        gamma = 0.0089008

    ISF=1
    coeffs = [g0, NP, a, ISF, gamma, PS, QOFF, QS, ISOFF]
        
    fig,ax = plt.subplots(figsize=(14,6))
    gratpos = np.arange(90000,2000000,2000)
    for mo in range(25):
        for pix in range(16):
            w,dw = computeWavelength(pix, mo, order, coeffs, gratpos)
            wc,dwc = computeWavelength(pix, mo, order, comparison, gratpos)
            resol = getResolution(channelorder, w)
            plt.plot(w, (1-wc/w)*resol)
    
    plt.ylabel('$(1 - \lambda_{comp}/\lambda) R$')
    #plt.ylim(-0.01,0.01)
    plt.grid()
    plt.show()

