def reduceKVdata(rootdir, names=None):
    """
    Routine to reduce files created for the K-mirror vectors calibration

    Parameters
    ----------
    rootdir : TYPE string
        DESCRIPTION. directory containing all calibration data
    names : TYPE, optional
        DESCRIPTION. The default is 'None'. List of numbers of data (ex: ['01','02'])

    Returns
    -------
    None.

    """

    from fifipy.fit import computeSpectra
    from fifipy.calib import computeAllWaves
    from fifipy.io import saveSlopeFits, readData
    from glob import glob as gb
    from astropy.io import fits
    import os
    import numpy as np
    
    # Create new directory to save reduced data
    reducedir = os.path.join(rootdir,'Reduced')
    if not os.path.exists(reducedir):
        os.makedirs(reducedir)

    if names is None:
        names = ['{0:02d}'.format(i) for i in range(1,50)]
        
    for name in names:
        for channel in ['sw','lw']:
            filenames = '*KV'+name+'*'+channel+'.fits'
            matchfile = os.path.join(rootdir, '**', filenames)
            files = sorted(gb(matchfile, recursive=True))
            if len(files) > 0: 
                print('\n',matchfile,': ', len(files))
                gpos, specs = computeSpectra(files, telSim=True)
                aor, hk, gratpos, flux = readData(files[0])
                obsdate,telpos,pos,xy,an,za,alti,wv = hk
                detchan, order, dichroic, ncycles, nodbeam, filegpid, filenum = aor
                wave, dwave = computeAllWaves(gpos, dichroic, obsdate, detchan, order)
                # Compute coordinates from name
                #xcoords = []
                #ycoords = []
                #match1 = r"(-?\d,-?\d)"
                #match2 = r"(\d)x"
                #for f in files:
                #    x1 = re.search(match1, f)
                #    coords = x1.group(1)
                #    xcoord, ycoord = coords.split(',')
                #    x2 = re.search(match2, f)
                #    factor = int(x2.group(1))
                #    xcoords.append(int(xcoord)*factor)
                #     ycoords.append(int(ycoord)*factor)
                #xcoords = np.array(xcoords)
                #ycoords = np.array(ycoords)
                # Coordinates of source are recorded in dmm (0.1mm) in OBSBET and OBSLAM
                xcoords = []
                ycoords = []
                for f in files:
                    header = fits.getheader(f)
                    xcoords.append(header['OBSBET'] * 0.1)
                    ycoords.append(header['OBSLAM'] * 0.1)
                xcoords = np.array(xcoords)
                ycoords = np.array(ycoords)
                # Get basic info from header
                # header = fits.getheader(files[0])
                kmirrpos = header['KMIRRPOS']
                channel = header['DETCHAN']
                dichroic = header['DICHROIC']
                if channel == 'BLUE':
                    isu = header['G_STRT_B']
                    ch = 'B'
                    order = header['G_ORD_B']
                else:
                    isu = header['G_STRT_R']
                    ch = 'R'
                    order = 1
                # Output file
                outname = os.path.join(reducedir, ch+str(order)+'_D'+str(dichroic)+'_K'+str(kmirrpos)+'_G'+str(isu)+'.fits')
                saveSlopeFits(gpos, dichroic, obsdate, detchan, order, specs, wave, dwave, outname,
                              xcoords=xcoords, ycoords=ycoords, kmirr=kmirrpos, gratpos=isu)
                

def readKVfile(infile):
    from astropy.io import fits
    import numpy as np
    
    with fits.open(infile) as hdul:
        header = hdul[0].header
        specs = hdul['SPECS'].data
        wave = hdul['WAVE'].data
        xcoords = hdul['XCOORDS'].data
        ycoords = hdul['YCOORDS'].data
        
    return xcoords, ycoords, specs, np.nanmedian(wave), header['ORDER']

def saveKVfile(rootdir, channel, dichroic, kmirr, gpos, waves, orders,
               fluxes, xcoords, ycoords):
    """
    Save collapsed cube for each K-mirror and grating position for one channel/order

    Parameters
    ----------
    rootdir : TYPE
        DESCRIPTION.
    channel : TYPE
        DESCRIPTION.
    dichroic : TYPE
        DESCRIPTION.
    kmirr : TYPE
        DESCRIPTION.
    gpos : TYPE
        DESCRIPTION.
    waves : TYPE
        DESCRIPTION.
    orders : TYPE
        DESCRIPTION.
    fluxes : TYPE
        DESCRIPTION.
    xcoords : TYPE
        DESCRIPTION.
    ycoords : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import os
    from astropy.io import fits
    filename = channel+'_'+dichroic+'.fits'
    outname = os.path.join(rootdir, filename)
    hdu = fits.PrimaryHDU()
    hdu.header['CHANNEL'] = channel
    hdu.header['DICHROIC'] = dichroic
    hdu1 = fits.ImageHDU()
    hdu1.data = kmirr
    hdu1.header['EXTNAME'] = 'Kmirror Position'
    hdu2 = fits.ImageHDU()
    hdu2.data = gpos
    hdu2.header['EXTNAME'] = 'Grating Position'
    hdu3 = fits.ImageHDU()
    hdu3.data = fluxes
    hdu3.header['EXTNAME'] = 'FLUXES'
    hdu4 = fits.ImageHDU()
    hdu4.data = xcoords
    hdu4.header['EXTNAME'] = 'XCOORDS'
    hdu5 = fits.ImageHDU()
    hdu5.data = ycoords
    hdu5.header['EXTNAME'] = 'YCOORDS'
    hdu6 = fits.ImageHDU()
    hdu6.data = waves
    hdu6.header['EXTNAME'] = 'WAVES'
    hdu7 = fits.ImageHDU()
    hdu7.data = orders
    hdu7.header['EXTNAME'] = 'ORDERS'
    
    hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7])
    hdul.writeto(outname, overwrite=True)
    hdul.close()
    


def collapseCubes(rootdir):
    """
    Collapse along spectral direction of cubes for each observation.
    The cubes are conserved in one structure for each channel/dichroic combination.

    Parameters
    ----------
    rootdir : TYPE string
        DESCRIPTION. Directory where the files reside.

    Returns
    -------
    None.

    """
    from glob import glob as gb
    import os
    import numpy as np
    from fifipy.stats import biweightLocation
    from fifipy.kvector import readKVfile, saveKVfile

    # Find files
    channels = ['B', 'B', 'R', 'R']
    dichroics = ['105', '130', '105', '130']
    for channel, dichroic in zip(channels, dichroics):
        filenames = channel+'*_D'+dichroic+'_K*.fits'
        matchfile = os.path.join(rootdir, '**', filenames)
        files = gb(matchfile, recursive=True)
        nfiles = len(files)
        if nfiles > 0:
            print(filenames, nfiles)
            # Order files by Kmirror and Grating position
            files = np.array(files)
            kmirr = np.array([int(file.split('_')[2][1:]) for file in files])
            gpos = np.array([int(file.split('_')[3][1:-5]) for file in files])
            s1 = np.argsort(gpos)
            files = files[s1]
            kmirr = kmirr[s1]
            gpos = gpos[s1]
            s2 = np.argsort(kmirr)
            files = files[s2]
            kmirr = kmirr[s2]
            gpos = gpos[s2]
            fluxes = np.zeros((nfiles, 25, 289))
            # Open files and collapse along spectral direction
            waves = []
            orders = []
            for flux, file in zip(fluxes, files):
                xcoords, ycoords, specs, wave, order = readKVfile(file)
                waves.append(wave)
                orders.append(order)
                med = biweightLocation(specs, axis=1)
                for k in range(25):
                    flux[k] = med[:,k]
            waves = np.array(waves)
            orders = np.array(orders)
            # Save new files
            saveKVfile(rootdir, channel, dichroic, kmirr, gpos, waves, orders, fluxes, xcoords, ycoords)
            
def plot2D(infile):
    """
    Plot 2D interpolations of the 25 pixels at one K-mirror and grating position. 

    Parameters
    ----------
    infile : TYPE
        DESCRIPTION. File with reduced data.

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    from fifipy.kvector import readKVfile
    from fifipy.stats import biweightLocation
    #infile = '/home/dario/Python/Fifilab/KVectorData/Reduced/B1_D105_K1092_G180000.fits'
    xcoords, ycoords, specs = readKVfile(infile)
    med = biweightLocation(specs, axis=1)
    fig,axes =plt.subplots(5,5,figsize=(15,15))
    for npix in range(25):
        ax = axes[npix // 5 , npix % 5]
        ax.tricontourf(xcoords, ycoords, med[:,npix], cmap='inferno',levels=14)
        ax.text(np.nanmedian(xcoords), np.nanmax(ycoords)+1,str(npix+1),color='black')
    plt.show()


def plotPixels(infile):
    """
    Plot a figure with subplot for each pixel from one observation at a particular
    K-mirror and grating positions.

    Parameters
    ----------
    infile : TYPE
        DESCRIPTION. File with reduced data

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    from fifipy.stats import biweightLocation
    from fifipy.kvector import readKVfile
    #infile = '/home/dario/Python/Fifilab/KVectorData/Reduced/R1_D105_K1092_G1003000.fits'
    xcoords, ycoords, specs = readKVfile(infile)
    
    fig,axes =plt.subplots(5,5,figsize=(15,15))
    pixel = np.zeros((25,25))
    
    #med = np.median(specs, axis=1)
    med = biweightLocation(specs, axis=1)
    
    for npix in range(25):
        ax = axes[npix // 5 , npix % 5]
        idx = (xcoords <= 6) & (xcoords >=-6) & (ycoords <= 6) & (ycoords >=-6)
        pixel[ycoords[idx]+12, xcoords[idx]+12] = med[idx, npix]
        idx = (xcoords >=6) & (ycoords >=6)
        for ix in [11,12]:
            for iy in [11,12]:
                pixel[ycoords[idx]+iy, xcoords[idx]+ix] = med[idx, npix]
        idx = (xcoords >=6) & (ycoords <6)
        for ix in [11,12]:
            for iy in [12,13]:
                pixel[ycoords[idx]+iy, xcoords[idx]+ix] = med[idx, npix]
        idx = (xcoords < 6) & (ycoords >=6)
        for ix in [12,13]:
            for iy in [11,12]:
                pixel[ycoords[idx]+iy, xcoords[idx]+ix] = med[idx, npix]
        idx = (xcoords < 6) & (ycoords <6)
        for ix in [12,13]:
            for iy in [12,13]:
                pixel[ycoords[idx]+iy, xcoords[idx]+ix] = med[idx, npix]
        ax.imshow(pixel, origin='lower', extent=[-12.5,12.5,-12.5,12.5],cmap='inferno')
        ax.text(np.nanmedian(xcoords), np.nanmax(ycoords)+1,str(npix+1),color='black')
    plt.show()
    
def getPSFsigma(channel, order, w):
    import numpy as np
    m1diam = 2.500
    tFWHM = 1.013 * w *1.e-6/m1diam * 3600. * 180/np.pi
    if channel == 'R':
        iFWHM =  3.55*(np.sqrt(0.0156*0.0116)*w + np.sqrt(1.3214*1.6466)) - 4.5
    else:
        if order == '1':
            iFWHM = 3.55 * (np.sqrt(0.0151*0.0179)*w + np.sqrt(0.8621*0.2169)) - 2.
        else:
            iFWHM = 3.55 * (np.sqrt(0.0057*0.0064)*w + np.sqrt(1.5699*1.0353)) - 1.0
    FWHM = np.sqrt(tFWHM * tFWHM + iFWHM * iFWHM)
    sigma = FWHM / 2.355
    return sigma

def residuals2Dgauss(p, x, y, data=None):
    import numpy as np
    v = p.valuesdict()
    x0 = v['x0']
    y0 = v['y0']
    sx = v['sx']
    sy = v['sy']
    A  = v['A']
    B  = v['B']
    x_ = (x-x0)/sx
    #y_ = (y-y0)/sx
    y_ = (y-y0)/sy
    model = A / (2 *np.pi * sx * sy) * np.exp(-0.5 * (x_**2 +y_**2) ) + B * np.exp(-0.5 * x_**2)
    #model = A / (2 *np.pi * sx * sx) * np.exp(-0.5 * (x_**2 +y_**2) ) + B * np.exp(-0.5 * x_**2)
    if data is None:
        return model
    else:
        return model-data

def fitSource(infile, plane, plot=True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from lmfit import Parameters, minimize
    from fifipy.kvector import getPSFsigma
    from astropy.io import fits
    import numpy as np

    with fits.open(infile) as hdul:
        xcoords = hdul['XCOORDS'].data
        ycoords = hdul['YCOORDS'].data
        fluxes = hdul['FLUXES'].data
        waves = hdul['WAVES'].data
        orders = hdul['ORDERS'].data
        header = hdul[0].header
        
    xmin = np.nanmin(xcoords)
    xmax = np.nanmax(xcoords)
    ymin = np.nanmin(ycoords)
    ymax = np.nanmax(ycoords)
    
    channel = header['CHANNEL']
    fluxplane = fluxes[plane]
    order = orders[plane]
    wave = waves[plane]
    if plot:
        fig,axes =plt.subplots(5,5,figsize=(15,15))
    xcen = []
    ycen = []
    xsig = []
    ysig = []
    for npix in range(25):
        
        flux = fluxplane[npix]
        scale = 3.55 # arcsec/mm
        #step = 3 #mm
        sigma = getPSFsigma(channel, order, wave)  # Blue   
        amplitude = np.nanmax(flux) - np.nanmedian(flux)
        sigma /= scale#/step 
        amplitude *= 2*np.pi*sigma**2
        
        idx = np.argmax(flux)
        x0guess = xcoords[idx]
        y0guess = ycoords[idx]
            
        fit_params = Parameters()
        fit_params.add('x0', value = x0guess, max = x0guess+2, min=x0guess-2)
        fit_params.add('y0', value = y0guess, max = y0guess+2, min=y0guess-2)
        fit_params.add('sx', value = sigma, min=sigma*0.5, max=sigma*2)
        fit_params.add('sy', value = sigma, min=sigma*0.5, max=sigma*2)
        fit_params.add('A' , value = amplitude)
        fit_params.add('B' , value = 0.1, min=0)
        
        out = minimize(residuals2Dgauss, fit_params, 
                       args=(xcoords, ycoords,), 
                       #kws={'data': flux_})
                       kws={'data': flux-np.nanmedian(flux)})
        outparams = out.params
        #print('B is ', outparams['B'].value)
        x0 = outparams['x0'].value
        y0 = outparams['y0'].value
        sx = outparams['sx'].value
        sy = outparams['sy'].value
        xcen.append(x0)
        ycen.append(y0)
        xsig.append(outparams['x0'].stderr)
        ysig.append(outparams['y0'].stderr)
        if plot:
            ax = axes[npix // 5 , npix % 5]
            ax.tricontourf(xcoords, ycoords, flux - np.nanmedian(flux), cmap='inferno', levels=14)
            ellipse = Ellipse((x0,y0), sx*2.355, sy*2.355, 0, fill=False)
            #ellipse = Ellipse((x0,y0), sx*2.355, sx*2.355, 0, fill=False)
            ax.add_patch(ellipse)
            
            ax.plot(x0guess,y0guess,'+',color='white')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.grid()
            ax.text(np.nanmedian(xcoords), np.nanmax(ycoords)+1,str(npix+1),color='black')
    if plot:
        plt.show()
    xcen = np.array(xcen)
    ycen = np.array(ycen)
    xsig = np.array(xsig)
    ysig = np.array(ysig)
    return xcen, ycen, xsig, ysig

def fitSources(infile):
    from fifipy.kvector import fitSource
    from astropy.io import fits
    import numpy as np
    
    with fits.open(infile) as hdul:
        fluxes = hdul['FLUXES'].data
    nplanes = len(fluxes)
    xcenters = np.zeros((nplanes, 25))
    ycenters = np.zeros((nplanes, 25))
    xcenerrs = np.zeros((nplanes, 25))
    ycenerrs = np.zeros((nplanes, 25))
    for plane in range(nplanes):
        print('Plane ', plane,' out of ', nplanes)
        xcen, ycen, xsig, ysig = fitSource(infile, plane, plot=False)
        xcenters[plane] = xcen
        ycenters[plane] = ycen
        xcenerrs[plane] = xsig
        ycenerrs[plane] = ysig
        
    return xcenters, ycenters, xcenerrs, ycenerrs

def fitSourcePosition(infile, plot=True):
    import matplotlib.pyplot as plt
    from lmfit import Parameters, minimize
    from fifipy.kvector import getPSFsigma
    from astropy.io import fits
    import numpy as np
    from fifipy.stats import biweightLocation

    with fits.open(infile) as hdul:
        header = hdul[0].header
        xcoords = hdul['XCOORDS'].data
        ycoords = hdul['YCOORDS'].data
        specs = hdul['SPECS'].data
        wave = hdul['WAVE'].data
        
    channel = header['channel']
    order = header['order']
    w = np.nanmedian(wave)  
    xcen = np.zeros((16,25))
    ycen = np.zeros((16,25))
    kmirrpos = header['KMIRRPOS']  
    kmirrangle = (kmirrpos - 52) * 0.0871

    # Evaluation of centers
    if channel == 'RED':
        x0cen = np.zeros(25)
        y0cen = np.zeros(25)
        cflux = biweightLocation(specs, axis=1)
        for npix in range(25):
            idx = np.argmax(cflux[:,npix])
            x0cen[npix] = xcoords[idx]
            y0cen[npix] = ycoords[idx]
        x0cen = np.reshape(x0cen, (5, 5))
        y0cen = np.reshape(y0cen, (5,5))
        xcenters = np.nanmedian(x0cen,axis=1)
        # Everything at same coordinate
        for i in range(5):
            x0cen[i,:] = xcenters[i]
        # Move 
        dy = 3.6 * np.sin(kmirrangle * np.pi/180)
        for i in range(5):
            y0cen[i,0] = y0cen[i,2] - 2 * dy
            y0cen[i,1] = y0cen[i,2] - dy
            y0cen[i,3] = y0cen[i,2] + dy
            y0cen[i,4] = y0cen[i,2] + 2 * dy
        x0cen = x0cen.ravel()
        y0cen = y0cen.ravel()
    
    for i in range(16):
        for npix in range(25):
            flux = specs[:,i,npix]
            scale = 3.55 # arcsec/mm
            sigma = getPSFsigma(channel, order, w)  # Blue   
            amplitude = np.nanmax(flux) - np.nanmedian(flux)
            sigma /= scale#/step 
            amplitude *= 2*np.pi*sigma**2
    
            if channel == 'BLUE':
                idx = np.argmax(flux)
                x0guess = xcoords[idx]
                y0guess = ycoords[idx]
            else:
                x0guess = x0cen[npix]
                y0guess = y0cen[npix]
                
    
            fit_params = Parameters()
            fit_params.add('x0', value = x0guess, max = x0guess+1, min=x0guess-1)
            fit_params.add('y0', value = y0guess, max = y0guess+2, min=y0guess-2)
            fit_params.add('sx', value = sigma, min=sigma*0.5, max=sigma*2)
            fit_params.add('sy', value = sigma, min=sigma*0.5, max=sigma*2)
            fit_params.add('A' , value = amplitude)
            fit_params.add('B' , value = 0.1, min=0)
            
            flux_ = flux-np.nanmedian(flux)
            idx = np.isfinite(flux_)
            if np.sum(idx) > 10:
                out = minimize(residuals2Dgauss, fit_params, 
                                args=(xcoords[idx], ycoords[idx],), 
                                kws={'data': flux_[idx]})
                outparams = out.params
                x0 = outparams['x0'].value
                y0 = outparams['y0'].value
            else:
                x0 = np.nan
                y0 = np.nan
            xcen[i,npix] = x0
            ycen[i,npix] = y0
            
    from fifipy.stats import biweightLocation
    xcmed = biweightLocation(xcen, axis=0)
    ycmed = biweightLocation(ycen , axis=0)

    if plot:
        fig,ax = plt.subplots(figsize=(10,10))
        for xc, yc in zip(xcen, ycen):
            ax.plot(xc,yc,'.')
        ax.plot(xcmed, ycmed, 'x', ms=15, color='black')
        ax.grid()
        plt.show()
        
    return xcmed, ycmed

def saveSourcePositions(rootdir, channel, dichroic, kmirr, gpos, xcenters, ycenters):
    import os
    from astropy.io import fits
    filename = channel+'_'+dichroic+'.fits'
    outname = os.path.join(rootdir, filename)
    hdu = fits.PrimaryHDU()
    hdu.header['CHANNEL'] = channel
    hdu.header['DICHROIC'] = dichroic
    hdu1 = fits.ImageHDU()
    hdu1.data = kmirr
    hdu1.header['EXTNAME'] = 'KMIRROR'
    hdu2 = fits.ImageHDU()
    hdu2.data = gpos
    hdu2.header['EXTNAME'] = 'GRATING'
    hdu3 = fits.ImageHDU()
    hdu3.data = xcenters
    hdu3.header['EXTNAME'] = 'XCENTERS'
    hdu4 = fits.ImageHDU()
    hdu4.data = ycenters
    hdu4.header['EXTNAME'] = 'YCENTERS'
    
    hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3, hdu4])
    hdul.writeto(outname, overwrite=True)
    hdul.close()   

def fitSourcePositions(rootdir, channel, dichroic):
    from fifipy.kvector import fitSourcePosition, saveSourcePositions
    from astropy.io import fits
    import numpy as np
    from glob import glob as gb
    import os
    
    # Select files
    filenames = channel+'*_D'+dichroic+'_K*.fits'
    matchfile = os.path.join(rootdir, '**', filenames)
    files = gb(matchfile, recursive=True)
    nfiles = len(files)
    if nfiles > 0:
        files = np.array(files)
        kmirr = np.array([int(file.split('_')[2][1:]) for file in files])
        gpos = np.array([int(file.split('_')[3][1:-5]) for file in files])
        dates = []
        for f in files:
            header = fits.getheader(f)
            dates.append(header['OBSDATE'])
        dates = np.array(dates)
        # Sort by date
        idx = np.argsort(dates)
        files = files[idx]
        kmirr = kmirr[idx]
        gpos = gpos[idx]
        # Compute centers
        xcenters = np.zeros((nfiles, 25))
        ycenters = np.zeros((nfiles, 25))
        for i, f in enumerate(files):
            print('File ', f)
            xcen, ycen = fitSourcePosition(f, plot=False)
            xcenters[i] = xcen
            ycenters[i] = ycen
        # Save result
        saveSourcePositions(rootdir, channel, dichroic, kmirr, gpos, xcenters, ycenters)           
        
        
def computeDeltaVector(infile):
    
    import matplotlib.pyplot as plt
    from astropy.io import fits
    import numpy as np
    import os
    from lmfit.models import LinearModel
    from fifipy.stats import biweightLocation

    rootdir = os.path.dirname(infile)
    with fits.open(infile) as hdul:
        header = hdul[0].header
        xcen = hdul['XCENTERS'].data
        ycen = hdul['YCENTERS'].data
        kmirror = hdul['KMIRROR'].data
        grating = hdul['GRATING'].data
        
    channel = header['CHANNEL']
    dichroic = header['DICHROIC']
    grat = np.unique(grating)
    kmir = np.unique(kmirror)

    ng = len(grat)
    x0 = np.zeros(ng)
    y0 = np.zeros(ng)
    r  = np.zeros(ng)
    alpha = np.zeros(ng)

    alphak = (kmir[0] - 52) * 0.0871

    for i in range(ng):
        id1 = (kmirror == kmir[0]) & (grating == grat[i])
        id2 = (kmirror == kmir[1]) & (grating == grat[i])
        x1, y1 = np.reshape(xcen[id1,:],25), np.reshape(ycen[id1,:],25)
        x2, y2 =  np.reshape(xcen[id2,:],25),  np.reshape(ycen[id2,:],25)
        if channel == 'B':
            idx = np.array([7,8,9,12,13,14,17,18,19])
            rotation_achieved = - np.arctan2(y1[13]-y1[11],x1[13]-x1[11]) * 180/np.pi
        else:
            idx = np.array([12,13,14])
            rotation_achieved = 180 - np.arctan2(y1[13]-y1[11],x1[13]-x1[11]) * 180/np.pi
        x0[i] = biweightLocation(0.5*(x1[idx-1]+x2[idx-1]))
        y0[i] = biweightLocation(0.5*(y1[idx-1]+y2[idx-1]))
        dx = x1[12] - x0[i]
        dy = y1[12] - y0[i]
        #alpha[i] = - alphak - np.arctan2(dy, dx) * 180/np.pi
        print(alphak, rotation_achieved)
        alpha[i] = -rotation_achieved - np.arctan2(dy, dx) * 180/np.pi
        r[i] = np.sqrt((x1[12] - x0[i])**2+ (y1[12] - y0[i])**2)

    if channel == 'B':
        order = np.array([2,2,2,1,1,1,2])
        id2 = order == 2
        id1 = order == 1
    
    # Previous results (from Sebastian - Aug 2020)
    if channel == 'R':
        if dichroic == '105':
            bx_, ax_ = -9.8232E-01, -1.5215E-07
            by_, ay_ = 1.1017E+00, -6.6339E-08
        else:
            bx_, ax_ = -9.1213E-01, -2.0640E-07
            by_, ay_ = 1.2067E+00, -2.1206E-07
    else:
        if dichroic == '105':
            bx_, ax_ = 5.5525E-01, 5.9397E-07
            by_, ay_ = 9.4078E-01, -5.1105E-09
        else:
            bx_, ax_ = -1.8782E-02, 5.9025E-07
            by_, ay_ = 1.0713E+00, 1.3475E-08

 
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
    if channel == 'R':
        ax1.plot(grat, r, 'o', color='red')
    else:
        ax1.plot(grat[id1], r[id1], 'o',color='red', label='order 1')
        ax1.plot(grat[id2], r[id2], 'o',color='blue', label='order 2')
        ax1.legend()
    ax1.grid()
    ax1.set_title('R')
    if channel == 'R':
        ax2.plot(grat, alpha, 'o', color='red')
    else:
        ax2.plot(grat[id1], alpha[id1], 'o',color='red', label='order 1')
        ax2.plot(grat[id2], alpha[id2], 'o',color='blue', label='order 2')
        ax2.legend()
    ax2.grid()
    ax2.set_title('$\\alpha_0$')
    plt.show()

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
    
    dx = r * np.cos(alpha * np.pi/180)
    dy = -r * np.sin(alpha * np.pi/180)
    # Plot and fit
    model = LinearModel()
    pars = model.guess(dx, x=grat)
    out = model.fit(dx, pars, x=grat)
    ax = out.params['slope'].value
    bx = out.params['intercept'].value
    sax = out.params['slope'].stderr
    sbx = out.params['intercept'].stderr
    
    if channel == 'R':
        ax1.plot(grat, dx, 'o')
    else:
        ax1.plot(grat[id1], dx[id1], 'o',color='red', label='order 1')
        ax1.plot(grat[id2], dx[id2], 'o',color='blue', label='order 2')
        
    ax1.plot(grat, bx_ + ax_ * grat, label='Sebastian')
    ax1.plot(grat, out.best_fit, label='Dario')
    ax1.grid()
    ax1.set_title(channel + ' ' + dichroic)
    if channel == 'R':
        ax1.set_ylim(-1.5,-0.5)
    else:
        ax1.set_ylim(0.0,1.6)
    ax1.set_ylabel('$\Delta$x [mm]')
    ax1.set_xlabel('Grating position [ISU]')
    ax1.legend()

    model = LinearModel()
    pars = model.guess(dy, x=grat)
    out = model.fit(dy, pars, x=grat)
    ay = out.params['slope'].value
    by = out.params['intercept'].value
    say = out.params['slope'].stderr
    sby = out.params['intercept'].stderr

    if channel == 'R':
        ax2.plot(grat, dy, 'o')
    else:
        ax2.plot(grat[id1], dy[id1], 'o',color='red', label='order 1')
        ax2.plot(grat[id2], dy[id2], 'o',color='blue', label='order 2')
    ax2.plot(grat, by_ + ay_ * grat, label='Sebastian')
    ax2.plot(grat, out.best_fit, label='Dario')
    ax2.grid()
    ax2.set_ylim(0.6,1.6)
    ax2.set_title(channel + ' ' + dichroic)
    ax2.set_ylabel('$\Delta$y [mm]')
    ax2.set_xlabel('Grating position [ISU]')
    ax2.legend()
    plt.show()
    filename = channel + dichroic + '.pdf'
    fig.savefig(os.path.join(rootdir,filename))

    if channel == 'R':
        gp = 822000
    else:
        gp = 1195000
    Rx = (ax * gp + bx)*0.842
    Ry = - (ay * gp + by) * 0.842
    print('ax {0:.4e} +/- {1:.4e}  bx {2:.4e} +/- {3:.4e}'.format(ax,sax,bx,sbx))
    print('ay {0:.4e} +/- {1:.4e}  by {2:.4e} +/- {3:.4e}'.format(ay,say,by,sby))

    if channel == 'R':
        oformat = '99999999  r  '+dichroic
    else:
        oformat = '99999999  b  '+dichroic
    oformat += '{0:13.4e} {1:13.4e}  {2:5.2f}    {3:13.4e} {4:13.4e}  {5:5.2f}'

    print(' Date    ch dich   b_x        a_x           Rx_cx(U) b_y         a_y        Rx_cy(V) ')
    print(oformat.format(bx,ax,Rx,by,ay,Ry))
    return oformat.format(bx,ax,Rx,by,ay,Ry)
