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
    import re
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
                xcoords = []
                ycoords = []
                match1 = r"(-?\d,-?\d)"
                match2 = r"(\d)x"
                for f in files:
                    x1 = re.search(match1, f)
                    coords = x1.group(1)
                    xcoord, ycoord = coords.split(',')
                    x2 = re.search(match2, f)
                    factor = int(x2.group(1))
                    xcoords.append(int(xcoord)*factor)
                    ycoords.append(int(ycoord)*factor)
                xcoords = np.array(xcoords)
                ycoords = np.array(ycoords)
                # Get basic info from header
                header = fits.getheader(files[0])
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
                print('shape specs ', np.shape(specs))
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
    from fifipy.kvector import readKVfile
    from fifipy.stats import biweightLocation
    #infile = '/home/dario/Python/Fifilab/KVectorData/Reduced/B1_D105_K1092_G180000.fits'
    xcoords, ycoords, specs = readKVfile(infile)
    med = biweightLocation(specs, axis=1)
    fig,axes =plt.subplots(5,5,figsize=(15,15))
    for npix in range(25):
        ax = axes[npix // 5 , npix % 5]
        ax.tricontourf(xcoords, ycoords, med[:,npix], cmap='inferno',levels=14)
        ax.text(0,13,str(npix+1),color='black')
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
        ax.text(0,13,str(npix+1),color='black')
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
    x_ = (x-x0)/sx
    y_ = (y-y0)/sy
    model = A / (2 *np.pi * sx * sy) * np.exp(-0.5 * (x_**2 +y_**2) )
    if data is None:
        return model
    else:
        return model-data

def fitSource(infile, plane):
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

    channel = header['CHANNEL']
    fluxplane = fluxes[plane]
    order = orders[plane]
    wave = waves[plane]
    fig,axes =plt.subplots(5,5,figsize=(15,15))
    xcen = []
    ycen = []
    xsig = []
    ysig = []
    for npix in range(25):
        ax = axes[npix // 5 , npix % 5]
        flux = fluxplane[npix]
        scale = 3.55 # arcsec/mm
        step = 3 #mm
        sigma = getPSFsigma(channel, order, wave)  # Blue   
        amplitude = np.nanmax(flux) - np.nanmedian(flux)
        sigma /= scale/step 
        amplitude *= 2*np.pi*sigma**2
        
        idx = np.argmax(flux)
        x0guess = xcoords[idx]
        y0guess = ycoords[idx]
            
        fit_params = Parameters()
        fit_params.add('x0', value = x0guess, max = 13, min=-13)
        fit_params.add('y0', value = y0guess, max = 13, min=-13)
        fit_params.add('sx', value = sigma, min=sigma*0.5, max=sigma*2)
        fit_params.add('sy', value = sigma, min=sigma*0.5, max=sigma*2)
        fit_params.add('A' , value = amplitude)
        
        out = minimize(residuals2Dgauss, fit_params, 
                       args=(xcoords, ycoords,), 
                       kws={'data': flux-np.nanmedian(flux)})
        outparams = out.params
        x0 = outparams['x0'].value
        y0 = outparams['y0'].value
        sx = outparams['sx'].value
        sy = outparams['sy'].value
        xcen.append(x0)
        ycen.append(y0)
        xsig.append(outparams['x0'].stderr)
        ysig.append(outparams['y0'].stderr)
    
        ax.tricontourf(xcoords, ycoords, flux - np.nanmedian(flux), cmap='inferno', levels=14)
        ellipse = Ellipse((x0,y0), sx*2.355, sy*2.355, 0, fill=False)
        ax.add_patch(ellipse)
        
        ax.plot(x0guess,y0guess,'+',color='white')
        ax.set_xlim(-13,13)
        ax.set_ylim(-13,13)
        ax.grid()
        ax.text(0,13,str(npix+1),color='black')
    plt.show()
    xcen = np.array(xcen)
    ycen = np.array(ycen)
    xsig = np.array(xsig)
    ysig = np.array(ysig)
    return xcen, ycen, xsig, ysig
