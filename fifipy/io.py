import numpy as np
from astropy.io import fits

def readAllData(fitsfile):
    """Data reader for Level 1 raw FIFI-LS files."""

    hdulist = fits.open(fitsfile)
    scidata = hdulist[1].data
    header = hdulist[0].header
    data = scidata.DATA
    hdulist.close()
    procstat = header['PROCSTAT']
    #  obstype = header['OBSTYPE']
    if procstat != 'LEVEL_1':
        print ("This program works only with raw FIFI-LS files (Level 1)")
        return
    else:
        detchan = header['DETCHAN']
        obsdate = header['DATE-OBS']
        dichroic = header['DICHROIC']
        if detchan == 'RED':
            #wvc = header['G_WAVE_R']
            ncycles = header['C_CYC_R']
            start = header['G_STRT_R']
            step = header['G_SZUP_R']
            ngrat = header['G_PSUP_R']
            order = 1
        else:
            #wvc = header['G_WAVE_B']
            ncycles = header['C_CYC_B']
            start = header['G_STRT_B']
            step = header['G_SZUP_B']
            ngrat = header['G_PSUP_B']
            order = header['G_ORD_B']

        filegpid=header['FILEGPID']    
        nodbeam = header['NODBEAM']
        # Position
        xmap = float(header['DLAM_MAP'])
        ymap = float(header['DBET_MAP'])
        xoff = float(header['DLAM_OFF'])
        yoff = float(header['DBET_OFF'])
        ra   = float(header['OBSLAM'])
        dec  = float(header['OBSBET'])
        dx = (xmap+xoff)/3600.
        dy = (ymap+yoff)/3600.
        # House keeping
        alti_sta = header['ALTI_STA']
        alti_end = header['ALTI_END']
        za_sta = header['ZA_START']
        za_end = header['ZA_END']
        wv_sta = header['WVZ_STA']
        wv_end = header['WVZ_END']
        angle = header['DET_ANGL']
        filename = header['FILENAME']
        filenum = int(filename[:5])            
        data = np.float32(data)+2**15  # signed integer to float
        data *= 3.63/65536.            # ADU to V
        nramps = np.size(data[:,0,0])
        if nramps < (ncycles*4*ngrat*32):
            print ("WARNING: Number of ramps does not agree with header for ",
                   fitsfile)
            print('n ramps ', nramps)
            print('ncycles, ngrat ', ncycles, ngrat)
        else:
            data = data[:ncycles*4*ngrat*32,:,:25]
            flux = data.reshape(ngrat,ncycles*4*32,18,25)
            gratpos = start+step*np.arange(ngrat)
            aor = (detchan, order, dichroic, ncycles, 
                   nodbeam, filegpid, filenum)
            hk  = (obsdate, (ra,dec), (dx,dy), angle, (za_sta,za_end), 
                   (alti_sta,alti_end), (wv_sta,wv_end))
            return aor, hk, gratpos, flux


def readData(fitsfile, subtractZero=True):
    """Data reader for Level 1 raw FIFI-LS files."""

    hdulist = fits.open(fitsfile)
    scidata = hdulist[1].data
    header = hdulist[0].header
    data = scidata.DATA
    hdulist.close()
    procstat = header['PROCSTAT']
    #  obstype = header['OBSTYPE']
    if procstat != 'LEVEL_1':
        print ("This program works only with raw FIFI-LS files (Level 1)")
        return
    else:
        try:
            detchan = header['DETCHAN']
        except:
            detchan = header['CHANNEL']
        obsdate = header['DATE-OBS']
        telra = header['TELRA']
        teldec = header['TELDEC']
        dichroic = header['DICHROIC']
        if detchan == 'RED':
            #wvc = header['G_WAVE_R']
            ncycles = header['C_CYC_R']
            start = header['G_STRT_R']
            step = header['G_SZUP_R']
            ngrat = header['G_PSUP_R']
            order = 1
        else:
            #wvc = header['G_WAVE_B']
            ncycles = header['C_CYC_B']
            start = header['G_STRT_B']
            step = header['G_SZUP_B']
            ngrat = header['G_PSUP_B']
            order = header['G_ORD_B']

        try:
            filegpid=header['FILEGPID']    
        except:
            filegpid='NONE'
        nodbeam = header['NODBEAM']
        # Position
        xmap = float(header['DLAM_MAP'])
        ymap = float(header['DBET_MAP'])
        xoff = float(header['DLAM_OFF'])
        yoff = float(header['DBET_OFF'])
        ra   = float(header['OBSLAM'])
        dec  = float(header['OBSBET'])
        dx = (xmap+xoff)/3600.
        dy = (ymap+yoff)/3600.
        # House keeping
        alti_sta = header['ALTI_STA']
        alti_end = header['ALTI_END']
        za_sta = header['ZA_START']
        za_end = header['ZA_END']
        wv_sta = header['WVZ_STA']
        wv_end = header['WVZ_END']
        angle = header['DET_ANGL']
        filename = header['FILENAME']
        filenum = int(filename[:5])            
        data = np.float32(data)+2**15  # signed integer to float
        data *= 3.63/65536.            # ADU to V
        if subtractZero:
            openpix = data[:,0,:25]
            for i in range(1,18):
                data[:,i,:25] -= openpix
        
        nramps = np.size(data[:,0,0])
        if nramps < (ncycles*4*ngrat*32):
            print ("WARNING: Number of ramps does not agree with header for ",
                   fitsfile)
            print('n ramps ', nramps)
            print('ncycles, ngrat ', ncycles, ngrat)
        else:
            data = data[:ncycles*4*ngrat*32,1:17,:25]
            flux = data.reshape(ngrat,ncycles*4*32,16,25)
            gratpos = start+step*np.arange(ngrat)
            aor = (detchan, order, dichroic, ncycles, 
                   nodbeam, filegpid, filenum)
            hk  = (obsdate, (telra, teldec), (ra,dec), (dx,dy), angle, (za_sta,za_end), 
                   (alti_sta,alti_end), (wv_sta,wv_end))
            return aor, hk, gratpos, flux
        
def saveSlopeFits(gpos, dichroic, obsdate, detchan, order, specs, wave, dwave, 
                  outname, xcoords=None, ycoords=None, kmirr=None, gratpos=None):
    """Save in a FITS file the results of slope fitting and wavelength calibration."""
    hdu = fits.PrimaryHDU()
    hdu.header['CHANNEL'] = detchan
    hdu.header['ORDER'] = order
    hdu.header['DICHROIC'] = dichroic
    hdu.header['OBSDATE'] = obsdate
    if kmirr is not None:
        hdu.header['KMIRRPOS'] = kmirr
    if gratpos is not None:
        hdu.header['GRATPOS'] = gratpos
    hdu1 = fits.ImageHDU()
    hdu1.data = gpos
    hdu1.header['EXTNAME'] = 'Grating Position'
    hdu2 = fits.ImageHDU()
    hdu2.data = specs
    hdu2.header['EXTNAME'] = 'SPECS'
    hdu3 = fits.ImageHDU()
    hdu3.data = wave
    hdu3.header['EXTNAME'] = 'WAVE'
    hdu4 = fits.ImageHDU()
    hdu4.data = dwave
    hdu4.header['EXTNAME'] = 'DWAVE'
    if (xcoords is None) & (ycoords is None):
        hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3, hdu4])
    elif (xcoords is not None) & (ycoords is not None):
        hdu5 = fits.ImageHDU()
        hdu5.data = xcoords
        hdu5.header['EXTNAME'] = 'XCOORDS'
        hdu6 = fits.ImageHDU()
        hdu6.data = ycoords
        hdu6.header['EXTNAME'] = 'YCOORDS'
        hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6])
    hdul.writeto(outname, overwrite=True)
    hdul.close()
    
def readSlopeFits(path, filename):
    with fits.open(path+filename) as hdl:
        g = hdl['Grating Position'].data
        w = hdl['WAVE'].data
        dw = hdl['DWAVE'].data
        s = hdl['SPECS'].data
    return g, w, dw, s

def saveMediumSpectrum(w, medspec, path, filename):
    # Save the BB curve
    hdu = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU()
    hdu1.data = w
    hdu1.header['EXTNAME'] = 'Wavelength'
    hdu2 = fits.ImageHDU()
    hdu2.data = medspec
    hdu2.header['EXTNAME'] = 'Flux'
    hdul = fits.HDUList([hdu, hdu1, hdu2])
    hdul.writeto(path+filename, overwrite=True)
    hdul.close()   

def readMediumSpectrum(file, silent=False):
    hdl = fits.open(file)
    if silent == False:
        hdl.info()
    w = hdl['Wavelength'].data
    f = hdl['Flux'].data
    hdl.close()
    return w, f

def saveFlats(w, specflat, especflat, spatflat, channel, outfile):
    hdu = fits.PrimaryHDU()
    hdu.header['CHANNEL'] = channel
    hdu1 = fits.ImageHDU()
    hdu1.data = w
    hdu1.header['EXTNAME'] = 'WAVE'
    hdu2 = fits.ImageHDU()
    hdu2.data = specflat
    hdu2.header['EXTNAME'] = 'SPECFLAT'
    hdu3 = fits.ImageHDU()
    hdu3.data = especflat
    hdu3.header['EXTNAME'] = 'ESPECFLAT'
    hdu4 = fits.ImageHDU()
    hdu4.data = spatflat
    hdu4.header['EXTNAME'] = 'SPATFLAT'
    hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3, hdu4])
    hdul.writeto(outfile, overwrite=True)
    hdul.close()   
    
def saveSpatFlats(spatflat, dates, channel, outfile):
    hdu = fits.PrimaryHDU()
    hdu.header['CHANNEL'] = channel
    hdu1 = fits.ImageHDU()
    hdu1.data = dates
    hdu1.header['EXTNAME'] = 'DATES'
    hdu2 = fits.ImageHDU()
    hdu2.data = spatflat
    hdu2.header['EXTNAME'] = 'SPATFLAT'
    hdul = fits.HDUList([hdu, hdu1, hdu2])
    hdul.writeto(outfile, overwrite=True)
    hdul.close()   
    
def saveSpecFlats(w, specflat, especflat, channel, order, dichroic, outfile):
    hdu = fits.PrimaryHDU()
    hdu.header['CHANNEL'] = channel
    hdu.header['ORDER'] = order
    hdu.header['DICHROIC'] = dichroic
    hdu1 = fits.ImageHDU()
    hdu1.data = w
    hdu1.header['EXTNAME'] = 'WAVE'
    hdu2 = fits.ImageHDU()
    hdu2.data = specflat
    hdu2.header['EXTNAME'] = 'SPECFLAT'
    hdu3 = fits.ImageHDU()
    hdu3.data = especflat
    hdu3.header['EXTNAME'] = 'ESPECFLAT'
    hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3])
    hdul.writeto(outfile, overwrite=True)
    hdul.close()   
   
def saveResponse(wtot, response, eresponse, output, channel, order, dichroic):
    import numpy as np
    from astropy.io import fits
    wr = np.arange(np.nanmin(wtot),np.nanmax(wtot),0.2)
    fr = response(wr)
    er = eresponse(wr)
    data = []
    data.append(wr)
    data.append(fr)
    data.append(er)
    data = np.array(data)
    hdr = fits.Header()
    hdr['DETCHAN'] = channel
    hdr['ORDER'] = order
    hdr['DICHOIC'] = dichroic
    hdr['XUNIT'] = 'microns'
    hdr['YUNIT'] = 'ADU/Hz/Jy'
    hdu = fits.PrimaryHDU(data, header=hdr)
    hdul = fits.HDUList([hdu])
    hdul.writeto(output,overwrite=True)
    
def readResponse(file):
    from astropy.io import fits
    
    hdr = fits.open(file)
    data = hdr['PRIMARY'].data
    hdr.close()
    w = data[0,:]
    f = data[1,:]
    e = data[2,:]
    
    return w, f, e