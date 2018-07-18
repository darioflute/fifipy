import numpy as np
from astropy.io import fits


def readData(fitsfile):
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
        else:
            data = data[:ncycles*4*ngrat*32,1:17,:25]
            flux = data.reshape(ngrat,ncycles*4*32,16,25)
            gratpos = start+step*np.arange(ngrat)
            aor = (detchan, order, dichroic, ncycles, 
                   nodbeam, filegpid, filenum)
            hk  = (obsdate, (ra,dec), (dx,dy), angle, (za_sta,za_end), 
                   (alti_sta,alti_end), (wv_sta,wv_end))
            return aor, hk, gratpos, flux