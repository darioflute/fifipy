import numpy as np
import os
import fnmatch
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

class spectralCube(object):
    """Spectral cube from FIFI-LS."""
    
    def __init__(self, infile):
        hdl = fits.open(infile, memmap=None, ignore_blank=True)
        self.header = hdl['PRIMARY'].header
        self.wcs = WCS(self.header).celestial
        self.objname = self.header['OBJ_NAME']
        self.flux = hdl['UNCORRECTED_FlUX'].data
        self.eflux = hdl['UNCORRECTED_ERROR'].data
        self.wave = hdl['wavelength'].data
        self.X = hdl['X'].data
        self.Y = hdl['Y'].data
        self.R = self.header['RESOLUN']  # spectral resolution
        self.pixscale = self.header['PIXSCAL']
        utrans = hdl['UNSMOOTHED_TRANSMISSION'].data
        self.wt = utrans[0,:]
        self.at = utrans[1,:]
        D = 2.5 #m
        lam = np.nanmean(self.wave)*1.e-6
        #fwhm = 1.01 * lam / D * 180 / np.pi * 3600 #mirror with obstruction  
        fwhm = 1.22 * lam / D * 180 / np.pi * 3600 #mirror without obstruction (close to resolution of FIFI-LS)  
        #sigma = fwhm / 2.355
        self.radius =  fwhm * 0.5 * 1.5 # kernel 1.5x the FWHM
        self.channel = self.header['DETCHAN']        
        if self.channel == 'BLUE':
            self.order = self.header["G_ORD_B"]
        else:
            self.order = '1'
        self.nz, self.ny, self.nx = np.shape(self.flux)
        xi = np.arange(self.nx); yi = np.arange(self.ny)
        xi,yi = np.meshgrid(xi, yi)
        self.points = np.c_[np.ravel(xi), np.ravel(yi)]
        self.pixscale, ypixscale = proj_plane_pixel_scales(self.wcs) * 3600. # Pixel scale in arcsec
        print('scale is ', self.pixscale, ' arcsec')
        
class spectralCloud(object):
    """Cloud of points from CAL files."""
    
    def __init__(self, path, pixscale, extension='CAL'):
        calfiles = fnmatch.filter(os.listdir(path),"*"+extension+"*.fits")
        nstack = 0
        for calfile in sorted(calfiles):
            hlf = fits.open(os.path.join(path, calfile))
            header = hlf['PRIMARY'].header
            data = hlf[1].data
            hlf.close()
            #dx = header['dlam_map']
            #dy = header['dbet_map']
            obslam = header['obslam']
            obsbet = header['obsbet']
            channel = header['DETCHAN']
            #dx = header['dlam_map']
            #dy = header['dbet_map']
            #detangle = header['det_angl']
            #mid = int(header['MISSN-ID'][-3:])
            #ca = np.cos(detangle * np.pi / 180.)
            #sa = np.sin(detangle * np.pi / 180.)
            ys = data.YS / 3600. + obsbet
            xs = - data.XS / 3600. / np.cos( ys * np.pi / 180.) + obslam
            platscale = header['PLATSCAL']
            if channel == 'RED':
                #pixfactor = (12.2*12.5)/pixscale**2 # size of pixels from Colditz et al. 2018
                pixfactor = (platscale*3/pixscale)**2
            else:
                #pixfactor = (6.14*6.25)/pixscale**2
                pixfactor = (platscale*1.5/pixscale)**2
            fs = data.UNCORRECTED_DATA / pixfactor   # Normalize for resampling
            ws = data.LAMBDA
            ns,nz,ny,nx = np.shape(ws)
            for i in range(ns):
                if nstack == 0:
                    x = xs[i]
                    y = ys[i]
                    f = fs[i]
                    w = ws[i]
                else:
                    x = np.vstack((x, xs[i]))
                    y = np.vstack((y, ys[i]))
                    f = np.vstack((f, fs[i]))
                    w = np.vstack((w, ws[i]))
                nstack += 1
        print('stack of ', nstack,' frames')
        idx = np.isfinite(f)
        self.y = y[idx]
        self.x = x[idx]
        self.w = w[idx]
        self.f = f[idx]        
        
class Spectrum(object):
    """Spectrum at coordinate."""
    def __init__(self, wave, flux, eflux, w, f, distance, wt, at):
        self.wave = wave
        self.flux = flux
        self.eflux = eflux
        self.w = w
        self.f = f
        self.d = distance
        self.wt = wt
        self.at = at

    def set_colors(self):
        colors = []
        dists = self.d
        for d in dists:
            if d < 1.:
                colors.append('lime')
            elif (d >= 1) & (d < 2):
                colors.append('green')
            elif (d >=2 ) & (d < 4):
                colors.append('forestgreen')
            elif (d >=4 ) & (d < 6):
                colors.append('cyan')
            elif (d >= 6) & (d < 8):
                colors.append('blue')
            elif (d >=8 ) & (d < 10):
                colors.append('orange')
            else:
                colors.append('red')
        self.colors = np.array(colors)
        
    def set_filter(self, delta, radius):
        self.delta = delta * 0.5  # Half of the FWHM 
        flux1 = []
        n = []
        # Compute the number of points for spectral resolution
        for wm in self.wave:
            idx = np.abs(self.w - wm) <= self.delta
            n.append(np.sum(idx))
            
        # Weight the delta as a function of the median
        n = np.array(n)
        n90 = np.percentile(n, 90, interpolation = 'midpoint') 
        deltas = []
        # First run
        for wm, nm in zip(self.wave, n):
            delta = self.delta * np.sqrt(n90 / nm) # Adjust interval
            deltas.append(delta)
            idx = np.abs(self.w - wm) <= delta
            fi = self.f[idx]
            wi = self.w[idx]
            di = self.d[idx] / radius
            dw = (wi - wm) / delta
            wt = (1 - di**2)**2 * (1 - dw**2)**2 # Biweight
            f0 = np.sum(fi*wt)/np.sum(wt)
            flux1.append(f0)
        # Rejection of outliers
        flux1 = np.array(flux1)
        flux = []
        noise = []
        nn = []
        wr = []
        fr = []
        for wm, nm, delta in zip(self.wave, n, deltas):
            idx = np.abs(self.w - wm) <= delta
            nn.append(np.sum(idx))
            fi = self.f[idx]
            wi = self.w[idx]
            di = self.d[idx] / radius
            dw = (wi - wm) / delta
            idf = np.isfinite(flux1)
            if np.sum(idf) > 10:
                residual = fi - np.interp(wi, self.wave[idf], flux1[idf])
                m0 = np.nanmedian(residual)
                m1 = np.nanmedian(np.abs(residual - m0))
                idx = np.abs(residual) < 3 * m1
                wr.extend(wi[~idx])
                fr.extend(fi[~idx])
                fi = fi[idx]
                wi = wi[idx]
                di = di[idx]
                dw = dw[idx]
                wt = (1 - di**2)**2 * (1 - dw**2)**2
                wtsum = np.sum(wt)
                nt = len(wt)
                f0 = np.sum(fi * wt) / wtsum
                e2 = np.nansum((nt*wt*fi/wtsum - f0)**2)/(nt-1)
                e0 = np.sqrt(e2/nt)
            else:
                f0 = 0
                e0 = 0
            flux.append(f0)
            noise.append(e0)
        
        self.fflux = np.array(flux)
        self.noise = np.array(noise)
        self.nflux = np.array(n)
        self.deltas = np.array(deltas)
        self.nn = np.array(nn)
        self.wrejected = np.array(wr)
        self.frejected = np.array(fr)