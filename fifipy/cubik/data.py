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
        #self.flux = hdl['UNCORRECTED_FlUX'].data
        #self.eflux = hdl['UNCORRECTED_ERROR'].data
        self.flux = hdl['FlUX'].data
        self.eflux = hdl['ERROR'].data
        self.wave = hdl['wavelength'].data
        self.n = len(self.wave)
        # Read reference wavelength from file group name
        try:
            self.l0 = self.header['RESTWAV']
            print('Rest wavelength is ', self.l0)
        except:
            try:
                if self.channel == 'RED':
                    filegp = self.header['FILEGP_R'].strip()
                else:
                    filegp = self.header['FILEGP_B'].strip()
                print('file group id ', filegp)
                names = filegp.split('_')
                wr = names[-1]
                print('wr is ',wr)
                self.l0 = float(wr)
            except:
                self.l0 = np.nanmedian(self.wave)
                print('No file group present, assumed central wavelength ', self.l0)
        print('ref wav ', self.l0)
        # Found reference pixel
        self.n0 = np.argmin((np.abs(self.wave-self.l0)))
        self.X = hdl['X'].data
        self.Y = hdl['Y'].data
        self.R = self.header['RESOLUN']  # spectral resolution
        self.pixscale = self.header['PIXSCAL']
        utrans = hdl['UNSMOOTHED_TRANSMISSION'].data
        self.wt = utrans[0,:]
        self.at = utrans[1,:]
        D = 2.5 #m
        lam = np.nanmean(self.wave)*1.e-6
        fwhm = 1.01 * lam / D * 180 / np.pi * 3600 #mirror with obstruction  
        #fwhm = 1.22 * lam / D * 180 / np.pi * 3600 #mirror without obstruction (close to resolution of FIFI-LS)  
        #sigma = fwhm / 2.355
        self.radius =  fwhm * 0.5 # HWHM
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
        
        
class spectralCloudOld(object):
    """Cloud of points from CAL files."""
    
    def __init__(self, path, pixscale, extension='WSH'):
        calfiles = fnmatch.filter(os.listdir(path),"*"+extension+"*.fits")
        nstack = 0
        self.pixscale = pixscale
        for calfile in sorted(calfiles):
            hlf = fits.open(os.path.join(path, calfile))
            header = hlf['PRIMARY'].header
            data = hlf[1].data
            hlf.close()
            #dx = header['dlam_map']
            #dy = header['dbet_map']
            obslam = header['OBSLAM']
            obsbet = header['OBSBET']
            channel = header['DETCHAN']
            #dx = header['dlam_map']
            #dy = header['dbet_map']
            #detangle = header['det_angl']
            #mid = int(header['MISSN-ID'][-3:])
            #ca = np.cos(detangle * np.pi / 180.)
            #sa = np.sin(detangle * np.pi / 180.)
            ys = data.YS / 3600.
            xs = - data.XS / 3600. / np.cos( (ys+obsbet) * np.pi / 180.)
            skyangle = header['SKY_ANGL']
            # Rotate coordinates
            if skyangle != 0:
                angle = skyangle * np.pi/180.
                cosa = np.cos(angle)
                sina = np.sin(angle)
                x_ = xs.copy()
                y_ = ys.copy()
                xs = x_ * cosa + y_ * sina
                ys = -x_ * sina + y_ * cosa
            ys += obsbet
            xs += obslam
            platscale = header['PLATSCAL']
            if channel == 'RED':
                #pixfactor = (12.2*12.5)/pixscale**2 # size of pixels from Colditz et al. 2018
                self.pixfactor = (platscale*3/pixscale)**2
            else:
                #pixfactor = (6.14*6.25)/pixscale**2
                self.pixfactor = (platscale*1.5/pixscale)**2
            #fs = data.UNCORRECTED_DATA / pixfactor   # Normalize for resampling
            #print('Platscale ', platscale)
            #print('Resampling factor: ', self.pixfactor)
            fs = data.DATA / self.pixfactor   # Normalize for resampling
            ws = data.LAMBDA
            ns,nz,ny,nx = np.shape(ws)
            print(ns,nz,ny,nx)
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
  
class spectralCloud(object):
    """Cloud of points from CAL files."""
    
    def __init__(self, path, pixscale, extension='WSH'):
        calfiles = fnmatch.filter(os.listdir(path),"*"+extension+"*.fits")
        nstack = 0
        self.pixscale = pixscale
        x = []
        y = []
        w = []
        f = []
        for calfile in sorted(calfiles):
            print(calfile)
            with fits.open(os.path.join(path, calfile)) as hlf:
            #hlf = fits.open(os.path.join(path, calfile))
                header = hlf['PRIMARY'].header
                obslam = header['OBSLAM']
                obsbet = header['OBSBET']
                channel = header['DETCHAN']
                ys = hlf['YS'].data / 3600.
                xs = -hlf['XS'].data / 3600. / np.cos( (ys+obsbet) * np.pi / 180.)
                skyangle = header['SKY_ANGL']
                # Rotate coordinates
                if skyangle != 0:
                    angle = skyangle * np.pi/180.
                    cosa = np.cos(angle)
                    sina = np.sin(angle)
                    x_ = xs.copy()
                    y_ = ys.copy()
                    xs = x_ * cosa + y_ * sina
                    ys = -x_ * sina + y_ * cosa
                ys += obsbet
                xs += obslam
                platscale = header['PLATSCAL']
                print('channel ', channel)
                if channel == 'RED':
                    #pixfactor = (12.2*12.5)/pixscale**2 # size of pixels from Colditz et al. 2018
                    self.pixfactor = (platscale*3/pixscale)**2
                else:
                    #pixfactor = (6.14*6.25)/pixscale**2
                    self.pixfactor = (platscale*1.5/pixscale)**2
                #fs = data.UNCORRECTED_DATA / pixfactor   # Normalize for resampling
                fs = hlf['FLUX'].data / self.pixfactor   # Normalize for resampling
                ws = hlf['LAMBDA'].data
                print(np.shape(fs))
                
            x.append(np.ravel(xs))
            y.append(np.ravel(ys))
            w.append(np.ravel(ws))
            f.append(np.ravel(fs))
                
                
            #shape = np.shape(xs)  
            #if len(shape) == 2:
            #    nz, nxy = shape
            #    xs = xs.reshape(1, nz, 5, 5)
            #    ys = ys.reshape(1, nz, 5, 5)
            #    ws = ws.reshape(1, nz, 5, 5)
            #    fs = fs.reshape(1, nz, 5, 5)
            #    
            #if nstack == 0:
            #    x = xs
            #    y = ys
            #    f = fs
            #    w = ws
            #    nstack += 1
            #else:
            #    x = np.vstack((x, xs))
            #    y = np.vstack((y, ys))
            #    f = np.vstack((f, fs))
            #    w = np.vstack((w, ws))
            #    nstack += 1
                
            #print(np.shape(x))
            #ns,nz,ny,nx = np.shape(ws)
            #print(ns,nz,ny,nx)
            #for i in range(ns):
            #    if nstack == 0:
            #        x = xs[i]
            #        y = ys[i]
            #        f = fs[i]
            #        w = ws[i]
            #    else:
            #        x = np.vstack((x, xs[i]))
            #        y = np.vstack((y, ys[i]))
            #        f = np.vstack((f, fs[i]))
            #        w = np.vstack((w, ws[i]))
            #    nstack += 1
        x = np.concatenate(x)
        y = np.concatenate(y)
        w = np.concatenate(w)
        f = np.concatenate(f)
        print('shape of f ',np.shape(f))
        #print('stack of ', nstack,' frames')
        idx = np.isfinite(f)
        print('n finite ',np.sum(idx))
        self.y = y[idx]
        self.x = x[idx]
        self.w = w[idx]
        self.f = f[idx]   
        print('final shape ', np.shape(self.x))
        
    
class Spectrum(object):
    """Spectrum at coordinate."""
    def __init__(self, wave, flux, eflux, w, f, distance, wt, at, hwhm):
        self.wave = wave
        self.flux = flux
        self.eflux = eflux
        self.w = w
        self.f = f
        self.d = distance
        self.wt = wt
        self.at = at
        self.hwhm = hwhm

    def set_colors(self):
        colors = []
        dists = self.d / self.hwhm
        for d in dists:
            if d < .5:
                color = 'lime'
            elif (d >= .5) & (d < 1.0):
                color = 'yellow'
            elif (d >= 1.0 ) & (d < 1.5):
                color = 'green'
            elif (d >= 1.5 ) & (d < 2):
                color = 'cyan'
            elif (d >= 2) & (d < 2.5):
                color = 'blue'
            elif (d >= 2.5 ) & (d < 3.0):
                color ='orange'
            else:
                color = 'magenta'
            colors.append(color)
        self.colors = np.array(colors)
        
    def set_filter(self, delta, radius, pixscale):
        #from fifipy.stats import biweight
        self.delta = delta * 0.5  # Half of the FWHM 
        areafactor = (pixscale/radius)**2/np.pi
        #print('pixscale, radius, area factor is ', pixscale, radius, areafactor)
        # O-th run - approx baseline, get rid of outliers from baseline
        m0 = np.nanmedian(np.ravel(self.f))
        if m0 < 0:
            m0 = 0
        m1 = np.nanmedian(np.abs(np.ravel(self.f) - m0))
        idx = np.abs(self.f - m0) < 4 * m1
        m0 = np.nanmedian(np.ravel(self.f[idx]))
        m1 = np.nanmedian(np.abs(np.ravel(self.f[idx]) - m0))
        self.baseline = m0
        self.m1 = m1
        
        #trans = np.interp(self.w, self.wt, self.at)
        
        flux = []
        n = []
        # Compute the number of points for spectral resolution
        for wm in self.wave:
            idx = np.abs(self.w - wm) <= self.delta
            n.append(np.sum(idx))
        # Weight the delta as a function of the median
        n = np.array(n)
        n90 = np.percentile(n, 90, interpolation = 'midpoint') 
        deltas = []
        # I can convert this to multi-processing to speed up the output
        # First run - flux1 
        # I should filter on a third of FWHM (Shannon's criterion)
        for wm, nm in zip(self.wave, n):
            delta = self.delta * np.sqrt(n90 / nm) # Adjust interval
            deltas.append(delta)
            idx = (np.abs(self.w - wm) <= delta) & (np.abs(self.f - self.baseline) < 5 * self.m1)
            # Compute the biweight mean
            #biw, sbiw = biweight(self.f[idx])
            flux.append(np.nanmedian(self.f[idx]))
            #fi = self.f[idx]
            #wi = self.w[idx]
            #di = self.d[idx] / radius
            #dw = (wi - wm) / delta
            #wt = (1 - di**2)**2 * (1 - dw**2)**2 # Biweight
            #f0 = np.sum(fi*wt)/np.sum(wt)
            #flux.append(f0)
        # Rejection of outliers
        for kiter in range(10):       
            flux1 = np.array(flux)
            nn = []
            wr = []
            fr = []
            flux = []
            noise = []
            for wm, nm, delta in zip(self.wave, n, deltas):
                idx = np.abs(self.w - wm) <= delta
                nn.append(np.sum(idx))
                fi = self.f[idx]
                wi = self.w[idx]
                #ti = trans[idx]
                di = self.d[idx] / radius
                dw = (wi - wm) / delta
                idf = np.isfinite(flux1)
                if np.sum(idf) > 10:
                    residual = fi - np.interp(wi, self.wave[idf], flux1[idf])
                    m0 = np.nanmedian(residual)
                    m1 = np.nanmedian(np.abs(residual - m0))
                    if m1 > self.m1:
                        m1 = self.m1
                    # idx = np.abs(residual) < 4 * m1
                    idx = (residual < 5 * m1) & (residual > - 5 * m1) & (fi - self.baseline > - 4 * self.m1)
                    wr.extend(wi[~idx])
                    fr.extend(fi[~idx])
                    fi = fi[idx]
                    wi = wi[idx]
                    di = di[idx]
                    dw = dw[idx]
                    #ti = ti[idx]
                    """
                    Formula from Wikipedia:
                        en.wikipedia.org/wiki/Weighted_arithmetic_mean#Bootstrapping_validation
                        Note that the flux is already renormalized to the area
                        of the pixel. So, the computed error is already correctly
                        normalized to the area of the cube pixel.
                    """
                    # Biweight
                    #wt = (1 - di**2)**2 * (1 - dw**2)**2
                    # Tricube
                    #wt = (1 - np.abs(di)**3)**3 * (1 - np.abs(dw)**3)**3
                    # Gaussian
                    wt = np.exp(- 0.5 * (2.355*di)**2) * np.exp (-0.5 * (2.355*dw)**2) 
                    #wt *= ti # weight by transmission
                    w1 = np.sum(wt)
                    f0 = np.sum(fi * wt) / w1
                    nt = len(wt)
                    e2 = nt / (nt-1) * np.nansum((wt * (fi - f0))**2) / w1**2                    
                    #e2 = np.nansum((nt*wt*fi/wtsum - f0)**2)/(nt-1)
                    #e0 = np.sqrt(e2/nt)
                    e0 = np.sqrt(e2 / areafactor)
                    # Should we consider the ratio circle to pixel ?
                    #biw, sbiw = biweight(fi)
                    #f0 = biw
                    #e0 = sbiw / np.sqrt(areafactor * nt)
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
        
def filterSpectrum(wave, w, f, d, delta, radius, areafactor):
        #from fifipy.stats import biweight
        delta = delta * 0.5  # Half of the FWHM 
        flux = []
        # O-th run - approx baseline, get rid of outliers from baseline
        m0 = np.nanmedian(np.ravel(f))
        if m0 < 0:
            m0 = 0
        m1 = np.nanmedian(np.abs(np.ravel(f) - m0))
        idx = np.abs(f - m0) < 3 * m1
        m0 = np.nanmedian(np.ravel(f[idx]))
        m1 = np.nanmedian(np.abs(np.ravel(f[idx]) - m0))
        base0 = m0
        base1 = m1

        
        m0 = np.nanmedian(np.ravel(f))
        m1 = np.nanmedian(np.abs(np.ravel(f) - m0))
        idx = (f - m0) < - 4 * m1
        f[idx] = np.nan
        
        # Discard everything lower than - (k x sigma) 
        # First run - flux1 is 1st approximation
        for wm in wave:
            idx = (np.abs(w - wm) <= delta) & ((f - base0) > - 3 * base1)
            # Compute the biweight mean
            if np.sum(idx) > 5:
                fi = f[idx]
                biw = np.nanmedian(fi)
                #biw, sbiw = biweight(fi)
            else:
                biw = np.nan
            flux.append(biw)
            #fi = f[idx]
            #wi = w[idx]
            #di = d[idx] / radius
            #if np.sum(idx) > 5:
            #    dw = (wi - wm) / delta
            #    wt = (1 - di**2)**2 * (1 - dw**2)**2 # Biweight
            #    f0 = np.sum(fi*wt)/np.sum(wt)
            #    #print(f0)
            #else:
            #    f0 = np.nan
            #flux.append(f0)
        # Reject excessive negative numbers
        #flux = np.array(flux)
        #mf = np.nanmedian(flux)
        #ms = np.nanmedian(np.abs(flux-mf))
        #idx = (flux - mf) < -20 * ms
        #flux[idx] = np.nan
        
        # Rejection of outliers
        for kiter in range(6):       
            flux1 = np.array(flux)
            #print('flux min ', np.nanmin(flux1))
            flux = []
            noise = []
            for wm in wave:
                idx = np.abs(w - wm) <= delta
                fi = f[idx]
                wi = w[idx]
                di = d[idx] / radius
                dw = (wi - wm) / delta
                idf = np.isfinite(flux1)
                if (np.sum(idf) > 10) & (np.sum(idx) > 5):
                    residual = fi - np.interp(wi, wave[idf], flux1[idf])
                    m0 = np.nanmedian(residual)
                    m1 = np.nanmedian(np.abs(residual - m0))
                    if m1 > base1:
                        m1 = base1
                    # idx = np.abs(residual) < 4 * m1
                    idx = (residual < 4 * m1) & (residual > - 3.5 * m1) & (fi - base0 > - 3 * base1)
                    fi = fi[idx]
                    wi = wi[idx]
                    di = di[idx]
                    dw = dw[idx]
                    """
                    Formula from Wikipedia:
                        en.wikipedia.org/wiki/Weighted_arithmetic_mean#Bootstrapping_validation
                        Note that the flux is already renormalized to the area
                        of the pixel. So, the computed error is already correctly
                        normalized to the area of the cube pixel.
                    """
                    # Biweight
                    #wt = (1 - di**2)**2 * (1 - dw**2)**2
                    # Tricube
                    #wt = (1 - np.abs(di)**3)**3 * (1 - np.abs(dw)**3)**3
                    # Gaussian (adopt FWHM half of the window)
                    wt = np.exp(- 0.5 * (2.355*di)**2) * np.exp (-0.5 * (2.355*dw)**2) 
                    w1 = np.sum(wt)
                    f0 = np.sum(fi * wt) / w1
                    nt = len(wt)
                    try:
                        e2 = nt / (nt-1) * np.nansum((wt * (fi - f0))**2) / w1**2                    
                        e0 = np.sqrt(e2 / areafactor)
                    except:
                        e0= np.nan
                else:
                    f0 = np.nan
                    e0 = np.nan
                flux.append(f0)
                noise.append(e0)
        return np.array(flux), np.array(noise)
        
def computeNoise(wave, scw, scf, delta, x0, y0, radius, areafactor, center):
        xc, yc = center
        distance = np.hypot(x0 - xc, y0 - yc)   # distance in pixels
        idx = distance <= radius
        dists = distance[idx]
        w = scw[idx]
        f = scf[idx]        
        nflux, noise = filterSpectrum(wave, w, f, dists, delta, radius, areafactor)
        return nflux, noise       
