#!/usr/bin/env python
import os
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QSplitter,
                             QAction, QFileDialog, QVBoxLayout,QInputDialog)
from PyQt5.QtCore import Qt
from fifipy.cubik.data import spectralCube, spectralCloud, spectralCloudOld, Spectrum
from fifipy.cubik.graphics import (ImageCanvas, SpectrumCanvas, 
                                   SegmentInteractor, CircleInteractor)

import warnings
warnings.filterwarnings('ignore')

class GUI (QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.title = 'CUBIK - cube viewer and manipulator'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        # Order exit
        self.setAttribute(Qt.WA_DeleteOnClose)
        # Path of package
        self.path0, file0 = os.path.split(__file__)
        
        # Start UI
        self.initUI()
        
    def initUI(self):
        """Define the user interface."""
        self.setWindowTitle(self.title)
        #self.setWindowIcon(QIcon(self.path0+'/icons/sospex.png'))
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Create main widget
        wid = QWidget()
        self.setCentralWidget(wid)
        # Main layout is horizontal
        mainLayout = QHBoxLayout()
        # Horizontal splitter
        self.hsplitter = QSplitter(Qt.Horizontal)        
        # Create main panels
        self.createImagePanel()
        self.createSpectralPanel()
        # Add panels to splitter
        self.hsplitter.addWidget(self.imagePanel)
        self.hsplitter.addWidget(self.spectralPanel)
        # Add panels to main layout
        mainLayout.addWidget(self.hsplitter)
        wid.setLayout(mainLayout)
        self.show()
        # Menu
        self.createMenu()
        
    def createImagePanel(self):
        self.imagePanel = QWidget()
        self.ic = ImageCanvas(self.imagePanel)
        self.imagePanel.layout = QVBoxLayout()
        self.imagePanel.setLayout(self.imagePanel.layout)
        self.imagePanel.layout.addWidget(self.ic)
        
    def createSpectralPanel(self):
        self.spectralPanel = QWidget()
        self.sc = SpectrumCanvas(self.spectralPanel)
        self.spectralPanel.layout = QVBoxLayout()
        self.spectralPanel.setLayout(self.spectralPanel.layout)
        self.spectralPanel.layout.addWidget(self.sc)
        
    def createMenu(self):
        bar = self.menuBar()
        file = bar.addMenu("File")
        file.addAction(QAction("Quit", self, shortcut='Ctrl+q', triggered=self.fileQuit))
        file.addAction(QAction("Open cube", self, shortcut='Ctrl+n', triggered=self.newFile))
        file.addAction(QAction("Compute new cube", self, shortcut='Ctrl+n', triggered=self.newUncertainty))
        file.addAction(QAction("Compute continuum subtracted cube", self, shortcut='Ctrl+n', triggered=self.newContSubtracted))
        bar.setNativeMenuBar(False)

    def fileQuit(self):
        """Quit program."""
        self.close()
    
    def newFile(self):
        """Select files."""
        fd = QFileDialog()
        fd.setLabelText(QFileDialog.Accept, "Import")
        fd.setNameFilters(["Fits Files (*.fit*)", "WXY fits files (*WXY*.fits*)", "All Files (*)"])
        fd.setOptions(QFileDialog.DontUseNativeDialog)
        fd.setViewMode(QFileDialog.List)
        fd.setFileMode(QFileDialog.ExistingFile)
        if (fd.exec()):
            fileName= fd.selectedFiles()
            print('Reading file ', fileName[0])
            # Save the file path for future reference
            self.pathFile, self.WXYfile = os.path.split(fileName[0])
            self.loadFiles(self.pathFile, fileName[0])
            print('Directory ', self.pathFile)
            print('File WXY  ', self.WXYfile)
            try:
                self.initializeImage()
                print('images initialized ')
                self.initializeAperture()
                print('aperture initialized')
                self.initializeSpectrum()
                print('spectra initialized ')
            except:
                print('No spectral cube is defined')
                pass

    def loadFiles(self, path, infile):
        """Load the cube and cal files."""
        try:
            print('Reading file ', infile)
            self.specCube = spectralCube(infile)
            print('Reading CAL files from ', path)
            try:
                self.specCloud = spectralCloud(path, self.specCube.pixscale)
            except:
                self.specCloud = spectralCloudOld(path, self.specCube.pixscale)
        except:
            print('The spectral cube cannot be read')
            
    def initializeImage(self):
        """Display the image and location of cal spaxels."""
        s = self.specCube
        sc = self.specCloud
        #image = np.nanmean(s.flux, axis=0)
        image = s.flux[s.n0,:,:]
        #xy = np.array([[x_, y_] for x_, y_ in zip(sc.x, sc.y)], dtype=np.float64)
        xy = np.column_stack((sc.x, sc.y, sc.flight))
        #xy = np.array([sc.x, sc.y], np.float_)
        self.ic.compute_initial_figure(image, s.wcs, xy)
        pass
    
    def initializeAperture(self):
        # Initialize aperture
        #radius = 10. / self.ic.pixscale
        radius = self.specCube.radius / self.ic.pixscale
        sc = self.specCloud
        r0 = np.nanmedian(sc.x)
        d0 = np.nanmedian(sc.y)
        x0, y0 = self.ic.wcs.wcs_world2pix(r0, d0, 0)
        #print('center circle is ', x0, y0)
        #self.ic.axes.plot(r0, d0, 'o', color='red',transform=self.ic.axes.get_transform('world'))
        self.CI = CircleInteractor(self.ic.axes, (x0, y0), radius)
        fraction = self.CI.radius/self.CI.hwhm
        self.ic.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.ic.xlannotation = self.ic.axes.annotate(" $r/HWHM$ = {:.2f}".format(fraction),
                                              xy=(0.,-0.1), picker=5, xycoords='axes fraction')
        self.ic.draw_idle()
        self.CI.modSignal.connect(self.onModifiedAperture)
        self.press = None
        self.ptime = None
        self.dv = 0
        cid1 = self.ic.mpl_connect('scroll_event', self.onWheel)
        cid2 = self.ic.mpl_connect('motion_notify_event', self.onMotion)
        cid3 = self.ic.mpl_connect('button_press_event', self.onPress)
        cid4 = self.ic.mpl_connect('button_release_event', self.onRelease)
        
    def onpick(self, event):
        """React to onpick events."""
        print('picked event ', event.artist)
        if event.artist == self.ic.xlannotation:
            rnew = self.getDouble(self.CI.radius/self.CI.hwhm)
            if rnew is not None:
                print('rnew is ',rnew)
                # Update caption
                self.ic.xlannotation.remove()
                self.ic.xlannotation = self.ic.axes.annotate(" $r/HWHM$ = {:.2f}".format(rnew),
                                              xy=(0.,-0.1), picker=5, xycoords='axes fraction')
                self.CI.circle.radius = rnew * self.CI.hwhm
                self.CI.radius = rnew * self.CI.hwhm
                self.CI.updateMarkers()
                self.ic.draw_idle()
                self.onModifiedAperture('updated aperture')
                                
    def getDouble(self, v):
        value, okPressed = QInputDialog.getDouble(self, "r/HWHM", "r/HWHM", v, 0.1, 10, 3)
        if okPressed:
            return value
        else:
            return None
        
    def onWheel(self, event):
        eb = event.button
        self.zoomImage(eb)
        
    def zoomUp(self):
        self.zoomImage('up')
    
    def zoomDown(self):
        self.zoomImage('down')
        
    def zoomImage(self, eb):
        ic = self.ic
        curr_xlim = ic.axes.get_xlim()
        curr_ylim = ic.axes.get_ylim()
        curr_x0 = (curr_xlim[0]+curr_xlim[1])*0.5
        curr_y0 = (curr_ylim[0]+curr_ylim[1])*0.5
        if eb == 'up':
            factor=0.9
        elif eb == 'down':
            factor=1.1
        new_width = (curr_xlim[1]-curr_xlim[0])*factor*0.5
        new_height= (curr_ylim[1]-curr_ylim[0])*factor*0.5
        x = [curr_x0-new_width,curr_x0+new_width]
        y = [curr_y0-new_height,curr_y0+new_height]
        ic.axes.set_xlim(x)
        ic.axes.set_ylim(y)
        ic.fig.canvas.draw_idle()
        
    def onPress(self, event):
        import time
        if event.inaxes:
            self.press = event.xdata, event.ydata
            self.ptime = (event.x, event.y, time.time())
        else:
            self.press = None
            self.ptime = None
            return
        
    def onRelease(self, event):
        print('mouse released')
        self.press = None
        self.ptime = None
        self.dv = 0
        
    def onMotion(self, event):
        import time
        if self.press is None:
            return
        if event.inaxes:
            dx = event.xdata - self.press[0]
            dy = event.ydata - self.press[1]
            d = np.hypot(self.ptime[0]-event.x, self.ptime[1]-event.y)
            dt = time.time() - self.ptime[2]
            self.ptime = (event.x, event.y, time.time())
            self.dv = d/dt
            print('d, dt', d, dt, self.dv)
        else:
            return
        if event.button == 2:
            # Pan using the mouse middle button
            ic = self.ic
            xlim = np.asarray(ic.axes.get_xlim())
            ylim = np.asarray(ic.axes.get_ylim())
            x = xlim - dx
            y = ylim - dy
            ic.axes.set_xlim(x)
            ic.axes.set_ylim(y)
            ic.fig.canvas.draw_idle()
        
            
    def initializeSpectrum(self):
        """Display the spectrum and cloud of flux points at location in image."""
        # Communicate new spectrum to spectral window
        self.onModifiedAperture('initial aperture')
        
    def onModifiedAperture(self, event):
        """Reacts to change in position and size of aperture."""
 
        s = self.specCube
        sc = self.specCloud
        radius = self.CI.circle.radius
        if event != 'segment modified':
            if self.CI.radius != radius:
                self.CI.radius = radius
                fraction = self.CI.radius/self.CI.hwhm
                self.ic.xlannotation.remove()
                self.ic.xlannotation = self.ic.axes.annotate(" $r/HWHM$ = {:.2f}".format(fraction),
                                              xy=(0.,-0.1), picker=5, xycoords='axes fraction')
            xc, yc = self.CI.circle.center
            print('dv ',self.dv)
            if self.dv < 20.:  # to check if we have to adapt
                # Select points inside 
                x0, y0 = self.ic.wcs.wcs_world2pix(sc.x, sc.y, 0)
                distance = np.hypot(x0 - xc, y0 - yc)   # distance in pixels
                idx = distance <= radius
                dists = distance[idx]
                w = sc.w[idx]
                f = sc.f[idx] 
                fl = sc.flight[idx]
                # Choose closest grid point for specCube
                pdistance = np.hypot(s.points[:,0] - xc, s.points[:,1] - yc)
                imin = np.argmin(pdistance)
                flux = s.flux[:, s.points[imin,1], s.points[imin,0]]
                eflux = s.eflux[:, s.points[imin,1], s.points[imin,0]]
                self.sc.spectrum = Spectrum(s.wave, flux, eflux, w, f, fl, dists, s.wt, 
                                            s.at, s.radius / self.ic.pixscale)
                self.sc.spectrum.set_colors()
        if event == 'segment modified':
            # check if the segment has shifted
            c = self.SI.center
            n0 = np.argmin(np.abs(s.wave-c))
            if n0 != s.n0:
                s.n0 = n0
                image = s.flux[s.n0,:,:]
                self.ic.updateImage(image)
                
        if event == 'initial aperture':
            print('Initialize length')
            medw = np.nanmedian(self.sc.spectrum.wave)
            print('Middle wavelength', medw)
            self.length = medw/s.R # spectral resolution in wavelength (FWHM = 2.355 * sigma)
            print('window size in wavelength ', self.length)
            self.sc.spectrum.set_filter(self.length, radius, s.pixscale)
            medf = np.nanpercentile(self.sc.spectrum.f, 10)
            print('Height ', medf)
            self.sc.drawSpectrum()
            self.SI = SegmentInteractor(self.sc.ax1, (medw, self.sc.spectrum.baseline-self.sc.spectrum.m1*5), 
                                        self.length, color='Blue')
            self.SI.modSignal.connect(self.onModifiedAperture)            
            self.sc.fig.canvas.mpl_connect('pick_event', self.onpick2)
            self.sc.xlannotation = self.sc.ax3.annotate(" $d/HWHM$ = {:.2f}".format(self.SI.delta/self.SI.fwhm),
                                              xy=(0.,-0.5), picker=5, xycoords='axes fraction')                        
        else:
            if self.dv < 2:
                self.sc.spectrum.set_filter(self.SI.delta, radius, s.pixscale)
                self.sc.drawSpectrum()
                x, y = zip(*self.SI.xy)
                siy = self.sc.spectrum.baseline-self.sc.spectrum.m1*5
                self.SI.xy = [(x_, siy) for x_ in x]
                self.SI.updateLinesMarkers()
                # Update caption
                self.sc.xlannotation.remove()
                self.sc.xlannotation = self.sc.ax3.annotate(" $d/HWHM$ = {:.2f}".format(self.SI.delta/self.SI.fwhm),
                                                            xy=(0.,-0.5), picker=5, xycoords='axes fraction')
            

    def onpick2(self, event):
        """React to onpick events."""
        print('picked event ', event.artist)
        if event.artist == self.sc.xlannotation:
            rnew = self.getDouble(self.SI.delta/self.SI.fwhm)
            if rnew is not None:
                print('rnew is ',rnew)
                self.SI.delta  = self.SI.fwhm * rnew
                self.onModifiedAperture('segment_modified')
                
                
    def newContSubtracted(self, event):
        """Compute and save cube continuum subtracted for each flight"""
        from astropy.io import fits
        from fifipy.cubik.data import computeNoise
        from dask import delayed, compute        

        aperture = self.CI.circle
        s = self.specCube
        sc = self.specCloud
        radius = aperture.radius
        nz, ny, nx = np.shape(s.eflux)
        uncertainty = np.empty((nz, ny, nx))
        newflux = np.empty((nz, ny, nx))
        idx = np.isnan(s.eflux)
        uncertainty[idx] = np.nan
        newflux[idx] = np.nan
        xyerror = np.nanmedian(s.eflux, axis=0)
        idy, idx = np.where(np.isfinite(xyerror))
        # Compute the uncertainty for all the spatial pixels in the list
        x0, y0 = self.ic.wcs.wcs_world2pix(sc.x, sc.y, 0)  
        areafactor = (s.pixscale/radius)**2/np.pi
        
        # Copy WXY file
        infile = os.path.join(self.pathFile, self.WXYfile)
        outfile = os.path.join(self.pathFile, 'WXY_cs_cubik.fits')
        if sys.platform in ['linux', 'darwin']:
            os.popen('cp -f '+infile+' '+outfile)
        else:
            os.popen('copy '+infile+' '+outfile)
        
        pixels = [delayed(computeNoise)(s.wave, sc.w, sc.f, sc.flight, self.SI.delta, x0, 
                                        y0, radius, (idx[i],idy[i]), contSub=True) 
                  for i in range(len(idx))]
        print('Starting the computation of uncertainty for ',len(idx),' points')
        ifluxnoise = compute(* pixels, scheduler='processes')

        print('Storing data')        
        for i, j, fluxnoise  in zip(idx, idy, ifluxnoise):
            newflux[:,j,i], uncertainty[:,j,i] = fluxnoise

        # Update new WXY file with flux and uncertainty
        with fits.open(outfile, mode='update') as hdl:
            hdl['FLUX'].data = newflux
            hdl['ERROR'].data = uncertainty
        

    def newUncertainty(self, event):
        """ Compute and save new uncertainty for the WXY file. """
        from astropy.io import fits
        from fifipy.cubik.data import computeNoise
        from dask import delayed, compute        

        aperture = self.CI.circle
        s = self.specCube
        sc = self.specCloud
        radius = aperture.radius
        nz, ny, nx = np.shape(s.eflux)
        uncertainty = np.empty((nz, ny, nx))
        newflux = np.empty((nz, ny, nx))
        idx = np.isnan(s.eflux)
        uncertainty[idx] = np.nan
        newflux[idx] = np.nan
        xyerror = np.nanmedian(s.eflux, axis=0)
        idy, idx = np.where(np.isfinite(xyerror))
        # Compute the uncertainty for all the spatial pixels in the list
        x0, y0 = self.ic.wcs.wcs_world2pix(sc.x, sc.y, 0)  
        areafactor = (s.pixscale/radius)**2/np.pi
        
        # Copy WXY file
        infile = os.path.join(self.pathFile, self.WXYfile)
        outfile = os.path.join(self.pathFile, 'WXY_cubik.fits')
        outname = os.path.join(self.pathFile, 'cubik.fits')
        if sys.platform in ['linux', 'darwin']:
            os.popen('cp -f '+infile+' '+outfile)
        else:
            os.popen('copy '+infile+' '+outfile)
        
        pixels = [delayed(computeNoise)(s.wave, sc.w, sc.f, sc.flight, self.SI.delta, x0, 
                                        y0, radius, (idx[i],idy[i])) 
                  for i in range(len(idx))]
        print('Starting the computation of uncertainty for ',len(idx),' points')
        ifluxnoise = compute(* pixels, scheduler='processes')

        #ifluxnoise = []
        #for i in range(len(idx)):
        #    ifluxnoise.append(computeNoise(s.wave, sc.w, sc.f, self.SI.delta, x0, 
        #                                y0, radius, areafactor, (idx[i],idy[i])))

        print('Storing data')        
        for i, j, fluxnoise  in zip(idx, idy, ifluxnoise):
            newflux[:,j,i], uncertainty[:,j,i] = fluxnoise
        
        # Save the computed noise
        from astropy.io import fits
        hdu = fits.PrimaryHDU()
        hdu1 = fits.ImageHDU()
        hdu1.data = newflux
        hdu1.header['EXTNAME'] = 'Flux'
        hdu2 = fits.ImageHDU()
        hdu2.data = uncertainty
        hdu2.header['EXTNAME'] = 'Uncertainty'
        hdul = fits.HDUList([hdu, hdu1, hdu2])
        hdul.writeto(outname, overwrite=True)
        hdul.close()
        print('Data saved in ', outfile)
        # Update new WXY file with computed uncertainty
        with fits.open(outfile, mode='update') as hdl:
            hdl['FLUX'].data = newflux
            hdl['ERROR'].data = uncertainty

        
def main():
    
    app = QApplication(sys.argv)
    gui = GUI()    
    # Adjust geometry to size of the screen
    screen_resolution = app.desktop().screenGeometry()
    width = screen_resolution.width()
    gui.setGeometry(width*0.025, 0, width*0.95, width*0.5)
    gui.hsplitter.setSizes ([width*0.40,width*0.48])
    # Add an icon for the application
    #app.setWindowIcon(QIcon(os.path.join(gui.path0,'icons','sospex.png')))
    app.setApplicationName('CUBIK')
    #app.setApplicationVersion(__version__)
    sys.exit(app.exec_())

    
