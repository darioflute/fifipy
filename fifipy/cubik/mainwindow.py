#!/usr/bin/env python
import os
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QSplitter,
                             QAction, QFileDialog, QVBoxLayout)
from PyQt5.QtCore import Qt
from fifipy.cubik.data import spectralCube, spectralCloud, Spectrum
from fifipy.cubik.graphics import ImageCanvas, SpectrumCanvas, CircleInteractor

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
        file.addAction(QAction("Quit",self,shortcut='Ctrl+q',triggered=self.fileQuit))
        file.addAction(QAction("Open cube",self,shortcut='Ctrl+n',triggered=self.newFile))
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
            self.pathFile, file = os.path.split(fileName[0])
            self.loadFiles(self.pathFile, fileName[0])
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
            self.specCloud = spectralCloud(path, self.specCube.pixscale)
        except:
            print('The spectral cube cannot be read')
            
    def initializeImage(self):
        """Display the image and location of cal spaxels."""
        s = self.specCube
        sc = self.specCloud
        image = np.nanmean(s.flux, axis=0)
        #xy = np.array([[x_, y_] for x_, y_ in zip(sc.x, sc.y)], dtype=np.float64)
        xy = np.column_stack((sc.x, sc.y))
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
        self.ic.draw_idle()
        self.CI.modSignal.connect(self.onModifiedAperture)
        
    def initializeSpectrum(self):
        """Display the spectrum and cloud of flux points at location in image."""
        # Communicate new spectrum to spectral window
        s = self.specCube
        medw = np.nanmedian(s.wave)
        self.delta = medw/s.R/2.355  # spectral resolution in wavelength (FWHM = 2.355 * sigma)
        print('window size in wavelength ', self.delta)
        self.onModifiedAperture('initial aperture')
        
    def onModifiedAperture(self, event):
        """Reacts to change in position and size of aperture."""
        aperture = self.CI.circle
        s = self.specCube
        sc = self.specCloud
        xc, yc = aperture.center
        radius = aperture.radius
        # Select points inside 
        x0, y0 = self.ic.wcs.wcs_world2pix(sc.x, sc.y, 0)
        distance = np.hypot(x0 - xc, y0 - yc)   # distance in pixels
        idx = distance <= radius
        dists = distance[idx]
        w = sc.w[idx]
        f = sc.f[idx]        
        # Choose closest grid point for specCube
        pdistance = np.hypot(s.points[:,0] - xc, s.points[:,1] - yc)
        imin = np.argmin(pdistance)
        flux = s.flux[:, s.points[imin,1], s.points[imin,0]]
        self.sc.spectrum = Spectrum(s.wave, flux, w, f, dists, s.wt, s.at)
        self.sc.spectrum.set_colors()
        #print('delta, radius ', self.delta, radius)
        self.sc.spectrum.set_filter(self.delta, radius)
        #print('updating spectrum ... ')
        self.sc.drawSpectrum()

        
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

    