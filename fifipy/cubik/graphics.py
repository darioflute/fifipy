import numpy as np
# Matplotlib
import matplotlib
#import os
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# Matplotlib parameters
#import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.artist import Artist
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams['font.family']='STIXGeneral'
rcParams['font.size']=13
rcParams['mathtext.fontset']='stix'
rcParams['legend.numpoints']=1

# Astropy
from astropy.wcs.utils import proj_plane_pixel_scales as pixscales
from astropy.coordinates import SkyCoord
from astropy import units as u


from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QSize, Qt, pyqtSignal, QObject

class CircleInteractor(QObject):

    epsilon = 5
    showverts = True
    mySignal = pyqtSignal(str)
    modSignal = pyqtSignal(str)
   
    def __init__(self, ax, center, radius):
        super().__init__()
        from matplotlib.patches import Circle
        from matplotlib.lines import Line2D
        # from matplotlib.artist import Artist
        # To avoid crashing with maximum recursion depth exceeded
        import sys
        sys.setrecursionlimit(10000) # 10000 is 10x the default value

        self.ax = ax
        self.circle = Circle(center, radius, edgecolor='Lime',
                             facecolor='none', fill=False, animated=True)
        self.ax.add_patch(self.circle)
        self.canvas = self.circle.figure.canvas
        print('circle added')

        # Create a line with center, width, and height points
        self.center = self.circle.center
        self.radius = self.circle.radius

        x0, y0 = self.center
        r0 = self.radius

        x = [x0, x0 + r0, x0]
        y = [y0, y0, y0 + r0]
        self.xy = [(i,j) for (i,j) in zip(x,y)]
        self.line = Line2D(x, y, marker='o', linestyle=None, linewidth=0., markerfacecolor='g', animated=True)
        self.ax.add_line(self.line)
        print('line added')

        self.cid = self.circle.add_callback(self.circle_changed)
        self._ind = None  # the active point

        self.connect()
        self.press = None
        self.lock = None
  
    def connect(self):
        self.cid_draw = self.canvas.mpl_connect('draw_event', self.draw_callback)
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.cid_key = self.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.canvas.draw_idle()
  
    def disconnect(self):
        self.canvas.mpl_disconnect(self.cid_draw)
        self.canvas.mpl_disconnect(self.cid_press)
        self.canvas.mpl_disconnect(self.cid_release)
        self.canvas.mpl_disconnect(self.cid_motion)
        self.canvas.mpl_disconnect(self.cid_key)
        self.circle.remove()
        self.line.remove()
        self.canvas.draw_idle()
        self.aperture = None
        
    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.circle)
        self.ax.draw_artist(self.line)

    def circle_changed(self, circle):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, circle)
        self.line.set_visible(vis)  
  
    def get_ind_under_point(self, event):
        'get the index of the point if within epsilon tolerance'
        x, y = zip(*self.xy)
        d = np.hypot(x - event.xdata, y - event.ydata)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]
        if d[ind] >= self.epsilon:
            ind = None
        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)
        x0, y0 = self.circle.center
        r0 = self.circle.radius
        self.press = x0, y0, r0, event.xdata, event.ydata
        self.xy0 = self.xy
        self.lock = "pressed"

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == 'd':
            self.mySignal.emit('circle deleted')
        self.canvas.draw_idle()

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None
        self.press = None
        self.lock = "released"
        self.background = None
        # To get other aperture redrawn
        self.canvas.draw_idle()

    def motion_notify_callback(self, event):
        'on mouse movement'
        #if not self.ellipse.contains(event): return
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x0, y0, r0, xpress, ypress = self.press
        self.dx = event.xdata - xpress
        self.dy = event.ydata - ypress
        self.update_circle()
        # Redraw ellipse and points
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.circle)
        self.ax.draw_artist(self.line)
        self.canvas.update()
        self.canvas.flush_events()
        # Notify callback
        self.modSignal.emit('circle modified')

    def update_circle(self):
        x0, y0, r0, xpress, ypress = self.press
        dx, dy = self.dx, self.dy        
        if self.lock == "pressed":
            if self._ind == 0:
                self.lock = "move"
            else:
                self.lock = "resizerotate"
        elif self.lock == "move":
            if x0+dx < 0:
                xn = x0
                dx = 0
            else:
                xn = x0+dx
            if y0+dy < 0:
                yn = y0
                dy = 0
            else:
                yn = y0+dy
            self.circle.center = xn,yn
            # update line
            self.xy = [(i+dx,j+dy) for (i,j) in self.xy0]
            # Redefine line
            self.line.set_data(zip(*self.xy))
        # otherwise rotate and resize
        elif self.lock == 'resizerotate':
           # Avoid to pass through the center            
            if self._ind == 1:
                r_ = r0+2*dx  if (r0+2*dx) > 0 else r0
            elif self._ind == 2:
                r_ = r0+2*dy  if (r0+2*dy) > 0 else r0
            # update ellipse
            self.circle.radius = r_
            # update points
            self.updateMarkers()

    def updateMarkers(self):
        # update points
        x0,y0  = self.circle.center
        r_ = self.circle.radius
        x = [x0, x0+r_, x0]
        y = [y0, y0, y0+r_]
        self.xy = [(i,j) for (i,j) in zip(x,y)]
        self.line.set_data(x,y)




class MplCanvas(FigureCanvas):
    """Basic matplotlib canvas."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,QSizePolicy.MinimumExpanding,QSizePolicy.MinimumExpanding)
        FigureCanvas.updateGeometry(self)
        self.compute_initial_figure()

    def sizeHint(self):
        w, h = self.get_width_height()
        return QSize(w, h)

    def minimumSizeHint(self):
        return QSize(5,5)
    
    def compute_initial_figure(self):
        pass

class ImageCanvas(MplCanvas):
    """Canvas with image and spaxel locations."""

    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        
    def compute_initial_figure(self, image=None, wcs=None, xy=None):
        print('compute initial image')
        if wcs is None:
            '''Initial definition'''
            pass
        else:
            self.wcs = wcs
            self.pixscale = pixscales(self.wcs)[0]*3600
            print('wcs ', self.wcs)
            self.axes = self.fig.add_axes([0.1,0.1,.8,.8], projection = self.wcs)
            self.axes.coords[0].set_major_formatter('hh:mm:ss')
            self.axes.grid(color='black', ls='dashed')
            self.axes.set_xlabel('R.A.')
            self.axes.set_ylabel('Dec')
            # Colorbar
            self.cbaxes = self.fig.add_axes([0.9,0.1,0.02,0.85])
            if image is not None:
                print('image is ', np.shape(image))
                self.showImage(image, xy)
            
            # Activate focus
            self.setFocusPolicy(Qt.ClickFocus)
            self.setFocus()
            
    def showImage(self, image, xy):
        self.image = self.axes.imshow(image, origin='lower', cmap='gist_yarg',
                                      interpolation='nearest')
        self.fig.colorbar(self.image, cax=self.cbaxes)  
        # Transform xy in coordinates and plot them
        #print('shape xy ', np.shape(xy))
        #rd = self.wcs.wcs_pix2world(xy, 0)
        #xy0 = self.wcs.wcs_world2pix(xy, 0)
        #print(xy0)
        #x0 = xy0[:,0]/self.pixscale
        #y0 = xy0[:,1]/self.pixscale
        self.axes.scatter(xy[:,0], xy[:,1], s=5, color='blue',transform=self.axes.get_transform('fk5'))
        # Cursor data format
        def format_coord(x,y):
            """ Redefine how to show the coordinates """
            pixel = np.array([[x, y]], np.float_)
            world = self.wcs.wcs_pix2world(pixel, 0)                    
            xx = world[0][0]
            yy = world[0][1]
            " Transform coordinates in string "
            radec = SkyCoord(xx*u.deg, yy*u.deg, frame='icrs')
            xx = radec.ra.to_string(u.hour,sep=':',precision=1)
            yy = radec.dec.to_string(sep=':',precision=0)
            return '{:s} {:s} ({:4.0f},{:4.0f})'.format(xx,yy,x,y)
        
        
        self.axes.format_coord = format_coord
        self.draw_idle()

class SpectrumCanvas(MplCanvas):
    """Canvas with spectrum and cloud of fluxes."""

    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        self.fig.set_edgecolor('none')
        gs = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[2, 2, 1])
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[2, 0])
        self.ax4 = self.ax3.twinx()
        self.ax5 = self.ax3.twinx()
        #self.ax1 = self.fig.add_axes([0.12,0.03,.8,.18])
        #self.ax2 = self.fig.add_axes([0.12,0.22,.8,.38])
        #self.ax3 = self.fig.add_axes([0.12,0.62,.8,.38])
        self.ax3.tick_params(axis='y', labelcolor='tab:red')
        self.ax4.tick_params(axis='y', labelcolor='tab:blue')
        self.ax5.tick_params(axis='y', labelcolor='tab:orange')
        self.fig.subplots_adjust(wspace=0, hspace=0)
   
    def compute_initial_spectrum(self, spectrum=None, xmin=None, xmax=None):
        if spectrum is None:
            ''' initial definition when spectrum not yet available '''
        else:
            # Spectrum
            self.spectrum = spectrum
            self.drawSpectrum()
            # Activate focus
            self.setFocusPolicy(Qt.ClickFocus)
            self.setFocus()

    def drawSpectrum(self):
        for ax in [self.ax1, self.ax2, self.ax3, self.ax5]:
            ax.clear()
            ax.grid(True, which='both')
            ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        s = self.spectrum
        nmedian = np.nanmedian(s.nflux)
        idx = s.nflux > (nmedian * 0.5)
        self.ax1.scatter(s.w, s.f,  s=2, color=s.colors)
        self.ax1.scatter(s.wrejected, s.frejected, s=4, color='red')
        for ax in [self.ax1, self.ax2]:
            ax.fill_between(s.wave[idx], s.fflux[idx]-s.noise[idx], s.fflux[idx]+s.noise[idx], color='green', alpha=0.2)
            ax.fill_between(s.wave[idx], s.flux[idx]-s.eflux[idx], s.flux[idx]+s.eflux[idx], color='blue', alpha=0.2)
            ax.plot(s.wave[idx], s.flux[idx], color='blue', label='Pipeline Cube')
            ax.plot(s.wave[idx], s.fflux[idx], color='green', label='Biweight filter')  
            ax.plot(s.wave[~idx], s.flux[~idx], color='blue', alpha=0.3)
            ax.plot(s.wave[~idx], s.fflux[~idx], color='green', alpha=0.3)  
        self.ax2.legend()
        ff = np.concatenate((s.flux[idx], s.fflux[idx]))
        if len(ff) > 10:
            minf = np.nanmin(ff)
            maxf = np.nanmax(ff)
            diff = 0.1 * (maxf - minf)
            self.ax2.set_ylim(minf - diff, maxf + diff)
        self.ax3.plot(s.wave, s.nflux, color='tab:red')
        self.ax3.plot(s.wave, s.nn, color='tab:pink')
        self.ax4.plot(s.wt, s.at, color='tab:blue')
        self.ax5.plot(s.wave, s.deltas, color='tab:orange')
        self.ax3.set_xlabel('Wavelength [$\\mu$m]')
        self.ax1.set_ylabel('Cloud')
        self.ax2.set_ylabel('Flux')
        self.ax3.set_ylabel('N', color='tab:red')
        self.ax4.set_ylabel('Atm', color='tab:blue')
        mw = np.nanmedian(s.wave)
        d  = s.delta
        self.ax3.plot([mw-d, mw+d], np.ones(2) * nmedian, color='green')
        d = np.nanmedian(s.deltas)
        self.ax3.plot([mw-d, mw+d], np.ones(2) * nmedian * 0.5, color='orange')
        self.draw_idle()