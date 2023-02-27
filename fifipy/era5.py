def retrieveEra5(year,month,day,starthour,endhour,minlon,maxlon,minlat,maxlat,flight):
    import cdsapi    
    outfile = 'F'+str(flight)+'era5.nc'
    time = ['{0:02d}:00'.format(i) for i in range(starthour, endhour+1)]
    print('time ', time)
    print(year, month, day)
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': 'specific_humidity',
            'year': '{0:4d}'.format(year),
            'month': '{0:02d}'.format(month),
            'day': '{0:02d}'.format(day),
            'pressure_level': [
                '1', '2', '3',
                '5', '7', '10',
                '20', '30', '50',
                '70', '100', '125',
                '150', '175', '200',
                '225', '250'
            ],
            'time': time,
            'area': [
                maxlat+1, minlon-1, minlat-1, maxlon+1,
            ],
        },
        outfile)

def retrieveEra5O3(year,month,day,starthour,endhour,minlon,maxlon,minlat,maxlat,flight):
    import cdsapi    
    outfile = 'F'+str(flight)+'era5O3.nc'
    time = ['{0:02d}:00'.format(i) for i in range(starthour, endhour+1)]
    print('time ', time)
    print(year, month, day)
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': 'ozone mass mixing ratio',
            'year': '{0:4d}'.format(year),
            'month': '{0:02d}'.format(month),
            'day': '{0:02d}'.format(day),
            'pressure_level': [
                '1', '2', '3',
                '5', '7', '10',
                '20', '30', '50',
                '70', '100', '125',
                '150', '175', '200',
                '225', '250'
            ],
            'time': time,
            'area': [
                maxlat+1, minlon-1, minlat-1, maxlon+1,
            ],
        },
        outfile)

def pwvFlight(flight, path='.', retrieve=True):

    import matplotlib.pyplot as plt
    from astropy.io import fits
    from glob import glob as gb
    from astropy.time import Time
    import pandas as pd 
    import numpy as np
    import cartopy.crs as ccrs
    import xarray as xr
    import os

    #flight = 803
    #retrieve = True
    flightname = os.path.join(path, 'F'+str(flight)+'path.fits')
    with fits.open(flightname) as hdul:
        header=hdul[0].header
        start_time = header['Time_sta']
        end_time = header['Time_end']
        longitude = hdul['Longitude'].data
        latitude = hdul['Latitude'].data
        unixtime = hdul['Time'].data
        pressure_inHg = hdul['Pressure'].data
    
    year = Time(start_time).ymdhms.year
    month = Time(start_time).ymdhms.month
    day =  Time(start_time).ymdhms.day
    start_hour = Time(start_time).ymdhms.hour
    if start_hour == 23:
        day += 1
        start_hour = 0
    end_hour = Time(end_time).ymdhms.hour + 1
    #print('year ', year)
    #print('month ', month)
    #print('day ',day)
    print('hour ',start_hour, end_hour)
    pressure_hPa = 33.863886666667 * pressure_inHg
    print('max pressure ', np.nanmax(pressure_hPa))

    # Trick for New Zealand flights
    idx = longitude < 0
    longitude[idx] += 360

    min_lon, max_lon = np.nanmin(longitude), np.nanmax(longitude)
    min_lat, max_lat = np.nanmin(latitude), np.nanmax(latitude)
    print('longitude limits ', min_lon, max_lon)
    print('latitude limits ', min_lat, max_lat)

    # Retrieve ERA 5 data
    if retrieve == True:
        #print(year, month, day)
        retrieveEra5(year, month, day, start_hour, end_hour, min_lon, max_lon, min_lat, max_lat, flight)

    try:
        ds = xr.open_dataset(os.path.join(path, "F"+str(flight)+"era5.nc"))
        print(ds)
        fig,ax = plt.subplots(figsize=[12,5])
        ds['q'][2,2,:,:].plot()
        ax.plot(longitude,latitude)
        plt.show()
    except:
        print('There is no ERA 5 file')
    
# Create flight path [Add object name, object type, altitude]
def flightpath(path, flight, outpath='.'):
    from glob import glob as gb
    from astropy.io import fits
    from astropy.time import Time
    from astropy.table import Table
    import numpy as np
    import re
    import os
    
    files = sorted(gb(path+'*lw.fits'))
    bfiles = sorted(gb(path+'*sw.fits'))
    
    # Sort according to time
    timefile = np.array([os.path.basename(file)[6:12] for file in files])
    s = np.argsort(timefile)
    files = np.array(files)
    files = files[s]
    if len(bfiles) > 0:
        bfiles = bfiles[s]

    lon = []
    lat = []
    time = []
    date = []
    press = []
    altitude = []
    obj = []
    obstype = []
    za = []
    restwav = []
    brestwav = []

    obsdate_old = ''
    for i, file in enumerate(files):
        # Check if file is manual or has XX in the name
        if re.match(r".*_manual_.*", file) or re.match(r".*XX.*", file):
            pass
        else:
            try:
                with fits.open(file) as hdl:
                    header = hdl['PRIMARY'].header
                    lon.append(header['LON_STA'])
                    lat.append(header['LAT_STA'])
                    press.append(header['HIERARCH STATICAIRPRESS'])
                    obsdate=header['DATE-OBS']
                    obj.append(header['OBJECT'])
                    obstype.append(header['OBSTYPE'])
                    altitude.append(header['HIERARCH BAROALTITUDE'])
                    za.append(header['ZA_START'])
                    restwav.append(header['RESTWAV'])
                    if obsdate[0:4] == '1970':
                        print(file ,' has year 1970')
                        if obsdate_old != '':
                            seconds = int(obsdate_old[-2:])
                            if seconds < 30:
                                seconds += 30
                            else:
                                seconds = 59
                            obsdate = '{0:s}{1:02d}'.format(obsdate_old[:-2], seconds)
                        else:
                            print('impossible to repair !')
                    date.append(obsdate)
                    unixt = Time(obsdate).unix
                    time.append(unixt)
                    obsdate_old = obsdate
                if len(bfiles) > 0:
                    with fits.open(bfiles[i]) as hdl:
                        header = hdl['PRIMARY'].header
                        brestwav.append(header['RESTWAV'])
            except:
                print(file, ' is corrupted')
        
    lon = np.array(lon)
    lat = np.array(lat)
    time = np.array(time)
    press = np.array(press)
    date = np.array(date)
    obj = np.array(obj)
    obstype = np.array(obstype)
    altitude = np.array(altitude)
    za = np.array(za)
    restwav = np.array(restwav)
    brestwav = np.array(brestwav)
    
    # Sort with unixt
    s = np.argsort(time)
    lon = lon[s]
    lat = lat[s]
    time = time[s]
    press = press[s]
    date = date[s]
    obj = obj[s]
    obstype = obstype[s]
    altitude = altitude[s]
    za = za[s]
    if len(bfiles) > 0:
        brestwav = brestwav[s]
    
    # Save in a fits file
    outname = outpath+'/F'+str(flight)+'path.fits'
    hdu = fits.PrimaryHDU()
    hdu.header['Flight'] = flight
    hdu.header['Time_sta'] = date[0]
    hdu.header['Time_end'] = date[-1]
    hdu1 = fits.ImageHDU()
    hdu1.data = lon
    hdu1.header['EXTNAME'] = 'Longitude'
    hdu2 = fits.ImageHDU()
    hdu2.data = lat
    hdu2.header['EXTNAME'] = 'Latitude'
    hdu3 = fits.ImageHDU()
    hdu3.data = press
    hdu3.header['EXTNAME'] = 'Pressure'
    hdu4 = fits.ImageHDU()
    hdu4.data = time
    hdu4.header['EXTNAME'] = 'Time'
    col1 = fits.Column(name='Object', format='20A', array=obj)
    hdu5 = fits.BinTableHDU.from_columns([col1], name='Object')
    col1 = fits.Column(name='Obstype', format='10A', array=obj)
    hdu6 = fits.BinTableHDU.from_columns([col1], name='Obstype')
    hdu7 = fits.ImageHDU()
    hdu7.data = altitude
    hdu7.header['EXTNAME'] = 'Altitude'
    hdu8 = fits.ImageHDU()
    hdu8.data = za
    hdu8.header['EXTNAME'] = 'ZenithAngle'
    hdu9 = fits.ImageHDU()
    hdu9.data = restwav
    hdu9.header['RESTWAV'] = 'RedRestWavelength'
    if len(bfiles) > 0:
        hdu10 = fits.ImageHDU()
        hdu10.data = brestwav
        hdu10.header['RESTWAV'] = 'RedRestWavelength'
        hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7, hdu8, hdu9, hdu10])
    else:
        hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7, hdu8, hdu9])
    hdul.writeto(outname, overwrite=True)
    hdul.close()


def addPWV(flight, path='.', altitude=None):
    from astropy.io import fits
    from astropy.time import Time
    import numpy as np
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator
    import os

    # Import path file
    flightname = os.path.join(path, 'F'+str(flight)+'path.fits')
    with fits.open(flightname) as hdul:
        header=hdul[0].header
        start_time = header['Time_sta']
        end_time = header['Time_end']
        longitude = hdul['Longitude'].data
        latitude = hdul['Latitude'].data
        unixtime = hdul['Time'].data
        pressure_inHg = hdul['Pressure'].data

    # Trick for New Zealand flights
    idx = longitude < 0
    longitude[idx] += 360

    # Import ERA5 file
    ds = xr.open_dataset(os.path.join(path,"F"+str(flight)+"era5.nc"))

    # For each frame, interpolate with lon, lat, time and assign q(t, press)
    lon = np.array(ds['q'].longitude.values)
    lat = np.array(ds['q'].latitude.values)
    time = ds['q'].time.values
    utime = np.array([Time(t).unix for t in time])
    idok = unixtime >= np.nanmin(utime)
    nzero = np.sum(~idok)

    levels = ds['q'].level.values
    nflight = len(longitude[idok])
    nlevels = len(levels)
    qt = np.zeros((nlevels, nflight))

    lat = np.flip(lat)
    pts = [[ut,la,lo] for ut, la, lo in zip(unixtime[idok],latitude[idok],longitude[idok])]
    pts = np.array(pts)

    for i in range(nlevels):
        V = np.array(ds['q'][:,i,:,:].values)
        V = np.flip(V, 1)
        fn = RegularGridInterpolator((utime, lat, lon), V)    
        qt[i, :] = fn(pts)

    # Compute PWV
    ntime = len(unixtime)
    ntimeok = len(unixtime[idok])
    pwv = np.zeros(ntime)
    g = 9.81 # m/s2
    
    data = None
    if altitude is None:
        pressure_hPa = 33.863886666667 * pressure_inHg[idok]
    else:
        # transform altitude (ft) in pressure (inHg)
        path0, file0 = os.path.split(__file__)
        data = np.loadtxt(os.path.join(path0, data, 'StdAtmATRAN.txt'))
        altitude_atran = data[:,0]
        temperature_atran = data[:,1]
        pressure_atran = data[:,2]
        altitude_atran *= 3280.84
        pressure_atran *= 29.9213
        pressure_hPa = 33.863886666667 * np.interp(altitude[idok], altitude_atran[idok], pressure_atran[idok])
        

    for i in range(ntimeok):
        qi = qt[:,i]
        # linear interpolation of qi at pressure of flight
        qinterp = np.interp(pressure_hPa[i], levels, qi)
        idx = levels < pressure_hPa[i]
        p = np.concatenate((levels[idx],[pressure_hPa[i]]))
        humidity = np.concatenate((qi[idx], [qinterp]))
        pwv[i+nzero] = np.trapz(humidity, x=p*100)/g * 1000

    #print(pwv)
    #Add PWV from satellite to the path file
    flightname = os.path.join(path, 'F'+str(flight)+'path.fits')
    with fits.open(flightname, mode='update') as hdul:
        if altitude is None:
            hdu = fits.ImageHDU()
            hdu.data = pwv
            hdu.header['EXTNAME'] = 'PWVera5'
            hdul.append(hdu)
        else:
            hdu = fits.ImageHDU()
            hdu.data = pwv
            hdu.header['EXTNAME'] = 'PWVera5planned'
            hdul.append(hdu)            
        hdul.flush()


def addObsWVZ(obsfile, flight, path='.'):
    """
    Add WVZ_OBS keyword for file starting from path file
    """
    from astropy.io import fits
    from astropy.time import Time
    import os
    import numpy as np

    flightname = os.path.join(path, 'F'+str(flight)+'path.fits')
    with fits.open(flightname) as hdul:
        header=hdul[0].header
        unixtime = hdul['Time'].data
        pwi = hdul['PWVera5'].data

    # Scale factor - to 0.6 for pre-telluric observations
    try:
        alpha = header['ALPHA']
    except:
        alpha = 0.6

    # Read file
    with fits.open(obsfile, mode='update') as hdl:
        header = hdl['PRIMARY'].header
        obsdate=header['DATE-OBS']
        if obsdate[0:4] == '1970':
            print(obsfile ,' has year 1970')
            print('Please repair !')
        else:
            unixt = Time(obsdate).unix
            # Find the closest time
            diff = unixt - unixtime
            indmin = np.argmin(np.abs(diff))
            WV = pwi[indmin] * alpha
            print('WVZ: ', WV)
            header['WVZ_OBS'] = WV
            hdul.flush()
            
