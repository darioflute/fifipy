# -*- coding: utf-8 -*-

def atm_layer_setup(alt_ft, water_vapor = 0, layers = 2, temp_base = 0,
                    azimuth = 0, o3dobson=320):
    """atm_layer_setup(alt_ft, water_vapor = 0,
                       layers = 2, temp_base = 0,
                       azimuth = 0,
                       o3dobson=320
                       )
    
    Derived from ATRAN technical report listing
    Lord, S.D. 'A New Software Tool for Computing Earth's Atmospheric
    Transmission of Near- and Far-Infrared Radiation', 
    NASA Technical Memorandum 103957, 1992
    Returns atmospheric layering setup for H2O, O3 and trace gases
       
    Parameters
    ----------
    alt_ft : number
        Altitude of base in feet
    
    water_vapor : number
        Zenith water vapor in microns if known (i.e. different from standard
        atmosphere)
    
    layers : number
        Number of layers that shall be used for the atmosphere setup
        Minimum is 1, Maximum is 300
        1 leads to single layer approximation see ATRAN report
    
    temp_base : number
        Temperature at base altitude in K 
        Value 0 will use standard atmosphere values
        Values >0 Difference between temp_base and standard atmposphere
        temperature will be applied for all altitudes, shifting temperatures to
        bigger or smaller values
        Will not have an effect on single layer setups or on the top layer
    azimuth : number
        Azimuth angle
        A value of 0 is straight up
        A value of 90 is horizontal (not allowed)

   o3dobson: number
        O3 concentration in Dobson Units,
        typical value is 320 in the northern emisphere
        
    Returns
    -------
    press_all : array[3, layers] 
        (Partial) pressures in atm at layer center for
        0: H2O
        1: All
        2: O3
        
    temp_all : array[3, layers]
        Temperatures in K at layer center for
        0: H2O
        1: All
        2: O3       
    
    w_gases : array[layers, 7]
        Gas columns in molecules/cm^2 within each layer for
        0: H2O
        1: CO2
        2: O3
        3: N2O
        4: CO
        5: CH4
        6: O2
    """
    import numpy as np
    import os
    parts = np.zeros((7))
    parts[0] = 0. # H2O  Separately evalueted
    #parts[1] = 330e-6 # CO2  Should be ~450e-6 now ...
    parts[1] = 450e-6 
    parts[2] = 0. # O3   Separately evalueted
    parts[3] = 0.28e-6 # N2O  total (should be around 0.331e-6)
    #parts[3] = 0.1e-6 # N2O  in the stratosphere ?
    parts[4] = 0.075e-6 # CO
    #parts[5] = 1.6e-6 # CH4  1.895e-6 according to wikipedia in 2021
    parts[5] = 1.895e-6
    #parts[6] = 2.0946e-1 # O2  concentration in atmosphere
    parts[6] = 1.05e-1 # O2 is this more correct in the stratosphere ?
    
    # Ozone values from 39° latitude?
    # Northern emisphere ~ 320 Dobson units (1 DU = 2.687e16 molecules/cm2)
    # cdoz = 9.13e18 # molecules/cm^2
    # cdoz = 8.59e18 # new value (https://gml.noaa.gov/ozwv/dobson/papers/wmobro/ozone.html)
    cdoz = o3dobson * 2.687e16
    
    i_dist_O3 = 2 #is this the 43° distribution?

    path0, file0 = os.path.split(__file__)
    std_atm = np.loadtxt(os.path.join(path0,'data','StdAtmATRAN.txt'))
    atm_alt = std_atm[:, 0] # in km
    atm_temp = std_atm[:, 1] # in K
    atm_press = std_atm[:, 2] # in atm
    atm_c_H2O = std_atm[:, 3] # in molecules/cm^2
    atm_c_all = std_atm[:, 4] # in molecules/cm^2
    atm_temp_H2O = std_atm[:, 5] # in K
    atm_temp_mix = std_atm[:, 6] # in K
    atm_press_half_press_H2O = std_atm[:, 7] # in atm
    atm_press_half_press_mix = std_atm[:, 8] # in atm
    atm_c_O3 = std_atm[:,9:] #in molecules/cm^2 for 4 different latitudes
    
    i_H2O_top = 291  # Lines of the standard atmosphere file
    
    alt_km = alt_ft*0.0003048
    
    w_column = np.interp(alt_km, atm_alt, atm_c_H2O) # Water vapor at altitude
    w_c_mu = w_column * 2.9940e-19 # conv microns to molecules/cm^2
    w_c_mu_copy = w_c_mu
    
    # increase/decrease temperatures to match temp_base at alt_ft while keeping
    # form of profile constant
    if temp_base > 0:
        temp_shift = temp_base - np.interp(alt_km, atm_alt, atm_temp)
        atm_temp = atm_temp + temp_shift
        
    
    # shift water vapor and scale if necessary
    if water_vapor>0:
        target_vapor = water_vapor / 2.994E-19 # conv microns to molecules/cm^2
        alt_new = np.interp(-target_vapor, -atm_c_H2O, atm_alt)
        alt_diff = alt_new-alt_km # Find altitude with measured water vapor
        i_tenths = int(np.floor(alt_diff*10))
        
        # shifting if necessary
        if i_tenths != 0:
            i_ten = abs(i_tenths)
            
            if i_tenths < 0:
                for i in range(1, i_ten+1):
                    atm_c_H2O[-1] = atm_c_H2O[-1] + atm_c_H2O[-1-i]
                for i in range(290, 0, -1):
                    if i <= i_ten:
                        atm_c_H2O[i] = atm_c_H2O[0]
                    else:
                        atm_c_H2O[i] = atm_c_H2O[i - i_ten]
            if i_tenths > 0:
                for i in range(0, 292-i_ten):
                    atm_c_H2O[i] = atm_c_H2O[i+i_ten]
                i_H2O_top=291-i_ten
            w_column = np.interp(alt_km, atm_alt, atm_c_H2O)
            w_c_mu_copy=w_column*2.9940e-19
            if (abs(water_vapor-w_c_mu_copy) / w_c_mu_copy) > .2:
                print('Could not shift to that water vapor\n')
                return
        # scale if necessary
        scale = water_vapor / w_c_mu_copy
        atm_c_H2O[0:i_H2O_top+1] = scale * atm_c_H2O[0:i_H2O_top+1]
        
    if (layers < 1) or (layers > 300):
        print('Number of atmospheric layers out of limits 0<layers<=300')
    
    if (azimuth>=90) or (azimuth<0):
        print('Azimuth angle out of limits 0<=angle<90')
    
    xmu = np.cos(np.deg2rad(azimuth))
    
    column_O3_lat = np.array((6.86e18, 8.41e18, 1.03e19, 1.21e19))
    i_top = i_H2O_top
    i_alt_start = int(alt_km*10+1)
    
    column_O3 = np.zeros((292,1))
    
    column_O3 = atm_c_O3[:, i_dist_O3] * cdoz / column_O3_lat[i_dist_O3]

    press_all = np.zeros((3,layers))
    temp_all = np.zeros((3,layers))    

    # Multilayer atmosphere
    if layers!=1:
        # Values at layer bottom
        column_bottom = np.interp(alt_km, atm_alt, atm_c_all)
        column_bottom_H2O = np.interp(alt_km, atm_alt[:i_top+1],
                                      atm_c_H2O[:i_top+1]
                                      )
        column_bottom_O3 = np.interp(alt_km, atm_alt, column_O3)
        
        # Total column between bottom and top
        total_column = column_bottom - 2.930E23
        total_column_H2O = column_bottom_H2O - atm_c_H2O[i_top]
        total_column_O3 = column_bottom_O3 - column_O3[-1]
        
        # Layer center values
        i_divisions = (layers - 1) * 2
        delta_column = total_column / i_divisions
        delta_column_H2O = total_column_H2O / i_divisions
        delta_column_O3 = total_column_O3 / i_divisions
        
        level=0

        for i in range(1, i_divisions, 2):
            x_layer_column = column_bottom - i * delta_column
            x_layer_column_H2O = column_bottom_H2O - i * delta_column_H2O
            x_layer_column_O3 = column_bottom_O3 - i * delta_column_O3
            
            press_all[0, level] = np.interp(-x_layer_column_H2O,
                                            -atm_c_H2O[i_alt_start:i_top+1],
                                            atm_press[i_alt_start:i_top+1]
                                            )
            press_all[1, level] = np.interp(-x_layer_column,
                                            -atm_c_all[i_alt_start:],
                                            atm_press[i_alt_start:]
                                            )
            press_all[2, level] = np.interp(-x_layer_column_O3,
                                            -column_O3[i_alt_start:],
                                            atm_press[i_alt_start:]
                                            )
            temp_all[0, level] = np.interp(-x_layer_column_H2O,
                                           -atm_c_H2O[i_alt_start:i_top+1],
                                           atm_temp[i_alt_start:i_top+1]
                                           )
            temp_all[1, level] = np.interp(-x_layer_column,
                                           -atm_c_all[i_alt_start:],
                                           atm_temp[i_alt_start:]
                                           )
            temp_all[2, level] = np.interp(-x_layer_column_O3,
                                           -column_O3[i_alt_start:],
                                           atm_temp[i_alt_start:]
                                           )
            level = level + 1
        
        # column densities for the layers (same for lower layers only top 
        # layer is different)
        w_gases = np.zeros((2,7))
        # H2O
        w_gases[0,  0] = 2 * delta_column_H2O / xmu
        w_gases[1,0] = atm_c_H2O[i_top] / xmu
        # O3
        w_gases[0, 2] = 2 * delta_column_O3 / xmu
        w_gases[1, 2] = column_O3[-1] / xmu  
        # mixed gases
        for i in (1, 3, 4, 5, 6):
            w_gases[0, i] = 2 * delta_column / xmu * parts[i]
            w_gases[1, i] = atm_c_all[-1] / xmu * parts[i] 
        # top layer
        temp_all[0, -1] = atm_temp_H2O[i_top]
        temp_all[1, -1] = atm_temp_mix[-1]
        temp_all[2, -1] = atm_temp_mix[-1]
        press_all[0, -1] = atm_press_half_press_H2O[i_top]
        press_all[1, -1] = atm_press_half_press_mix[-1]
        press_all[2, -1] = atm_press_half_press_mix[-1]
    else:
        # Single layer atmosphere
        # H2O
        temp_all[0, 0] = np.interp(alt_km,
                                   atm_alt[:i_top+1],
                                   atm_temp_H2O[:i_top+1]
                                   )
        press_all[0, 0] = np.interp(alt_km,
                                    atm_alt[:i_top+1],
                                    atm_press_half_press_H2O[:i_top+1]
                                    )
        # mixed gases
        temp_all[1, 0] = np.interp(alt_km,
                                   atm_alt[:],
                                   atm_temp_mix[:]
                                   )
        press_all[1, 0] = np.interp(alt_km,
                                    atm_alt[:],
                                    atm_press_half_press_mix[:]
                                    ) 
        # O3, using 22km values below 22km, because most O3 above 22km
        if alt_km < 22:
            temp_all[2, 0] = np.interp(22,
                                        atm_alt[:],
                                        atm_temp[:]
                                        )
            press_all[2, 0] = np.interp(22,
                                        atm_alt[:],
                                        atm_press[:]
                                        )         
        else:
            temp_all[2, 0] = np.interp(alt_km,
                                       atm_alt[:],
                                       atm_temp[:]
                                       )
            press_all[2, 0] = np.interp(alt_km,
                                        atm_alt[:],
                                        atm_press[:]
                                        )
        w_gases = np.zeros((2, 7))
        # H2O
        w_gases[0, 0] = 1 / xmu * np.interp(alt_km,
                                           atm_alt[:i_top+1],
                                           atm_c_H2O[:i_top+1]
                                           )
        # O3
        w_gases[0, 2] = 1 / xmu * np.interp(alt_km,
                                           atm_alt[:],
                                           column_O3[:]
                                           )
        # Mixed gases
        for i in (1, 3, 4, 5, 6):
            w_gases[0,i] = parts[i] / xmu * np.interp(alt_km,
                                                      atm_alt[:],
                                                      atm_c_all[:]
                                                      )
    return press_all, temp_all, w_gases

def make_am_amc_file(press_all, temp_all, w_gases, lam_start, lam_end,
                     filename = 'ATRAN'
                     ):
    """Setup the layers of an atmosphere
    
    Derived from ATRAN technical report listing
    Self broadening is ignored by using largely oversized layers h=100km    
    
    Parameters
    ----------
    press_all : array[3, layers] 
        (Partial) pressures in atm at layer center for
        0: H2O
        1: All
        2: O3
        
    temp_all : array[3, layers]
        Temperatures in K at layer center for
        0: H2O
        1: All
        2: O3       
    
    w_gases : array[layers, 7]
        Gas columns in molecules/cm^2 within each layer for
        0: H2O
        1: CO2
        2: O3
        3: N2O
        4: CO
        5: CH4
        6: O2
    
    lam_start : number
        Start wavelength for simulation in microns
    
    lam_end : number
        End wavelength for simulation in microns
        
    filename : string
        name for the output file '.amc' will be appended
        
    Return
    ------
    filename : string
        base name for the output file without end
    """
    from scipy import constants
    
    lam_start = lam_start * 1e-6 # in m
    lam_end = lam_end * 1e-6 # in m
    
    frequ_start = constants.c / lam_end * 1e-12 # in THz
    frequ_end = constants.c / lam_start * 1e-12 # in THz
    frequ_step = (frequ_end + frequ_start) / 2 * 1e12 * 1e-6 * 1e-6
    # first 1e12 * 1e-6 to voncert to MHz, 1e-6 scales step size to get enough 
    # samples

    file_id = open(filename + '.amc','w')
    # file header info
    file_id.write('f {:e} THz '.format(frequ_start)
                  + '{:e} THz '.format(frequ_end)
                  + '{:e} MHz\n'.format(frequ_step)
                  )
    file_id.write('\n')
    file_id.write('output f GHz tx Tb\n')
    file_id.write('\n')
    file_id.write('T0 2.7 K\n')
    file_id.write('za 0 deg\n')
    file_id.write('tol 1e-4\n\n\n')
    # Layersetup
    # Top layer for mult-layers
    layers = press_all.shape[1]
    if layers > 1: #checks if more than one layer
        file_id.write('\nlayer H2O\n')
        file_id.write('P {:e} mbar\n'.format(1.01325e3*press_all[0,-1]))
        file_id.write('T {:e} K\n'.format(temp_all[0,-1]))
        file_id.write('h 100 km\n')
        file_id.write('lineshape Lorentz h2o_lines\n')
        file_id.write('column h2o_lines {:e} cm^-2\n'.format(w_gases[1,0]))

        file_id.write('\nlayer mixed gases\n')
        file_id.write('P {:e} mbar\n'.format(1.01325e3*press_all[1,-1]))
        file_id.write('T {:e} K\n'.format(temp_all[1,-1]))
        file_id.write('h 100 km\n')
        file_id.write('lineshape Lorentz co2 n2o '
                      +'co ch4 o2_uncoupled o2_coupled\n'
                      )
        file_id.write('column co2 {:e} cm^-2\n'.format(w_gases[1,1]))
        file_id.write('column n2o {:e} cm^-2\n'.format(w_gases[1,3]))
        file_id.write('column co {:e} cm^-2\n'.format(w_gases[1,4]))
        file_id.write('column ch4 {:e} cm^-2\n'.format(w_gases[1,5]))
        file_id.write('column o2_uncoupled {:e} cm^-2\n'.format(w_gases[1,6]))
        file_id.write('column o2_coupled {:e} cm^-2\n'.format(w_gases[1,6]))

        file_id.write('\nlayer O3\n');
        file_id.write('P {:e} mbar\n'.format(1.01325e3*press_all[2,-1]))
        file_id.write('T {:e} K\n'.format(temp_all[2,-1]))
        file_id.write('h 100 km\n')
        file_id.write('lineshape Lorentz o3\n')
        file_id.write('column o3 {:e} cm^-2\n'.format(w_gases[1,2]))   
    
    # Lower layers or single layer
    if layers == 1:
        layers = 2
        
    for i in range(layers-2, -1, -1):
        file_id.write('\nlayer H2O\n')
        file_id.write('P {:e} mbar\n'.format(1.01325e3*press_all[0,i]))
        file_id.write('T {:e} K\n'.format(temp_all[0,i]))
        file_id.write('h 100 km\n')
        file_id.write('lineshape Lorentz h2o_lines\n')
        file_id.write('column h2o_lines {:e} cm^-2\n'.format(w_gases[0,0]))

        file_id.write('\nlayer mixed gases\n')
        file_id.write('P {:e} mbar\n'.format(1.01325e3*press_all[1,i]))
        file_id.write('T {:e} K\n'.format(temp_all[1,i]))
        file_id.write('h 100 km\n')
        file_id.write('lineshape Lorentz '
                      +'co2 n2o co ch4 o2_uncoupled o2_coupled\n'
                      )
        file_id.write('column co2 {:e} cm^-2\n'.format(w_gases[0,1]))
        file_id.write('column n2o {:e} cm^-2\n'.format(w_gases[0,3]))
        file_id.write('column co {:e} cm^-2\n'.format(w_gases[0,4]))
        file_id.write('column ch4 {:e} cm^-2\n'.format(w_gases[0,5]))
        file_id.write('column o2_uncoupled {:e} cm^-2\n'.format(w_gases[0,6]))
        file_id.write('column o2_coupled {:e} cm^-2\n'.format(w_gases[0,6]))

        file_id.write('\nlayer O3\n')
        file_id.write('P {:e} mbar\n'.format(1.01325e3*press_all[2,i]))
        file_id.write('T {:e} K\n'.format(temp_all[2,i]))
        file_id.write('h 100 km\n')
        file_id.write('lineshape Lorentz o3\n')
        file_id.write('column o3 {:e} cm^-2\n'.format(w_gases[0,2]))
        
    file_id.close()
    return filename

def run_am(filename = 'ATRAN'):
    """Run am using the supplied base filename
    
    Parameters
    ----------    
    filename : string
        name for the output file '.amc' will be appended
        
    Returns
    -------
    filename : string
        just feeds through base filename
    """
    import os
    
    os.system('am ' + filename + '.amc >' 
              + filename + '.dat 2>' 
              + filename + '.aux'
              )
    return filename

def read_am_dat(filename = 'ATRAN'):
    """Read the am data file belonging to base filename
    
    Parameters
    ----------    
    filename : string
        name for the read file '.dat' will be appended
        
    Returns
    -------
    wavelength : array
        wavelengths at which transmission was sampled
    transmission: array
        transmission at wavelenth samples
    """
    import pandas as pd
    from scipy import constants
    
    df = pd.read_csv(filename+'.dat', delimiter = ' ', header = None)
    
    wavelength = constants.c / df.loc[:, 0].values * 1e-3 #frequency to microns
    transmission = df.loc[:, 1]
    
    return wavelength, transmission

def callAM(altitude, wv, w1, w2, o3dobson=320, adjust=True):
    """Analogue to callAtran
    
    za is currently not used
    res is currently not used
    """
    p_all, t_all, w_gases = atm_layer_setup(altitude, 
                                            water_vapor = wv,
                                            layers = 2,
                                            o3dobson=o3dobson
                                            )
    filename = make_am_amc_file(p_all, t_all, w_gases, w1, w2)
    filename = run_am(filename)
    wavelength, transmission = read_am_dat(filename)
    if adjust:
        trasmission = adjustLines(wavelength, transmission)

    return wavelength, transmission

def adjustLines(w,t):
    # Adjust blue lines (too deep)
    import numpy as np
    idx = (w > 49) & (w < 52.5)
    if np.sum(idx) > 0:
        t[idx] = 1-(1-t[idx])*0.6
    return t
        
def convolveAM(wave, trans, band):
    """ Convolution with FIFI-LS spectral resolution """
    import numpy as np
    from fifipy.spectra import getResolution

    w1 = np.nanmin(wave)
    w2 = np.nanmax(wave)

    w = w1
    wgrid = [w]
    while w < w2:
        dw = w / getResolution(band, w)
        w += dw/6.
        wgrid.append(w)

    # Convolve with spectral resolution
    t = []
    pi2 = np.sqrt(2 * np.pi)
    for w in wgrid:
        dw = w / getResolution(band, w)
        s = dw/2.355
        idx = (wave > (w-3*s)) & (wave < (w+3*s))  
        ww = wave[idx]
        tt = trans[idx]    
        gg = np.exp(-0.5*((ww-w)/s)**2)/pi2/s
        t.append(np.sum(gg*tt)/np.sum(gg))

    wgrid = np.array(wgrid)
    t = np.array(t)

    return wgrid, t
