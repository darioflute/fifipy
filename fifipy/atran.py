#!/usr/bin/env python
import urllib#,urllib2
import urllib.request
import urllib.parse
from html.parser import HTMLParser
import pandas as pd
import numpy as np
from io import StringIO
class MyHTMLParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.data = []
        self.values = []
        self.recording = 0

    def handle_starttag(self, tag, attrs):
        # Only parse the 'anchor' tag.
        if tag == "a":
           # Check the list of defined attributes.
           for name, value in attrs:
               # If href is defined, print it.
               if name == "href":
                    #print name, value
                    self.recording = 1
                    self.values.append(value)
                    
    def handle_endtag(self,tag):
        if tag == 'a':
            self.recording -=1 
            #print "Encountered the end of a %s tag" % tag 

    def handle_data(self, data):
        if self.recording:
            self.data.append(data)
            
            
def addExtension(data, extname, unit, hdr):
    from astropy.io import fits
    hdu = fits.ImageHDU()
    hdu.data = data
    hdu.header['EXTNAME']=(extname)
    if unit !=None: hdu.header['BUNIT']=(unit)
    if hdr != None: hdu.header.extend(hdr)
    return hdu


def callAtran(i,altitude,za,wv,w1,w2,res):
    url = 'https://atran.arc.nasa.gov/cgi-bin/atran/atran.cgi'
    post_params = { 
        'Altitude'  : altitude,
        'WVapor' : wv,
        'ZenithAngle': za,
        'WaveMin': w1,
        'WaveMax': w2,
        'Resolution': res,
        'NLayers': '2',
        'Obslat': '39 deg',
        'Submit Form': 'Submit Form'
    }
    post_args = urllib.parse.urlencode(post_params).encode("utf-8")

    # Send HTTP POST request
    request = urllib.request.Request(url, post_args)
    response = urllib.request.urlopen(request)
    html = response.read()
    
    parser = MyHTMLParser()
    parser.feed(html.decode('utf-8'))
    data= parser.data
    values= parser.values
    parser.close()
    idx = data.index('View output data file')
    # Retrieve file
    file = 'https://atran.arc.nasa.gov'+values[idx]
    response = urllib.request.urlopen(file)
    content= response.read()
    df = pd.read_csv(StringIO(content.decode('utf-8')), delimiter=' ',skipinitialspace=True,index_col=0,header=None)
    return i,df.loc[:,1].values, df.loc[:,2].values


def cleanAtran(wave,trans):
    # Removes non-existent absorption from Atran models
    from scipy.interpolate import interp1d
    mask = np.zeros(np.size(wave), dtype=bool)
    regions = [
                [73.19,73.235],
                [91.82,91.925],
                [120.50,120.75],
                [123.85,124.10],
                [147.4,147.65],
                [155.4,155.75],
                [160.68,160.8],
                [170.28,170.6],
                [176.42,176.7],
                ]   
    for r in regions:
        try:
            mask[:] = False
            m = (wave > r[0]) & (wave < r[1])
            mask[m] = True
            model = interp1d(wave[~mask],trans[~mask],kind='cubic',fill_value='extrapolate')
            trans = model(wave)
        except:
            pass

    # Better correction for feature at 147.4-147.65
    try:
        lines = [
                [147.225, 147.375],
                [147.66, 147.74]
                ]
        linefit = []
        mls = []
        for l in lines:
            mask[:] = 0
            ml = (wave > l[0]) & (wave < l[1])
            mask[ml] = True
            model = interp1d(wave[~mask],trans[~mask],kind='linear',fill_value='extrapolate')
            line = trans[ml] - model(wave[ml])
            trans[ml] -= line
            linefit.append(line)
            mls.append(ml)
        id0 = int(np.argmin(np.abs(wave- 146.92)))
        trans[id0:id0+360] = trans[id0-360:id0][::-1]
        # Add back lines
        for line, ml in zip(linefit, mls):
            trans[ml] += line
    except:
        pass

        

    # Little correction for a line (slide to the left)
    try:
        l = [144.3,144.34]
        mask[:] = 0
        ml = (wave > l[0]) & (wave < l[1])
        mask[ml] = True
        model = interp1d(wave[~mask],trans[~mask],kind='cubic',fill_value='extrapolate')
        line = trans[ml] - model(wave[ml])
        trans[ml] -= line
        r= [144.12,144.5]
        idx = np.where( (wave > r[0]) & (wave < r[1]))
        idx = (np.array(idx))[0]
        nidx = int(np.size(idx)*0.04/(r[1]-r[0]))
        idxn = idx-nidx
        trans[idxn] = trans[idx]
        trans[ml] += line
    except:
        pass
    
    # Remove absorption by duplication
    regions = [
            [188,188.30],
            ]
    for r in regions:
        try:
            idx = np.where( (wave > r[0]) & (wave < r[1]))
            idx = (np.array(idx))[0]
            idxp = np.arange(-1,-np.size(idx),-1)+idx[0]+np.size(idx)*2+1
            trans[idxp]=trans[idx[:-1]]
        except:
            pass
        
    # remove feature adding back line on the feature
    lines = [
            [[164.91,164.96]],
            [[195.62,195.73]],
            [[96.73,96.775]],
            [[102.36,102.38]],
            [[111.06,111.09]],
            [[115.32,115.40]],
            [[139.29,139.325]],
            [[143.00,143.085]],
            [[151.025,151.065]],
            [[87.44,87.46],
             [87.35,87.37]],
            [[118.285,118.310],
             [118.385,118.44]],
            [[123.55,123.62]]
            ]
    bad = [
            [164.80,165.20],
            [195.55,196.1],
            [96.725,96.875],
            [102.23,102.45],
            [110.99,111.1],
            [115.30,115.60],
            [139.20,139.50],
            [142.9,143.30],
            [150.9,151.3],
            [87.35,87.47],
            [118.25,118.47],
            [123.2,123.7]
            ]

    for l_, b in zip(lines, bad):
        try:
            linefit = []
            mls = []
            for l in l_:
                mask[:] = 0
                ml = (wave > l[0]) & (wave < l[1])
                mask[ml] = True
                model = interp1d(wave[~mask],trans[~mask],kind='linear',fill_value='extrapolate')
                line = trans[ml] - model(wave[ml])
                trans[ml] -= line
                linefit.append(line)
                mls.append(ml)
            mb = (wave > b[0]) & (wave < b[1])
            mask[:] = False
            mask[mb] = True
            model = interp1d(wave[~mask],trans[~mask],kind='linear',fill_value='extrapolate')
            trans[mb] = model(wave[mb])
            # Add back lines
            for line, ml in zip(linefit, mls):
                trans[ml] += line
        except:
            pass
    return trans
    # Add emission feature
    #regions = [
    #    [139.16,139.5]
    #]
    #for r in regions:
    #    m = (wave > r[0]) & (wave < r[1])
    #    mask[:] = False
    #    mask[m] = True
    #    model = interp1d(wave[~mask],trans[~mask],kind='cubic',fill_value='extrapolate')
    #    diff =  model(wave) - trans
    #    trans[m] += 2*diff[m]
        

def getATransBand(band, altitude, za, wv):
    from fifipy.spectra import getResolution
    # Get ATRAN model  
    res = 0
    i = 0
    if band == 'B2':
        w1 = 45
        w2 = 80
        i,wave,trans = callAtran(i,altitude,za,wv,w1,w2,res)
    elif band == 'B1':
        w1 = 60
        w2 = 100
        i,wave1,trans1 = callAtran(i,altitude,za,wv,w1,w2,res)
        w1 = 100
        w2 = 130
        i,wave2,trans2 = callAtran(i,altitude,za,wv,w1,w2,res)
        wave = np.concatenate([wave1, wave2])
        trans = np.concatenate([trans1, trans2])
    elif band == 'R':
        w1 = 100
        w2 = 155
        i,wave1,trans1 = callAtran(i,altitude,za,wv,w1,w2,res)
        w1 = 155
        w2 = 210
        i,wave2,trans2 = callAtran(i,altitude,za,wv,w1,w2,res)
        wave = np.concatenate([wave1, wave2])
        trans = np.concatenate([trans1, trans2])

    # clean ATRAN model
    trans = cleanAtran(wave, trans)
    
    # Define grid
    if band == 'B2':
        w1 = 50
        w2 = 75
    elif band == 'B1':
        w1 = 67
        w2 = 125
    elif band == 'R':
        w1 = 105
        w2 = 205

    w = w1
    wgrid = [w]    
    while w < w2:
        dw = w / getResolution(band, w)
        w += dw/3.
        wgrid.append(w)

    # Convolve with spectral resolution
    t = []
    pi2 = np.sqrt(2 * np.pi)
    for w in wgrid:
        dw = w / getResolution(band, w)
        s = dw/2.355
        idx = (wave > (w-3*dw)) & (wave < (w+3*dw))  
        ww = wave[idx]
        tt = trans[idx]    
        gg = np.exp(-0.5*((ww-w)/s)**2)/pi2/s
        t.append(np.sum(gg*tt)/np.sum(gg))

    wgrid = np.array(wgrid)
    t = np.array(t)

    return wgrid, t

def getATransBlue2Old(altitude,za,wv,interp=False):
    from scipy.interpolate import interp1d
    from fifipy.spectra import getResolution
    
    #w1 =  [ 50,  55,  60,  65,  70]
    #w2 =  [ 55,  60,  65,  70,  75]
    #res = [940,1060,1270,1590,1980]
    w1 = np.arange(50,75,1.0)
    w2 = list(w1.copy())
    w2.append(75)
    w2 = w2[1:]
    w1 = np.array(w1)
    w2 = np.array(w2)
    res = getResolution('B2', np.sqrt(w1*w2))

    # t1 = timer()
    # pool = mp.Pool(len(w1))
    # res  = [pool.apply_async(callAtran, args=(i,altitude,za,wv,w1[i],w2[i],res[i])) for i in range(len(w1))]
    # results = [p.get() for p in res]
    # results.sort()
    # pool.terminate()
    # t2 = timer()

    # #print ("time elapsed: ",t2-t1)
    
    # wave = []
    # trans = []
    # for r in results:
    #     wave.append(r[1])
    #     trans.append(r[2])
    wave=[]
    trans=[]

    for w1i, w2i, resi in zip(w1,w2,res):
        try:
            result = callAtran(0,altitude,za,wv,w1i,w2i,resi)
        except:
            print('2nd try ...')
            result = callAtran(0,altitude,za,wv,w1i,w2i,resi)
        wave.append(result[1])
        trans.append(result[2])

        
    wave = np.concatenate(wave)
    trans = np.concatenate(trans)
    u, idx = np.unique(wave, return_index=True)
    wave=wave[idx]
    trans=trans[idx]
    
    # Interpolate non existent absorptions
    if interp:
        mask = np.zeros(np.size(wave), dtype=bool)
        regions = [
            [73.15,73.3],
        ]
        for r in regions:
            m = (wave > r[0]) & (wave < r[1])
            mask[m] = True
        model = interp1d(wave[~mask],trans[~mask],kind='cubic',fill_value='extrapolate')
        trans = model(wave)

        
    # Return request
    return wave,trans

def getATransBlue1Old(altitude,za,wv,interp=False):
    from fifipy.spectra import getResolution
    from scipy.interpolate import interp1d

    w1 =  np.arange(67,125,1.)
    w2 = list(w1.copy())
    w2.append(125)
    w2 = w2[1:]
    w1 = np.array(w1)
    w2 = np.array(w2)
    res = getResolution('B1', np.sqrt(w1*w2))
    #w1 =  [    67,78.75,  90,99.5, 104, 110, 120]
    #w2 =  [ 78.75,   90,99.5, 104, 110, 120, 125]
    #res = [   550,  590, 670, 740, 804, 920,1045]

    wave=[]
    trans=[]

    for w1i, w2i, resi in zip(w1,w2,res):
        try:
            result = callAtran(0,altitude,za,wv,w1i,w2i,resi)
        except:
            print('2nd try ...')
            result = callAtran(0,altitude,za,wv,w1i,w2i,resi)
        wave.append(result[1])
        trans.append(result[2])


    # t1 = timer()
    # pool = mp.Pool(len(w1))
    # res  = [pool.apply_async(callAtran, args=(i,altitude,za,wv,w1[i],w2[i],res[i])) for i in range(len(w1))]
    # results = [p.get() for p in res]
    # results.sort()
    # pool.terminate()
    # t2 = timer()
    # #print ("time elapsed: ",t2-t1)
    
    # wave = []
    # trans = []
    # for r in results:
    #     wave.append(r[1])
    #     trans.append(r[2])

    wave = np.concatenate(wave)
    trans = np.concatenate(trans)
    u, idx = np.unique(wave, return_index=True)
    wave=wave[idx]
    trans=trans[idx]
    
    # Interpolate non existent absorptions
    if interp:
        mask = np.zeros(np.size(wave), dtype=bool)
        regions = [
            [73.15,73.3],
            [85.95,86.15],
            [87.2,87.75],
            [91.7,92.1],
            [96.65,97],
            [102.15,102.5],
            [110.95,111.18],
            [115.25,115.8],
            [115.3,115.8],
            [118.1,118.5],
            [120.45,120.8],
            [123.2,123.7],
            [123.85,124.1]
        ]
        for r in regions:
            m = (wave > r[0]) & (wave < r[1])
            mask[m] = True
        model = interp1d(wave[~mask],trans[~mask],kind='cubic',fill_value='extrapolate')
        trans = model(wave)
        
    # Return request
    return wave,trans
    

            
def getATrans(altitude, za, wv):
    import multiprocessing as mp

    w1 =  [ 45,  50,  55,  65, 70, 80, 90,100,105,120, 140, 160, 180]
    w2 =  [ 50,  55,  65,  70, 80, 90,100,105,120,140, 160, 180, 200]
    res = [850, 900,1000,1300,550,600,750,800,1000,800,1000,1300,1600]

    pool = mp.Pool(len(w1))
    res  = [pool.apply_async(callAtran, args=(i,altitude,za,wv,w1[i],w2[i],res[i])) for i in range(len(w1))]
    results = [p.get() for p in res]
    results.sort()
    pool.terminate()

    #print ("time elapsed: ",t2-t1)
    
    wave = []
    trans = []
    for r in results:
        wave.append(r[1])
        trans.append(r[2])
    
    # Return request
    return np.concatenate(wave), np.concatenate(trans)

def getATransRedOld(altitude, za, wv,interp=False):
    from scipy.interpolate import interp1d
    from fifipy.spectra import getResolution

    #w1 =  [105,120, 140, 160, 180]
    #w2 =  [120,140, 160, 180, 206]
    #res = [670,840,1050,1320,1635]
    w1 = np.arange(105,205,2.0)
    w2 = list(w1.copy())
    w2 = list(w2)
    w2.append(206)
    w2 = w2[1:]
    w1 = np.array(w1)
    w2 = np.array(w2)
    res = getResolution('R',np.sqrt(w1*w2))

    # t1 = timer()
    # pool = mp.Pool(len(w1))
    # res  = [pool.apply_async(callAtran, args=(i,altitude,za,wv,w1[i],w2[i],res[i])) for i in range(len(w1))]
    # results = [p.get() for p in res]
    # results.sort()
    # pool.terminate()
    # t2 = timer()
    # #print ("time elapsed: ",t2-t1)
    # wave = []
    # trans = []
    # for r in results:
    #     wave.append(r[1])
    #     trans.append(r[2])

    wave=[]
    trans=[]
    
    for w1i, w2i, resi in zip(w1,w2,res):
        try:
            result = callAtran(0,altitude,za,wv,w1i,w2i,resi)
        except:
            print('2nd try ...')
            result = callAtran(0,altitude,za,wv,w1i,w2i,resi)
        wave.append(result[1])
        trans.append(result[2])

    wave = np.concatenate(wave)
    trans = np.concatenate(trans)
        
    u, idx = np.unique(wave, return_index=True)
    wave=wave[idx]
    trans=trans[idx]

    # Add feature in emission
    mask = np.zeros(np.size(wave), dtype=bool)
    regions = [
        [139.16,139.5]
    ]
    for r in regions:
        m = (wave > r[0]) & (wave < r[1])
        mask[m] = True
        model = interp1d(wave[~mask],trans[~mask],kind='cubic',fill_value='extrapolate')
        diff =  model(wave) - trans
        trans[m] += 2*diff[m]


    # Subtract features by using only the right side of them not affected by other lines
    regions = [
        [143.16,143.44],
        [151.11,151.35]
    ]
    for r in regions:
        idx = np.where( (wave > r[0]) & (wave < r[1]))
        idx = (np.array(idx))[0]
        idxn = np.arange(-1,-np.size(idx),-1)+idx[0]
        atn = trans[idx]-np.max(trans[idx])
        trans[idx] -= atn
        trans[idxn] -= atn[:-1]

    # Delete feature by replicating the right side of a line
    regions = [
        [156.2,156.99]
    ]
    for r in regions:
        idx = np.where( (wave > r[0]) & (wave < r[1]))
        idx = (np.array(idx))[0]
        idxn = np.arange(-1,-np.size(idx),-1)+idx[0]
        trans[idxn] = trans[idx[:-1]]
        
    # Delete feature by replicating the left side of a line
    regions = [
        [146.16,146.91],
        [169.5,170.1],
        [188,188.28],
        [195.3,195.68]
    ]
    for r in regions:
        idx = np.where( (wave > r[0]) & (wave < r[1]))
        idx = (np.array(idx))[0]
        idxp = np.arange(-1,-np.size(idx),-1)+idx[0]+np.size(idx)*2+1
        trans[idxp]=trans[idx[:-1]]

    # Interpolate non existent absorptions
    if interp:
        mask = np.zeros(np.size(wave), dtype=bool)
        regions = [
#            [139.2,139.5],
            [155.25,155.5],
            [159.90,160.03],
            [164.8,165.25],
            [176.42,176.7]
        ]
        for r in regions:
            m = (wave > r[0]) & (wave < r[1])
            mask[m] = True
        model = interp1d(wave[~mask],trans[~mask],kind='cubic',fill_value='extrapolate')
        trans = model(wave)

    # Little correction for a line (slide to the left)
    regions = [
        [144.12,144.5]
    ]
    for r in regions:
        idx = np.where( (wave > r[0]) & (wave < r[1]))
        idx = (np.array(idx))[0]
        nidx = int(np.size(idx)*0.05/(r[1]-r[0]))
        idxn = idx-nidx
        trans[idxn] = trans[idx]

        
    # Return request
    return wave, trans
