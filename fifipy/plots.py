#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:51:16 2018

@author: dfadda
"""

def plotResponse(channel,dichroic,order,w,medspec,alpha,caldir,period):
    from matplotlib import rcParams
    rcParams['font.family']='STIXGeneral'
    rcParams['font.size']=18
    rcParams['mathtext.fontset']='stix'
    rcParams['legend.numpoints']=1
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy import units as u
    from astropy.modeling.models import BlackBody
    from astropy.io import fits
    import os


    if (period != 'New') & (period != 'Old'):
        print('age has to be Old or New')
        return

    if channel == 'R':
        ctrans = 'Red'
        eff = 'GrtEffRed.txt'
        trans = 'RedChannel'+str(dichroic)+'Dichroic_'+period+'.txt'
        label1 = 'D'+str(dichroic)+'*R*GE'
        label2 = 'D'+str(dichroic)+'*R*GE*R$_{red}$'
        title = 'Red [D'+str(dichroic)+'] '+period
        out = 'D'+str(dichroic)+'R.pdf'
        xlim = [100,210]
        w1=130; w2=160
    else:
        ctrans = 'Blue'
        if order == 1:
            eff = 'GrtEffB1.txt'
            trans = 'BlueChannel1stOrder'+str(dichroic)+'Dichroic_'+period+'.txt'
            label1 = 'D'+str(dichroic)+'*B1*GE'
            label2 = 'D'+str(dichroic)+'*B1*GE*R$_{blue}$'
            title = 'Blue [D'+str(dichroic)+' M1] '+period
            out = 'D'+str(dichroic)+'B1.pdf'
            w1=75; w2=95
            xlim = [65,135]
        else:
            eff = 'GrtEffB2.txt'
            trans = 'BlueChannel2ndOrder'+str(dichroic)+'Dichroic_'+period+'.txt'
            label1 = 'D'+str(dichroic)+'*B2*GE'
            label2 = 'D'+str(dichroic)+'*B2*GE*R$_{blue}$'
            title = 'Blue [D'+str(dichroic)+' M2] '+period
            out = 'D'+str(dichroic)+'B2.pdf'
            if dichroic == 105:
                w1=55; w2=70
            else:
                w1=59; w2=65
            xlim = [45,75]

    path0, file0 = os.path.split(__file__)
    transdir = os.path.join(path0,'data','Trasmission')
    f = np.loadtxt(os.path.join(transdir , trans), delimiter='\t')
    wt = f[:,0]
    tt = f[:,1]
    mask = np.isfinite(tt)
    wt = wt[mask]
    tt = tt[mask]
    f = np.loadtxt(os.path.join(transdir, eff), delimiter='\t')
    we = f[:,0]
    te = f[:,1]
    # Interpolation
    yinterp = np.interp(wt, we, te)
    tt *= yinterp
    f = np.loadtxt(os.path.join(transdir, 'responsivity'+ctrans+'.txt'), delimiter=' ')
    wr = f[:,0]
    tr = f[:,1]
    rinterp = np.interp(wt, wr, tr)
    #tt *= rinterp
    
    wavelengths = w *1.e4 * u.AA
    temperature = 150 * u.K
    blackbody_nu = BlackBody(temperature)
    flux_nu = blackbody_nu(wavelengths)
    fnu = medspec/flux_nu
    fig,ax = plt.subplots(figsize=(16,8))
    ax.plot(wt,tt,label=label1,color='blue',alpha=0.75)

    ttr=tt*rinterp
    alphar = np.sum(ttr*tt)/np.sum(ttr*ttr)
    ax.plot(wt,tt*rinterp*alphar,label=label2,color='darkcyan',alpha=0.75)
    #ax.plot(wt,tt*rinterp,label='Trans D130+R+GE+Resp')
    if alpha == 1:
        yt = np.interp(w, wt, tt)
        alpha = np.sum(yt*fnu)/np.sum(fnu*fnu)
        f = fnu*alpha
        print('alpha is ', alpha)
    else:
        mt = (wt > w1) & (wt < w2)
        mw = (w > w1 ) & (w < w2)
        alpha = np.nanmedian(tt[mt])/np.nanmedian(fnu[mw])
        f = fnu*alpha
        print('alpha ', alpha)
    ax.plot(w,f,label='Rescaled $F_{det}/F_{BB_{150K}}$',color='darkorange',linewidth=3)


    if period == 'Old':
        #responsedir = os.path.join(path0,'data')
        responsedir = '/Users/dfadda/Pipeline/fifi-ls/data/response_files/20170816/'
        if channel == 'R':
            if dichroic == 105:
                responsefile ='Resp_FIFI_Red_D105_117-205_20170213.fits'
            else:
                responsefile = 'Resp_FIFI_Red_D130_129-206_20170213.fits'
        else:
            if dichroic == 105:
                if order == 1:
                    responsefile = 'Resp_FIFI_Blue_D105_Ord1_71-125_20170213.fits'
                else:
                    responsefile = 'Resp_FIFI_Blue_D105_Ord2_51-75_20170213.fits'
            else:
                if order == 1:
                    responsefile = 'Resp_FIFI_Blue_D130_Ord1_71-125_20170213.fits'
                else:
                    responsefile = 'Resp_FIFI_Blue_D130_Ord2_51-71_20170213.fits'
        hdl = fits.open(responsedir+responsefile)
        data = hdl['PRIMARY'].data
        hdl.close()
        w_r = data[0,:]
        f_r = data[1,:]

        # Renormalize to our response
        yt = np.interp(w_r, w, f)
        idx = np.isfinite(yt) & np.isfinite(f_r)
        alpha = np.sum(yt[idx]*f_r[idx])/np.sum(f_r[idx]*f_r[idx])
        ax.plot(w_r,f_r*alpha,label='Rescaled measured response',color='green',linewidth=2)
        
        
    ax.set_xlabel('Wavelength [$\mu$m]')
    ax.set_ylabel('Transmission')
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.legend()
    ax.grid()
    plt.show()
    fig.savefig(caldir+out)
