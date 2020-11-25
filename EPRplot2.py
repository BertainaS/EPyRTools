#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:32:00 2019

@author: sylvainbertaina
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from numpy import log,cumsum, reshape, zeros, kaiser,hamming, exp, shape, append, real, imag, pad
from numpy.fft import fft, fftshift, fftfreq, fft2
#from scipy.fftpack import fft,fftfreq, fftshift
from scipy.optimize import curve_fit

def biexp(x, a1, b1, c):
    
    return a1 * exp(-x/b1) +c

def plotCW(x, y, par=None, lw=None, color=None):
    plt.plot(x, y, color=color, lw=lw)
    plt.xlim(x[0], x[-1])
    plt.ylabel('EPR line dI/dH (a.u.)')
    try:
        par['JEX']=='field-sweep'
        plt.xlabel('Magnetic Field (G)')
    except ValueError:
        plt.xlabel('Time (s)')
    if par != None:
        plt.title(par['namefile'])
    #plt.yticks([])


def baselineCorrection(x, y, deg=1):
    coefs = poly.polyfit(x, y, deg)
    baseline = poly.polyval(x, coefs)
    corrected = y - baseline
    return corrected

def baselineCorrectionExp(x, y,p0):
    popt, pcov = curve_fit(biexp, x, y,p0=p0)
    baseline = biexp(x,*popt)
    corrected = y - baseline
    return corrected

def baseline(x, y):
    slope=(y[-1]-y[0])/(x[-1]-x[0])
    corrected = y - slope*x
    corrected = corrected-corrected[0]
    return corrected

def EPRAngle2D(x, y, par, angle, cmap='viridis', vmax=None, integrate=False):
    for k in range(len(angle)):
        y[k] = baseline(x, y[k])
        if integrate:
            y[k] = cumsum(y[k])
    plt.imshow(y.T, interpolation='quadric', aspect='auto', origin='lower',
               extent=[angle[0], angle[-1], par['XXLB'], par['XXLB'] + par['XXWI']], vmax=vmax, cmap=cmap)
    plt.xlabel('Angle (deg)')
    plt.ylabel('Magnetic Field (G)')


# def plotTime(time, re, im=None, par=None, baselinecorr=True, blcdeg=1):
#     if baselinecorr == True:
#         re = baselineCorrection(time, re, deg=blcdeg)
#         if im is not None:
#             im=baselineCorrection(time, im, deg=blcdeg)
#         else:
#             pass
#     else:
#         pass

#     plt.plot(time, re)
#     if im is not None:
#         plt.plot(time, im)


#     plt.xlim(time[0], time[-1])
#     plt.ylabel('Intensity(a.u.)')
#     plt.xlabel('Time (ns)')
#     if par is not None:
#         plt.title(par['namefile'])
#     plt.yticks([])


# def plotEFS(field, y, par=None, lw=None, color=None, label=None):
#     plt.plot(field, y, color=color, lw=lw, label=label)
#     plt.xlim(field[0], field[-1])
#     plt.ylabel('Intensity(a.u.)')
#     plt.xlabel('Magnetic field (G)')
#     if par != None:
#         plt.title(par['namefile'])
#     plt.yticks([])

def plotAngleLW(angle, linew, par=None, color=None, label=None):
    plt.scatter(angle, linew, color=color, label=label)
    plt.xlim(angle[0], angle[-1])
    plt.ylabel('$\Gamma$ (G)')
    plt.xlabel('Angle (deg)')

def plotAngleRes(angle, res, par=None, color=None, label=None):
    plt.scatter(angle, res, color=color, label=label)
    plt.xlim(angle[0], angle[-1])
    plt.ylabel('H$_R$ (H)')
    plt.xlabel('Angle (deg)')

def plotAngleGfac(angle, geff, par=None, color=None, label=None):
    plt.scatter(angle, geff, color=color, label=label)
    plt.xlim(angle[0], angle[-1])
    plt.ylabel('g effectif')
    plt.xlabel('Angle (deg)')
    

def plotTime2D(time, z, par, xmin=0, xmax=100, cmap='viridis', vmax=None):
    YPTS = int(par['YPTS'])
    for k in range(YPTS):
        z[k] = baselineCorrection(time, z[k], 1)
    plt.imshow(z.T, interpolation='quadric', aspect='auto', origin='lower',
               extent=[xmin, xmax, time[0], time[-1]], vmax=vmax, cmap=cmap)
    plt.title(par['namefile'])
    plt.ylabel('Time (ns)')


def plotFFT1D(x, y, par=None, lw=None, color=None, baselinecorr=True, blcdeg=1, label=None, kind=0):
    windows = kaiser(len(x), 4)
    if baselinecorr == True:
        y = baselineCorrection(x, y, deg=blcdeg)
    else:
        pass
    y *= windows
    yl = append(y, zeros(len(x)))
    
    if kind==0:
        amp = abs((fft(yl)))
    else:
        if kind==1:
            amp = real((fft(yl)))
        else:
            amp=imag((fft(yl))) 
    
    fq = (fftfreq(len(yl), x[1] - x[0]))
    plt.plot(fftshift(fq * 1000), fftshift(amp), color=color, lw=lw, label=label)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('FFT')
    plt.yticks([])
    if par is not None:
        plt.title(par['namefile'])
        


def plotFFT2D(time, z, par, xmin=0, xmax=100, cmap='viridis', vmin=None,vmax=None, xlab = 'Detuning (MHz)'):
    if type(z)==tuple:
        z=z[0].T
    ampFFT = zeros(z.shape)
    ampFFT = append(ampFFT, ampFFT, axis=1)
    windows = kaiser(len(time), 2)
    #window = kaiser(2*len(time), 3)
    #windows = window[len(time)-1:-1]

    for i in range(z.shape[0]):
        amp = baselineCorrection(time, z[i], 6)
        #amp = baselineCorrectionExp(time, z[i],[1e1,1e3,1e2])
        amp *= windows
        ampl = append(amp, zeros(len(time)))
        ampFFT[i] = abs(fftshift(fft(ampl)))

    fq = fftfreq(len(ampl), time[1] - time[0])
    plt.imshow(ampFFT.T, interpolation='quadric', aspect='auto', origin='lower',
               extent=[xmin, xmax, fq.min() * 1e3, fq.max() * 1e3], vmax=vmax,vmin=vmin, cmap=cmap)  #
    plt.xlabel(xlab)
    plt.ylabel('Frequency (MHz)')
    if par != None:
        plt.title(par['namefile'])
    else:
        pass

    
def FFT2D(time, z, nzeros=2,plot=True):
    if type(z)==tuple:
        z=z[0].T
    window = kaiser(2*len(z), 4)
    windows = window[len(z)-1:-1]
    #windows= hamming(len(z))
    for k in range(len(z)):
        z[k,:]=baselineCorrection(time,z[k,:],deg=2)
    for k in range(len(z)):
        z[:,k]=baselineCorrection(time,z[:,k],deg=2)
    for k in range(len(z)):
        z[k,:] *= windows
    for k in range(len(z)):
        z[:,k] *= windows
    z=pad(z,(0,(nzeros-1)*len(z)),'constant')
    amp=fft2(z)
    amp=abs(fftshift(amp))
    fq = 1e3*(fftfreq(len(amp), time[1] - time[0]))        
    return amp,fq


def plotHYSCORE_Bruker(real, par, xmin=0, xmax=100, cmap='viridis', vmax=None):
    plt.imshow(real.T, interpolation='quadric', aspect='auto', origin='lower',
               extent=[par['XMIN']*1e3,par['XMIN']*1e3+par['XWID']*1e3 , par['YMIN']*1e3,par['YMIN']*1e3+par['YWID']*1e3], cmap='viridis',vmax=vmax)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Frequency (MHz)')
    plt.axis('scaled')
    
def plotHYSCORE(amp,fq,cmap='viridis',vmax=None):
    plt.imshow((amp),interpolation='quadric', aspect='auto', origin='lower',cmap=cmap,
           extent=[fq.min(),fq.max() , fq.min(),fq.max() ],vmax=vmax)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Frequency (MHz)')
    plt.axis('scaled')

def plot1DElixys(x,y,parm,blCorrection=False):
    if type(y)==tuple:
        re=y[0].reshape(-1)
        im=y[1].reshape(-1)
    else:
        re=y
    if blCorrection == True:
        re = baselineCorrection(x, re, deg=10)
        try:
            im
            im=baselineCorrection(x, im, deg=10)
        except ValueError:
            pass
    else:
        pass

    try:
        im
        plt.plot(x, re,label="Real part")
        plt.plot(x, im,label="Imaginary part")
    except ValueError:
        plt.plot(x, re)
    plt.xlim(x[0], x[-1])
    plt.ylabel('Intensity(a.u.)')
    if parm['XUNI']=="'G'":
        plt.xlabel('Magnetic field (G)')
    elif parm['XUNI']=="'ns'": 
        plt.xlabel('Time (ns)')  
    elif parm['XUNI']=="'GHz'": 
        plt.xlabel('Frequency (GHz)')   
    plt.title(parm['namefile'])
    plt.legend(loc=1)
    plt.yticks([])


def plot1D(x,y,parm,blCorrection=False):
    try:
        parm['JUN']
        plotCW(x,y,parm)
    except KeyError:
        if parm['YTYP']=='NODATA':
            plot1DElixys(x,y,parm,blCorrection=blCorrection)
    except KeyError:
        print('erreur dans plot1D')

def plot2D(x,y,z,parm,cmap='inferno',blCorrection=False):
    if type(z)==tuple:
        z=z[0].T
    try:
        parm['SSY'] 
        if parm["JEY"]=='angle-sweep':
            EPRAngle2D(x,z,parm,y,cmap='inferno', integrate=True)
        elif parm["JEY"]=='power-sweep':
            print('faire le plot en puissance 2D')
    except KeyError:
        if blCorrection==True:
            YPTS = int(parm['YPTS'])
            for k in range(YPTS):
                z[k] = baselineCorrection(x, z[k], 1)
        plt.imshow(z, interpolation='quadric', aspect='auto', origin='lower',
               extent=[x[0], x[-1], y[0], y[-1]],cmap=cmap)
        plt.title(parm['namefile'])
        plt.ylabel('Time (ns)')
        pass
