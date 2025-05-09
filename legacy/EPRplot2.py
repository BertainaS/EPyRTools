#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:32:00 2019

@author: sylvainbertaina
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from numpy import log, cumsum, reshape, zeros, kaiser, hamming, exp, shape, append, real, imag, pad
from numpy.fft import fft, fftshift, fftfreq, fft2
#from scipy.fftpack import fft,fftfreq, fftshift
from scipy.optimize import curve_fit
import copy
from scipy import ndimage


def plot2D(x, y, z, parm, cmap='inferno', blCorrection=False, integrate=False, vmax=None, vmin=None):
    """
    Create a 2D plot.

    Args:
        x (array): X-axis data.
        y (array): Y-axis data.
        z (array or tuple): Z-axis data. If a tuple is provided, it's assumed to contain the z-data.
        parm (dict): Parameters dictionary containing metadata.
        cmap (str, optional): Colormap to use. Defaults to 'inferno'.
        blCorrection (bool, optional): Whether to perform baseline correction. Defaults to False.
        integrate (bool, optional): Whether to integrate. Defaults to False.
        vmax (float, optional): Maximum value for colormap normalization. Defaults to None.
        vmin (float, optional): Minimum value for colormap normalization. Defaults to None.

    Returns:
        None
    """
    # If z is a tuple, assume it contains the z-data and transpose it
    if isinstance(z, tuple):
        z = z[0].T

    try:
        # Check if 'SSY' exists in parameters
        parm['SSY']
        if parm["JEY"] == 'angle-sweep':
            # Call EPRAngle2D if JEY is 'angle-sweep'
            EPRAngle2D(x, z, parm, y, cmap=cmap, integrate=integrate, vmin=vmin, vmax=vmax)
        elif parm["JEY"] == 'power-sweep':
            # Print message for 'power-sweep'
            print('Create the 2D plot in power mode')
    except KeyError:
        # If 'SSY' doesn't exist in parameters
        if blCorrection:
            # Perform baseline correction if blCorrection is True
            YPTS = int(parm['YPTS'])
            for k in range(YPTS):
                z[k] = baselineCorrection(x, z[k], deg=1)

        # Create imshow plot
        plt.imshow(z, interpolation='quadric', aspect='auto', origin='lower',
                   extent=[x[0], x[-1], y[0], y[-1]], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(parm['namefile'])
        if parm['YNAM'] == 'Field':
            plt.ylabel('Magnetic field (G)')
        plt.xlabel('Time (ns)')


def monoexp(x, a1, b1, c):
    """Calculate the value of a monotonically decreasing exponential function.
    
    Args:
        x (float): The input value.
        a1 (float): The amplitude of the exponential function.
        b1 (float): The decay constant of the exponential function.
        c (float): The vertical shift of the exponential function.
    
    Returns:
        float: The value of the monotonically decreasing exponential function at the given input value.
    """
    
    return a1 * exp(-(x/b1)) + c


def expstretched(x, a1, b1, c,n):
    """Calculate the stretched exponential function value for the given parameters.
    
    Args:
        x (float): The input value.
        a1 (float): Coefficient a1.
        b1 (float): Coefficient b1.
        c (float): Coefficient c.
        n (float): Exponential power.
    
    Returns:
        float: The result of the exponential function.
    """
    
    return a1 * exp(-(x/b1)**n) + c

def plotCW(x, y, par=None, lw=None, color=None):
    """Plot a custom line graph with specified x and y values.
    
    Args:
        x (array-like): The x values for the plot.
        y (array-like): The y values for the plot.
        par (dict, optional): Additional parameters for customizing the plot.
        lw (float, optional): The line width of the plot.
        color (str, optional): The color of the plot line.
    
    Returns:
        None
    """
    
    plt.plot(x, y, color=color, lw=lw)
    plt.xlim(x[0], x[-1])
    plt.ylabel('EPR line dI/dH (a.u.)')
    try:
        par['JEX'] == 'field-sweep'
        plt.xlabel('Magnetic Field (G)')
    except KeyError:
        par['XNAM'] == 'Field'
        plt.xlabel('Magnetic Field (G)')
    except KeyError:
        plt.xlabel('Time (s)')
    if par != None:
        plt.title(par['namefile'])
    # plt.yticks([])


def baselineCorrection(x, y, deg=1, expdecay=False, p0=None):
    """Perform baseline correction on input data.
    
    Args:
        x (array-like): The x-coordinates of the data points.
        y (array-like): The y-coordinates of the data points.
        deg (int, optional): The degree of the polynomial to fit for baseline correction. Defaults to 1.
        expdecay (bool, optional): If True, perform exponential decay fitting for baseline correction. Defaults to False.
        p0 (array-like, optional): Initial guess for the parameters of the exponential decay function. Defaults to None.
    
    Returns:
        array-like: The baseline-corrected data.
    """
    if expdecay:
        popt, pcov = curve_fit(monoexp, x, y, p0=p0)
        baseline = monoexp(x, *popt)
        corrected = y - baseline
        return corrected
    coefs = poly.polyfit(x, y, deg)
    baseline = poly.polyval(x, coefs)
    corrected = y - baseline
    return corrected


def baselineCorrectionExp(x, y, p0):
    """Perform exponential baseline correction on the input data.
    
    Args:
        x (array-like): The x values of the data points.
        y (array-like): The y values of the data points.
        p0 (array-like): Initial guess for the parameters of the exponential function.
    
    Returns:
        array-like: The baseline-corrected data points.
    """
    popt, pcov = curve_fit(monoexp, x, y, p0=p0)
    baseline = monoexp(x, *popt)
    corrected = y - baseline
    return corrected


def baseline(x, y):
    """
    Calculate the baseline of a given set of data points.

    Args:
        x (list): List of x-coordinates.
        y (list): List of y-coordinates.

    Returns:
        list: Corrected y-coordinates after baseline correction.
    """
    slope = (y[-1]-y[0])/(x[-1]-x[0])
    corrected = y - slope*x
    corrected -= corrected[0]
    return corrected


def EPRAngle2D(x, y, par, angle, cmap='viridis', vmin=None, vmax=None, integrate=False):
    """
    Create a 2D plot for Electron Paramagnetic Resonance (EPR) data as a function of angle.

    Args:
        x (array): X-axis data.
        y (array): Y-axis data.
        par (dict): Parameters dictionary containing metadata.
        angle (array): Array of angle values.
        cmap (str, optional): Colormap to use. Defaults to 'viridis'.
        vmin (float, optional): Minimum value for colormap normalization. Defaults to None.
        vmax (float, optional): Maximum value for colormap normalization. Defaults to None.
        integrate (bool, optional): Whether to integrate the y-data. Defaults to False.

    Returns:
        None
    """
    # Deep copy of y to avoid modifying the original data
    yp = copy.deepcopy(y)
    
    # Apply baseline correction and optional integration
    for k in range(len(angle)):
        yp[k] = baselineCorrection(x, yp[k], deg=2)
        if integrate:
            yp[k] = np.cumsum(yp[k])

    # Create the 2D plot using imshow
    plt.imshow(yp.T, interpolation='quadric', aspect='auto', origin='lower',
               extent=[angle[0], angle[-1], par['XXLB'], par['XXLB'] + par['XXWI']], 
               vmin=vmin, vmax=vmax, cmap=cmap)
    
    # Labeling the plot
    plt.xlabel('Angle (deg)')
    plt.ylabel('Magnetic Field (G)')
    plt.title(par['namefile'])
    
    # No explicit return needed
    return None


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
    zp = copy.deepcopy(z)
    YPTS = int(par['YPTS'])
    for k in range(YPTS):
        zp[k] = baselineCorrection(time, zp[k], 1)
    plt.imshow(zp.T, interpolation='quadric', aspect='auto', origin='lower',
               extent=[xmin, xmax, time[0], time[-1]], vmax=vmax, cmap=cmap)
    plt.title(par['namefile'])
    plt.ylabel('Time (ns)')


def plotFFT1D(x, y, par=None, lw=None, color=None, baselinecorr=True, blcdeg=1, label=None, kind=0,expdecay=False, p0=None):
    im = []
    if type(y) == tuple:
        print('tupple')
        y = copy.deepcopy(y[0].reshape(-1))
        #im = copy.deepcopy(y[1].reshape(-1))
    else:
        re = copy.deepcopy(y)
    windows = kaiser(len(x), 4)
    if baselinecorr == True:
        y = baselineCorrection(x, y,expdecay=expdecay, deg=blcdeg,p0=p0)
    else:
        pass
    y *= windows
    yl = append(y, zeros(len(x)))

    if kind == 0:
        amp = abs((fft(yl)))
    else:
        if kind == 1:
            amp = real((fft(yl)))
        else:
            amp = imag((fft(yl)))

    fq = (fftfreq(len(yl), x[1] - x[0]))
    plt.plot(fftshift(fq * 1000), fftshift(amp),
             color=color, lw=lw, label=label)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('FFT')
    plt.yticks([])
    if par is not None:
        plt.title(par['namefile'])


def plotFFT2D(time, z, par, xmin=0, xmax=100, cmap='viridis', vmin=None, vmax=None, xlab='Detuning (MHz)', expdecay=True, p0=None, deg=10):
    if type(z) == tuple:
        zp = copy.deepcopy(z[0].T)
    else:
        zp = copy.deepcopy(z)

    ampFFT = zeros(zp.shape)
    ampFFT = append(ampFFT, ampFFT, axis=1)
    ampFFT = append(ampFFT, ampFFT, axis=1)
    windows = kaiser(len(time), 3)
    #window = kaiser(2*len(time), 3)
    #windows = window[len(time)-1:-1]

    for i in range(zp.shape[0]):
        amp = baselineCorrection(
            time, zp[i], deg=deg, expdecay=expdecay, p0=p0)
        #amp = baselineCorrectionExp(time, z[i],[1e1,1e3,1e2])
        amp *= windows
        ampl = append(amp, zeros(3*len(time)))
        ampFFT[i] = abs(fftshift(fft(ampl)))

    fq = fftfreq(len(ampl), time[1] - time[0])
    plt.imshow(ampFFT.T, interpolation='quadric', aspect='auto', origin='lower', extent=[
               xmin, xmax, fq.min() * 1e3, fq.max() * 1e3], vmax=vmax, vmin=vmin, cmap=cmap)  #
    plt.xlabel(xlab)
    plt.ylabel('Frequency (MHz)')
    if par != None:
        plt.title(par['namefile'])
    else:
        pass


def FFT2D(time, z, nzeros=2, plot=True):
    if type(z) == tuple:
        z = z[0].T
    window = kaiser(2*len(z), 4)
    windows = window[len(z)-1:-1]
    #windows= hamming(len(z))
    for k in range(len(z)):
        z[k, :] = baselineCorrection(time, z[k, :], deg=2)
    for k in range(len(z)):
        z[:, k] = baselineCorrection(time, z[:, k], deg=2)
    for k in range(len(z)):
        z[k, :] *= windows
    for k in range(len(z)):
        z[:, k] *= windows
    z = pad(z, (0, (nzeros-1)*len(z)), 'constant')
    amp = fft2(z)
    amp = abs(fftshift(amp))
    fq = 1e3*(fftfreq(len(amp), time[1] - time[0]))
    return amp, fq


def plotHYSCORE_Bruker(real, par, xmin=0, xmax=100, cmap='viridis', vmax=None):
    plt.imshow(real.T, interpolation='quadric', aspect='auto', origin='lower',
               extent=[par['XMIN']*1e3, par['XMIN']*1e3+par['XWID']*1e3, par['YMIN']*1e3, par['YMIN']*1e3+par['YWID']*1e3], cmap='viridis', vmax=vmax)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Frequency (MHz)')
    plt.axis('scaled')


def plotHYSCORE(amp, fq, cmap='viridis', vmax=None):
    plt.imshow((amp), interpolation='quadric', aspect='auto', origin='lower', cmap=cmap,
               extent=[fq.min(), fq.max(), fq.min(), fq.max()], vmax=vmax)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Frequency (MHz)')
    plt.axis('scaled')


def plot1DElixys(x, y, parm, blCorrection=False):
    im = []
    if type(y) == tuple:
        re = copy.deepcopy(y[0].reshape(-1))
        im = copy.deepcopy(y[1].reshape(-1))
    else:
        re = copy.deepcopy(y)
    if blCorrection == True:
        re = baselineCorrection(x, re, deg=10)
        try:
            im[0]
            im = baselineCorrection(x, im, deg=10)
        except IndexError:
            pass
    else:
        pass

    try:
        im[0]
        plt.plot(x, re, label="Real part")
        plt.plot(x, im, label="Imaginary part")
        plt.legend(loc=1)
    except IndexError:
        plt.plot(x, re)
    plt.xlim(x[0], x[-1])
    plt.ylabel('Intensity(a.u.)')
    if parm['XUNI'] == "'G'":
        plt.xlabel('Magnetic field (G)')
    elif parm['XUNI'] == "'ns'":
        plt.xlabel('Time (ns)')
    elif parm['XUNI'] == "'GHz'":
        plt.xlabel('Frequency (GHz)')
    plt.title(parm['namefile'])

    plt.yticks([])


def plot1D(x, y, parm, blCorrection=False):
    try:
        parm['JUN']
        plotCW(x, y, parm)
    except KeyError:
        if parm['YTYP'] == 'NODATA':
            plot1DElixys(x, y, parm, blCorrection=blCorrection)
    except KeyError:
        print('erreur dans plot1D')
