#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:32:00 2019

@author: sylvainbertaina
"""

from __future__ import division, absolute_import, unicode_literals, print_function
import struct
import sys
import getopt
import os
import glob
import fnmatch
import csv
import json
import numpy as np
import decimal
from pathlib import Path

from itertools import zip_longest

def BrukerListFile_0(PATH):
 
    filelist=[]
    for file in os.listdir(PATH):
        if fnmatch.fnmatch(file, '*.dta') or fnmatch.fnmatchcase(file, '*.spc') or fnmatch.fnmatch(file, '*.DTA') or fnmatch.fnmatchcase(file, '*.SPC'):
            full_path = os.path.join(PATH, file)
            filelist.append(full_path)
            filelist.sort()
    return filelist

def BrukerListFile(PATH,Recursive=False):
      
    exts = [".dsc",".DSC",".DTA",".dta",".spc",".SPC",".par",".PAR"]
    if Recursive:
        files = [p for p in Path(PATH).rglob('*') if p.suffix in exts]
    else:
        files = [p for p in Path(PATH).iterdir() if p.suffix in exts]

    return(list(sorted(files)))

def readEPR(file):
    filetmp = os.path.splitext(file)[0]
    try:
        with open(filetmp+'.dta'):
            pass
        print('elixis file')
        parm = parameters_read_Elixys(file)
        try:
            parm['YPTS']
            print('2D')
            xdata, ydata = generate_axisElixys(file)
            zdata = generate_dataElixys(file)
            return xdata, ydata, zdata, parm
        except KeyError:
            print('1D')
            xdata = generate_axisElixys(file)
            ydata = generate_dataElixys(file)
            return xdata, ydata, parm
    except IOError:
        with open(filetmp+'.spc'):
            pass
        print('EMX file')
        parm = parameters_read_EMX(filetmp+'.par')
        try:
            parm['SSY']
            print('2D')
            xdata = generate_xdataEMX(parm)
            zdata = generate_ydataEMX(filetmp+'.spc')
            zdata = np.reshape(zdata, (int(parm['SSY']), -1))
            ydata = np.linspace(
                parm['XYLB'], parm['XYLB'] + parm['XYWI'], int(parm['SSY']))
            return xdata, ydata, zdata, parm
        except KeyError:
            parm['JUN']
            print('1D')
            xdata = generate_xdataEMX(parm)
            ydata = generate_ydataEMX(filetmp+'.spc')
            return xdata, ydata, parm
        except KeyError:
            print('something wrong')
    except IOError:
        print('not exist')


def formatTester(file):
    filetmp = os.path.splitext(file)[0]
    try:
        with open(filetmp+'.dsc'):
            pass
        return 'Elixis'
    except IOError:
        with open(filetmp+'.spc'):
            pass
        return('EMX')
    except IOError:
        return('Wrong format')


def parameters_read_EMX(inputfile):
    """
    Generate a dictionary with parameter file EMX format

    Parameters:
        inputfile: .par parameter file

    Returns:
        parm = dictionary containing the parameters. The numerical data are converted in float
    """
    # Generate a dictionary with parameter file
    namefile = os.path.basename(inputfile)
    namefile = os.path.splitext(namefile)[0]
    parm = {}
    with open(inputfile) as fh:
        for line in fh:
            key, val = line.strip().split(' ', 1)
            parm[key] = val.strip()
            parm['namefile'] = namefile
            for key in parm:    # convert numbers from str to float when possible
                try:
                    float(parm[key])
                    parm[key] = float(parm[key])
                except ValueError:
                    False
    return parm


def generate_xdataEMX(parm):
    """
    Generate the x data from the parameters dictionary

    Parameters:
        parm: [dict] parameters

    Returns:
        xdata = nd.array[XNbPoints]
    """
    # Extracts the x axis data from the parameter file
    try:
        xpoints = parm['SSX']
    except KeyError:
        xpoints = parm['ANZ']

    try:
        xwid = parm['GSI']
        xstart = parm['GST']
    except KeyError:
        xwid = parm['XXWI']
        xstart = parm['XXLB']

    xdata = np.linspace(xstart, xstart+xwid, int(xpoints))

    return xdata


def generate_ydataEMX(inputfile):
    """
    Generate the y data from the .spc EMX file

    Parameters:
        inputfile: .spc file

    Returns:
        ydata = nd.array[XNbPoints]
    """
    # Extracts the y axis data from the input file
    ydata = []
    fin = open(inputfile, 'rb')

    with open(inputfile, 'rb') as inh:
        indata = inh.read()
    for i in range(0, len(indata), 4):
        pos = struct.unpack('<f', indata[i:i + 4])
        ydata.append(pos[0])
    fin.close()
    return np.array(ydata)


def parameters_read_Elixys(inputfile):
    inputfile = os.path.splitext(inputfile)[0]+'.dsc'
    parm = {}
    with open(inputfile) as fh:
        for line in fh:
            try:
                key, val = line.strip().split('\t', 1)
                parm[key] = val.strip()
                #parm['namefile'] = namefile
                for key in parm:    # convert numbers from str to float when possible
                    try:
                        float(parm[key])
                        parm[key] = float(parm[key])
                    except ValueError:
                        False
            except ValueError:
                pass
    with open(inputfile) as fh:
        for line in fh:
            try:
                key, val = line.strip().split(' ', 1)
                parm[key] = val.strip()
                #parm['namefile'] = namefile
                for key in parm:    # convert numbers from str to float when possible
                    try:
                        float(parm[key])
                        parm[key] = float(parm[key])
                    except ValueError:
                        False
            except ValueError:
                pass
    namefile = os.path.basename(inputfile)
    namefile = os.path.splitext(namefile)[0]
    parm['namefile']=namefile
    return parm


def read_Elixys_data(inputfile):
    """
    Get binary data from file and convert it

    Parameters:
        inputfile: file path and name of DTA file [str]

    Returns:
        DTA_data: type based on dtype
    """
    param = parameters_read_Elixys(inputfile)
    inputfile = os.path.splitext(inputfile)[0]+'.dta'
    with open(inputfile, 'rb') as f:
        DTA_data = np.frombuffer(f.read(), dtype='>d')

    if param['IKKF'] == "CPLX":
        DTA_shape = int(len(DTA_data) / 2)
        DTA_data = np.ndarray(shape=(DTA_shape, 2),
                              dtype='>d', buffer=DTA_data)

    elif param['IKKF'] == "REAL":
        DTA_shape = int(len(DTA_data))
        DTA_data = np.ndarray(shape=(DTA_shape, 1),
                              dtype='>d', buffer=DTA_data)

    else:
        raise TypeError(
            "Incorrect format <{}>. Expected REAL or CPLX.".format(param['IKKF']))

    return DTA_data


def generate_dataElixys(inputfile):
    """
     Get binary data from file, convert it and generate data output

    Parameters:
        inputfile: file path and name of DTA file [str]

    Returns:
        data_Re: type based on dtype
        data_Re,data_Im: type based on dtype
    """
    data = read_Elixys_data(inputfile)
    param = parameters_read_Elixys(inputfile)
    if param["IKKF"] == "CPLX":
        data_Re, data_Im = np.hsplit(data, 2)
        data_Re = np.reshape(data_Re, (-1,int(param['XPTS'])))
        data_Im = np.reshape(data_Im, (-1,int(param['XPTS'])))
        return data_Re.copy().T, data_Im.copy().T
    else:
        data_Re = np.reshape(data, (int(param['XPTS']),-1))
        return data_Re.copy()


def generate_axisElixys(inputfile):
    """
     Get parameter file and generate axis 

    Parameters:
        inputfile: file path and name of DSC file [str]

    Returns:
        X_axis: type based on dtype
        X_axis, Y_axis: type based on dtype
    """

    param = parameters_read_Elixys(inputfile)
    Y_axis = None

    # will have range of x vals by default
    X_start = param["XMIN"]
    X_stop = param["XWID"] + X_start
    X_pts = param["XPTS"]
    X_axis = np.linspace(X_start, X_stop, int(X_pts))

    if "YPTS" in param:
        Y_start = param["YMIN"]
        Y_stop = param["YWID"] + Y_start
        Y_pts = param["YPTS"]
        Y_axis = np.linspace(Y_start, Y_stop, int(Y_pts))
        return X_axis, Y_axis
    else: 
        return X_axis

