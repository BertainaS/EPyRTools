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


def BrukerListFiles(PATH,Recursive=False):
    """List all SPC (EMX) and DTA (Elixys) files in the PATH.  

    Args:
        PATH ([str]): path of the folder containing the Bruker files.
        Recursive (bool, optional): [to check the folder and subfolders]. Defaults to False.
    """      
    exts = [".DTA",".dta",".spc",".SPC"]
    if Recursive:
        files = [p for p in Path(PATH).rglob('*') if p.suffix in exts]
    else:
        files = [p for p in Path(PATH).iterdir() if p.suffix in exts]

    return(list(sorted(files)))

def readEPR(file):
    """
    Read EPR data and parameters from the specified file.

    This function reads Electron Paramagnetic Resonance (EPR) data and associated 
    parameters from a specified file. It supports both Elixys and EMX file formats 
    and can handle both 1D and 2D datasets. Depending on the data type and file format, 
    the function returns the appropriate data and parameters.

    Args:
        file (str): Path to the input EPR data file.

    Returns:
        tuple: Depending on the data type (1D or 2D) and file format, it returns:
            - 2D Data (Elixys): (X-axis data, Y-axis data, Z-axis data, parameters)
            - 1D Data (Elixys): (X-axis data, Y-axis data, parameters)
            - 2D Data (EMX): (X-axis data, Y-axis data, Z-axis data, parameters)
            - 1D Data (EMX): (X-axis data, Y-axis data, parameters)
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If required parameters are missing from the file.
        IOError: If there are issues reading the file.
    """
    filetmp = os.path.splitext(file)[0]
    
    try:
        # Check if the file is an Elixys file by attempting to open the .dta file
        with open(filetmp + '.dta'):
            pass
        print('Elixys file')

        # Read parameters for Elixys file
        parm = parameters_read_Elixys(file)

        try:
            # Check if the parameters indicate a 2D data set
            _ = parm['YPTS']
            print('2D')

            # Generate axes and data for 2D Elixys data
            xdata, ydata = generate_axisElixys(file)
            zdata = generate_dataElixys(file)
            return xdata, ydata, zdata, parm
        except KeyError:
            # If 'YPTS' is not in parameters, it's a 1D data set
            print('1D')
            xdata = generate_axisElixys(file)
            ydata = generate_dataElixys(file)
            return xdata, ydata, parm
    except IOError:
        # If not an Elixys file, check if it is an EMX file by attempting to open the .spc file
        with open(filetmp + '.spc'):
            pass
        print('EMX file')

        # Read parameters for EMX file
        parm = parameters_read_EMX(filetmp + '.par')

        try:
            # Check if the parameters indicate a 2D data set
            _ = parm['SSY']
            print('2D')

            # Generate axes and data for 2D EMX data
            xdata = generate_xdataEMX(parm)
            zdata = generate_ydataEMX(filetmp + '.spc')
            zdata = np.reshape(zdata, (int(parm['SSY']), -1))
            ydata = np.linspace(
                parm['XYLB'], parm['XYLB'] + parm['XYWI'], int(parm['SSY']))
            return xdata, ydata, zdata, parm
        except KeyError:
            try:
                # If 'SSY' is not in parameters, check if it's a 1D data set by looking for 'JUN'
                _ = parm['JUN']
                print('1D')

                # Generate axes and data for 1D EMX data
                xdata = generate_xdataEMX(parm)
                ydata = generate_ydataEMX(filetmp + '.spc')
                return xdata, ydata, parm
            except KeyError:
                print('Something went wrong with reading EMX parameters.')
    except IOError:
        # If neither file type exists, print an error message
        print('File does not exist')


def formatTester(file):
    """
    Determine the format of the EPR data file (Elixys or EMX).

    Args:
        file (str): Path to the input EPR data file.

    Returns:
        str: 'Elixys' if the file is in Elixys format,
             'EMX' if the file is in EMX format,
             'Wrong format' if the file format is unrecognized.
    """
    filetmp = os.path.splitext(file)[0]

    try:
        # Check if the file is an Elixys file by attempting to open the .dsc file
        with open(filetmp + '.dsc'):
            pass
        return 'Elixys'
    except IOError:
        # If not an Elixys file, check if it is an EMX file by attempting to open the .spc file
        try:
            with open(filetmp + '.spc'):
                pass
            return 'EMX'
        except IOError:
            # If neither file type exists, return 'Wrong format'
            return 'Wrong format'

def parameters_read_EMX(inputfile):
    """
    Read parameters from an EMX .par file and convert numerical data to float.

    Args:
        inputfile (str): Path to the .par parameter file.

    Returns:
        dict: Dictionary containing the parameters with numerical data converted to float.
    """
    # Get the base name of the file without extension
    namefile = os.path.basename(inputfile)
    namefile = os.path.splitext(namefile)[0]
    
    parm = {}

    # Read the parameter file
    with open(inputfile) as fh:
        for line in fh:
            key, val = line.strip().split(' ', 1)
            parm[key] = val.strip()

    parm['namefile'] = namefile

    # Convert numerical values from str to float when possible
    for key in parm:
        try:
            parm[key] = float(parm[key])
        except ValueError:
            pass  # Keep the value as string if it can't be converted to float

    return parm



def generate_xdataEMX(parm):
    """
    Generate the X-axis data from the parameters dictionary.

    Args:
        parm (dict): Dictionary of parameters.

    Returns:
        np.ndarray: Numpy array of X-axis data.
    """
    # Extract the number of X points
    xpoints = parm.get('SSX', parm.get('ANZ'))

    # Extract the width and start values for the X axis
    xwid = parm.get('GSI', parm.get('XXWI'))
    xstart = parm.get('GST', parm.get('XXLB'))

    # Generate the X-axis data using linspace
    xdata = np.linspace(xstart, xstart + xwid, int(xpoints))

    return xdata


def generate_ydataEMX(inputfile):
    """
    Generate the Y-axis data from a .spc EMX file.

    This function reads a .spc file in binary mode, unpacks its contents into 
    floats, and returns them as a NumPy array. The .spc file is assumed to 
    contain 32-bit floating point numbers in little-endian format.

    Args:
        inputfile (str): Path to the .spc file.

    Returns:
        np.ndarray: A NumPy array containing the Y-axis data extracted from the 
                    .spc file.
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
        struct.error: If the binary data cannot be unpacked into floats.
    """
    ydata = []

    # Open the .spc file in binary mode
    with open(inputfile, 'rb') as inh:
        indata = inh.read()

    # Read the binary data and unpack it into floats
    for i in range(0, len(indata), 4):
        pos = struct.unpack('<f', indata[i:i + 4])
        ydata.append(pos[0])

    return np.array(ydata)


def parameters_read_Elixys(inputfile):
    """
    Reads parameters from a .dsc file and returns them as a dictionary.
    
    This function reads a file with a .dsc extension, parses its contents,
    and extracts key-value pairs. If the value is numeric, it converts the
    value from a string to a float. Additionally, it includes the base name
    of the file (without extension) in the dictionary with the key 'namefile'.
    
    Parameters:
    inputfile (str): The path to the input file. The function changes its 
                     extension to .dsc.
    
    Returns:
    dict: A dictionary with keys as parameter names and values as parameter
          values. Numeric values are converted to floats when possible. The
          dictionary also includes the base name of the file under the key
          'namefile'.
    """
    
    # Change the file extension to .dsc
    inputfile = os.path.splitext(inputfile)[0] + '.dsc'
    
    parm = {}
    
    # Read and parse the file
    with open(inputfile) as fh:
        for line in fh:
            line = line.strip()
            
            # Try to split by tab first, then by space if tab splitting fails
            key_val_pair = line.split('\t', 1) if '\t' in line else line.split(' ', 1)
            
            if len(key_val_pair) == 2:
                key, val = key_val_pair
                key = key.strip()
                val = val.strip()
                parm[key] = val
                
                # Convert numbers from strings to floats when possible
                try:
                    parm[key] = float(val)
                except ValueError:
                    pass
    
    # Get the base name of the file without extension
    namefile = os.path.basename(inputfile)
    namefile = os.path.splitext(namefile)[0]
    parm['namefile'] = namefile
    
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

