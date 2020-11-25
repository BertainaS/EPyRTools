import sys
sys.path.append("/Users/sylvainbertaina/Documents/__Labo/Python Scripts/EPRfunctions")
import BrukerConverter  as conv
import EPRplot2 as EPR
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from os.path import splitext,getmtime
from IPython.display import Markdown as md
PATH='/Users/sylvainbertaina/Documents/__Labo/DATA/Dalal/DMZF/2015_11_16_Pulse/'

import numpy.polynomial.polynomial as poly

colors = cm.Set1.colors
from cycler import cycler
import os
import numpy.polynomial.polynomial as poly


mpl.style.use('default')


Params_JCP = {  # setup matplotlib to use latex for output
    "text.usetex": False,  # use LaTeX to write all text
    "font.family": "sans-serif",
    "font.sans-serif":"Arial",
    "font.size": 8,
    "axes.labelsize": 8, 
    "axes.titlesize": 8,
    "axes.labelweight" :'bold',
    "legend.fontsize": 8, 
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    'figure.titlesize': 8,
    "figure.figsize": [3.37,3.37/1.618],  # default fig size of 1 textwidth
    "figure.dpi" : 100 ,
    "figure.autolayout": True,
    "lines.markersize": 4,
    "lines.linewidth": .5,
    "lines.markeredgewidth": 0.3,
    "axes.linewidth": .6,
    "figure.facecolor": "white",
    "axes.formatter.useoffset": False,
    "errorbar.capsize": 2,
    "axes.prop_cycle" : cycler(color=colors)
}

#mpl.rcParams.update(Params_JCP)

plt.rcParams.update({'figure.autolayout': True})
filefolder=conv.BrukerListFile(PATH)
data=conv.readEPR(filefolder[28])
x,y,z,par=data
#plt.plot(x,y)
EPR.plotFFT2D(x,z[0],par=par, xmin=par['YMIN']/10, xmax=(par['YMIN']+par['YWID'])/10, 
              cmap='RdYlBu_r', vmax=1e5, xlab = 'Field (mT)')
plt.gca()
plt.ylim(0,4)
plt.show()