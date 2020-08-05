#!/usr/bin/env python
""" Utils for making plots out of numpy arrays """
import os
import shutil
import glob
import math

import numpy as np
import numpy.lib.recfunctions as recfn

# Matplotlib                                                                                  
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as tick
###############################################################################  
def make1DplotCompare(arr1,arr1Label,arr2,arr2Label,hname,ylabel,isLog): 
    """Plot a histogram with error bars."""
    print(np.array(arr1))
    print(np.array(arr2))

    fig = plt.figure(figsize=(9, 6),dpi=100)
    gs = gridspec.GridSpec(7, 1, hspace=0.0, wspace=0.0)
    ax1 = fig.add_subplot(gs[0:5])
    ax1.tick_params(labelbottom=False)
    ax2 = fig.add_subplot(gs[5:7], sharex=ax1)
    ax2.yaxis.set_major_locator(tick.LinearLocator(numticks=5))
    ax2.xaxis.set_major_locator(tick.MaxNLocator(symmetric=True, prune='both', min_n_ticks=5, nbins=4))
    ax2.autoscale(axis="x", tight=True)

    ax1.bar(range(0,len(np.array(arr1))),
            np.array(arr1),
            color='r',
            width=1.0,
            alpha=0.5,
            label=arr1Label,
            zorder=-1)
    
    ax1.bar(range(0,len(np.array(arr2))),
            np.array(arr2),
            color='b',
            width=1.0,
            alpha=0.5,
            label=arr2Label,
            zorder=2)
    
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel('Days')
    ax1.set_ylim(0,max(max(np.array(arr1)),max(np.array(arr2)))+20)
    ax1.set_xlim(0,len(np.array(arr2)))
    ax1.legend()
    
    ratio = getRatio(np.array(arr1),np.array(arr2))
    print(len(np.array(arr1)))
    print(len(np.array(arr2)))
    print(len(ratio))
    print (ratio)
    bin_centers = range(0,len(np.array(arr2)))

    ax2.plot(bin_centers, ratio, color='black', marker='.')
    
    ax2.set_xlim(0,len(np.array(arr2)))
    ax2.set_ylim(0,2)
    ax2.set_ylabel("Ratio")
    ax2.set_xlabel("Days")    

    if isLog==True:
        ax1.set_yscale('log')
        pltname = '{}_log.png'.format(hname)
    else:
        ax1.set_yscale('linear')
        pltname = '{}.pdf'.format(hname)

    plt.savefig(pltname)       
    print('saved {}'.format(pltname))
    plt.clf()
#-----------------------------------------------------
def make1Dplot(arr1,xarr,xname,xmin,xmax,ylabel,isLog): 
    """Plot a histogram with error bars."""
    print(np.array(arr1))
    print(np.array(xarr))

    fig, ax1 = plt.subplots(1,1)    

    ax1.bar(np.array(xarr),
            np.array(arr1),
            color='g',
            width=1,
            label='')
    
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel('Days')
    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(0,max(np.array(arr1))+20)

    ax1.hlines([100,200],xmin,xmax,colors='grey',linestyles='--')

    
    if isLog==True:
        ax1.set_yscale('log')
        pltname = '{}_log.png'.format(xname)
    else:
        pltname = '{}.pdf'.format(xname)


    plt.savefig(pltname)       
    print('saved {}'.format(pltname))
    plt.clf()
#-----------------------------------------------------

def make2Dplot(arr1,arr2,xname,yname):
    arr1Flat, arr2Flat = flatten_array(arr1, arr2)
    plt.hist2d(arr1Flat,arr2Flat,100,norm=mcolors.LogNorm())
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.colorbar()
    plt.show()    
    pltname = '{}_vs_{}.png'.format(xname,yname)
    plt.savefig(pltname)
    print('saved {}'.format(pltname))
    plt.clf()
#-----------------------------------------------------
def getRatio(arr1, arr2):
    if len(arr1) != len(arr2):
        print ("Can't make a ratio! Unequal number of bins!")
    bins=[]
    for b1,b2 in zip(arr1,arr2):
        if b1==0 and b2==0:
            bins.append(1.)
        elif b2==0:
            bins.append(0.)
        else:
            bins.append(float(b1)/float(b2))
    return bins
#-----------------------------------------------------
def error(bins,edges):
    # Just estimate the error as the sqrt of the count
    err = [np.sqrt(x) for x in bins]
    errmin = []
    errmax = []
    for x,err in zip(bins,err):
        errmin.append(x-err/2)
        errmax.append(x+err/2)
    return errmin, errmax
#-----------------------------------------------------
def flatten_array(arr1,arr2):
    if arr1.dtype!=np.dtype('float32'):
        arr1Flat = np.hstack(arr1)
    else: 
        arr1Flat = arr1

    if arr2.dtype!=np.dtype('float32'):
        arr2Flat = np.hstack(arr2)
    else: 
        arr2Flat = arr2
    return arr1Flat, arr2Flat
#-----------------------------------------------------
def makeHTML(outFile,title):

    plots = glob.glob('*.pdf')
    f = open(outFile,"w+")
    f.write("<!DOCTYPE html\n")
    f.write(" PUBLIC \"-//W3C//DTD HTML 3.2//EN\">\n")
    f.write("<html>\n")
    f.write("<head><title>"+ title +" </title></head>\n")
    f.write("<body bgcolor=\"EEEEEE\">\n")
    f.write("<table border=\"0\" cellspacing=\"5\" width=\"100%\">\n")
    for i in range(0,len(plots)):
        offset = 2
        if i==0 or i%3==0: f.write("<tr>\n")
        f.write("<td width=\"25%\"><a target=\"_blank\" href=\"" + plots[i] + "\"><img src=\"" + plots[i] + "\" alt=\"" + plots[i] + "\" width=\"100%\"></a></td>\n")
        if i==offset: 
            f.write("</tr>\n")
        elif (i>offset and (i-offset)%3==0) or i==len(plots): 
            f.write("</tr>\n")
            
    f.write("</table>\n")
    f.write("</body>\n")
    f.write("</html>")
    f.close()
