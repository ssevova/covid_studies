#!/usr/bin/env python
""" Utils for making plots out of numpy arrays """
import os
import shutil
import glob
import math
import datetime
import numpy as np
import numpy.lib.recfunctions as recfn
import pandas as pd
# Matplotlib                                                                                  
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as tick
###############################################################################  
def makeOverlayPlot(df_array,names,col,ylabel,xlabel,isLog):

    if col == 'cases':
        colName = 'new_cases_smoothed_per_100k'
    elif col == 'vax100k':
        colName = 'new_vaccinations_smoothed_per_100k'
    elif col == 'vaxFull':
        colName = 'vax_per_pop'
    elif col == 'vaxOne':
        colName = 'min_vax_per_pop'
        
    fig, ax1 = plt.subplots(1,1)  

    for name,df in zip(names,df_array):
        if name == 'EU': continue
        if name == 'CHE':
            latest_swiss_ = df[colName].iloc[-1]
        days = len(np.array(df[colName]))
        if col == 'cases':
            start = 0
            ls = '-'
        else:
            start = 310
            df = df.iloc[310:]
            ls = '.-'
            
        df = df.replace(0,np.nan)
        ax1.plot(range(start,days), np.array(df[colName]), ls, linewidth=0.75, label=name)

    if isLog:
        ax1.set_ylim(10E0,latest_swiss_+1000)
        ax1.set_yscale('log')
    else:
        ax1.set_ylim(bottom=0.0001)

    ax1.set_xlim(start,days+10)        
    ax1.tick_params(direction='in', left=True, right=True)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    #ax1.hlines(latest_swiss_cases+60,0,days,colors='cornflowerblue',linestyle='--',linewidth=0.85,label='CHE+60')
    ax1.legend()

    if isLog:
        plt.savefig('all_'+colName+'_log.pdf')
        print('saved all_'+colName+'_log.pdf')
    else:
        if xlabel == '2 week period':
            plt.savefig('2weeks_cases_per_100k.pdf')
            print('saved 2weeks_cases_per_100k.pdf')
        else:
            plt.savefig('all_'+colName+'.pdf')
            print('saved all_'+colName+'.pdf')

    
def make1DplotCompare(arr1,arr1Label,arr2,arr2Label,hname,ylabel,isLog): 
    """Plot a histogram with error bars."""

    print('Length of arr1: {}'.format(len(np.array(arr1))))
    print('Length of arr2: {}'.format(len(np.array(arr2))))

    if len(np.array(arr1)) > len(np.array(arr2)):
        n = len(np.array(arr1)) - len(np.array(arr2))
        n_arr1 = arr1[:-n]
        n_arr2 = arr2
    elif len(np.array(arr2)) > len(np.array(arr1)):
        n = len(np.array(arr2)) - len(np.array(arr1))
        n_arr1 = arr1
        n_arr2 = arr2[:-n]
    else:
        n_arr1 = arr1
        n_arr2 = arr2

    print('Length of n_arr1: {}'.format(len(np.array(n_arr1))))
    print('Length of n_arr2: {}'.format(len(np.array(n_arr2))))

    fig = plt.figure(figsize=(9, 6),dpi=100)
    gs = gridspec.GridSpec(7, 1, hspace=0.0, wspace=0.0)
    ax1 = fig.add_subplot(gs[0:5])
    ax1.tick_params(direction='in',left=True, right=True,labelbottom=False)
    ax2 = fig.add_subplot(gs[5:7], sharex=ax1)
    ax2.tick_params(direction='in',left=True, right=True)
    ax2.yaxis.set_major_locator(tick.LinearLocator(numticks=5))
    ax2.xaxis.set_major_locator(tick.MaxNLocator(symmetric=True, prune=None, min_n_ticks=6, nbins=6))
    ax2.autoscale(axis="x", tight=True)

    ax1.bar(range(0,len(np.array(n_arr1))),
            np.array(n_arr1),
            color='r',
            width=1.0,
            alpha=0.5,
            label=arr1Label,
            zorder=-1)
    
    ax1.bar(range(0,len(np.array(n_arr2))),
            np.array(n_arr2),
            color='b',
            width=1.0,
            alpha=0.5,
            label=arr2Label,
            zorder=2)
    
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel('Days')
    ax1.set_ylim(0,max(max(np.array(n_arr1)),max(np.array(n_arr2)))+20)
    ax1.set_xlim(0,len(np.array(n_arr2)))
    ax1.legend()
    
    ratio = getRatio(np.array(n_arr1),np.array(n_arr2))
    bin_centers = range(0,len(np.array(n_arr2)))

    ax2.plot(bin_centers, ratio, color='black', marker='.')
    
    ax2.set_xlim(0,len(np.array(n_arr2)))
    ax2.set_ylim(0,2)
    ax2.set_ylabel("Ratio")
    ax2.set_xlabel("Days")    

    if isLog==True:
        ax1.set_yscale('log')
        ax1.set_ylim(0.01,2*max(np.array(n_arr1)))
        pltname = '{}_log.pdf'.format(hname)
    else:
        ax1.set_yscale('linear')
        pltname = '{}.pdf'.format(hname)

    plt.savefig(pltname)       
    print('saved {}'.format(pltname))
#-----------------------------------------------------
def make1Dplot(arr1,xname,xmin,xmax,ylabel,isLog): 
    """Plot a histogram with error bars."""
    #print(np.array(arr1))

    fig, ax1 = plt.subplots(1,1)    

    if xname == "swiss_daily_infections":
        ar1 = np.array(arr1)
        friends = np.zeros(shape=len(ar1))
        
        friends[232] = 2
        ar1[232] = ar1[232]-friends[232]
        
        ax1.bar(range(xmin,xmax), friends, color='#FE01B1', width=1, label='')
        ax1.bar(range(xmin,xmax), ar1, color='g', width=1, label='',bottom=friends)
    else:
        
        ax1.bar(range(xmin,xmax),
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
        ax1.set_ylim(0.01,2*max(np.array(arr1)))
        pltname = '{}_log.pdf'.format(xname)
    else:
        ax1.set_yscale('linear')
        pltname = '{}.pdf'.format(xname)

    plt.savefig(pltname)       
    print('saved {}'.format(pltname))
#-----------------------------------------------------
def make1DplotSIR(arr1,t,I,xname,xmin,xmax,ylabel,isLog): 
    """Plot a histogram with error bars."""
    print(np.array(arr1))

    fig, ax1 = plt.subplots(1,1)    

    ax1.bar(range(xmin,xmax),
            np.array(arr1),
            color='b',
            alpha=0.5,
            width=1,
            label='')
    ax1.plot(t,I,'r',alpha=0.8,lw=2,label='Infected, beta=0.2, gamma=1/10')
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel('Days')
    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(0,max(I)+20)
    ax1.legend()
    
    if isLog==True:
        ax1.set_yscale('log')
        pltname = '{}_log.png'.format(xname)
    else:
        pltname = '{}.pdf'.format(xname)


    plt.savefig(pltname)       
    print('saved {}'.format(pltname))
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
def makeHTML(outFileName,title):

    plots = glob.glob('*.pdf')

    with open(outFileName, 'w') as outFile:
        # write HTML header
        outFile.write("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <title>covid-19 cases</title>
        <link href="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
        <script src="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js"></script>
        <script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
        </head>
        <body>
        <div class="container">
        <h1> covid-19 cases </h1> 
        <p>Last updated: {date}</p> 
        </div>
        """.format(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        outFile.write("<h2> Cases per 100k </h2>")
        outFile.write('<table style="width:100%">')
        outFile.write("<tr>\n")
        outFile.write("<td width=\"25%\"><a target=\"_blank\" href=\"all_new_cases_smoothed_per_100k.pdf\"><img src=\"all_new_cases_smoothed_per_100k.pdf\" alt=\"all_new_cases_smoothed_per_100k.pdf\" width=\"100%\"></a></td>\n")         
        #outFile.write("<td width=\"25%\"><a target=\"_blank\" href=\"all_new_cases_smoothed_per_100k_log.pdf\"><img src=\"all_new_cases_smoothed_per_100k_log.pdf\" alt=\"all_new_cases_smoothed_per_100k_log.pdf\" width=\"100%\"></a></td>\n")
        outFile.write("<td width=\"25%\"><a target=\"_blank\" href=\"2weeks_cases_per_100k.pdf\"><img src=\"2weeks_cases_per_100k.pdf\" alt=\"2weeks_cases_per_100k.pdf\" width=\"100%\"></a></td>\n") 
        outFile.write("</tr>\n")
        outFile.write("</table>\n")

        outFile.write("<h2> Vaccinations </h2>")
        outFile.write('<table style="width:100%">')
        outFile.write("<tr>\n")
        outFile.write("<td width=\"25%\"><a target=\"_blank\" href=\"all_new_vaccinations_smoothed_per_100k.pdf\"><img src=\"all_new_vaccinations_smoothed_per_100k.pdf\" alt=\"all_new_vaccinations_smoothed_per_100k.pdf\" width=\"100%\"></a></td>\n")         
        outFile.write("<td width=\"25%\"><a target=\"_blank\" href=\"all_new_vaccinations_smoothed_per_100k_log.pdf\"><img src=\"all_new_vaccinations_smoothed_per_100k_log.pdf\" alt=\"all_new_vaccinations_smoothed_per_100k_log.pdf\" width=\"100%\"></a></td>\n")
        outFile.write("</tr>\n")
        
        outFile.write("<tr>\n")
        outFile.write("<td width=\"25%\"><a target=\"_blank\" href=\"all_vax_per_pop.pdf\"><img src=\"all_vax_per_pop.pdf\" alt=\"all_vax_per_pop.pdf\" width=\"100%\"></a></td>\n")
        outFile.write("<td width=\"25%\"><a target=\"_blank\" href=\"all_min_vax_per_pop.pdf\"><img src=\"all_min_vax_per_pop.pdf\" alt=\"all_min_vax_per_pop.pdf\" width=\"100%\"></a></td>\n")
        outFile.write("</tr>\n")
        outFile.write("</table>\n")
        
        clist = ['usa','can','swiss','france','spain','bgr','pol','por','uk','gr']
        cdict = {
            'usa' : 'United States',
            'can' : 'Canada',
            'swiss' : 'Switzerland',
            'france' : 'France',
            'spain' : 'Spain',
            'bgr' : 'Bulgaria',
            'pol' : 'Poland',
            'por' : 'Portugal',
            'uk': 'United Kingdom',
            'gr': 'Greece'
        }
        fdict = {
            'usa' : 'us',
            'can' : 'ca',
            'swiss' : 'ch',
            'france' : 'fr',
            'spain' : 'es',
            'bgr' : 'bg',
            'pol' : 'pl',
            'por' : 'pt',
            'uk'  : 'gb',
            'gr'  : 'gr'
        }
        for c in clist:
            plots = glob.glob(c+'*.pdf')
            outFile.write("<h2> {country} <img src='https://www.countryflags.io/{flag}/shiny/64.png'> </h2>".format(country=cdict[c], flag=fdict[c]))
            outFile.write('<table style="width:100%">')
            for i in range(0,len(plots)):
                offset = 2
                if i==0 or i%3==0: 
                    outFile.write("<tr>\n")
                outFile.write("<td width=\"25%\"><a target=\"_blank\" href=\"" + plots[i] + "\"><img src=\"" + plots[i] + "\" alt=\"" + plots[i] + "\" width=\"100%\"></a></td>\n")
                if (i>offset and (i-offset)%3==0) or i==len(plots): 
                    outFile.write("</tr>\n")

            outFile.write("</table>\n")
        outFile.write("</body>\n")
        outFile.write("</html>")
        
