#!/usr/bin/env python
"""
Plot the new daily COVID-19 cases in different countries
"""
__author__ = "Stanislava Sevova"
###############################################################################
# Import libraries
################## 
import argparse
import sys
import os
import re
import glob
import shutil
import uproot as up
import uproot_methods
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
from plotUtils import *
###############################################################################
# Command line arguments
######################## 
def getArgumentParser():
    """ Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Plotting the daily covid numbers in various countries")
    parser.add_argument('-i',
                        '--infile',
                        dest='infile',
                        help='Input CSV file',
                        default='data-VgjQz.csv')
    parser.add_argument('-o',
                        '--outdir',
                        dest='outdir',
                        help='Output directory for plots, selection lists, etc',
                        default='outdir')
    
    return parser
###############################################################################
def main():
    """ Run script """

    options = getArgumentParser().parse_args()
    
    ### Make output dir
    dir_path = os.getcwd()
    out_dir = options.outdir
    path = os.path.join(dir_path, out_dir)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    df_all = pd.read_csv("owid-covid-data.csv")
    print (df_all.columns)
    df_swiss_official = pd.read_csv(options.infile)
    df_swiss = df_all[df_all['location']=='Switzerland']
    df_bgr   = df_all[df_all['location']=='Bulgaria']
    df_usa   = df_all[df_all['location']=='United States']
    df_can   = df_all[df_all['location']=='Canada']
    df_pol   = df_all[df_all['location']=='Poland']
    
    df_europe = df_all[df_all['continent']=='Europe']
                    
                            
    
    labelDay="Cases / day"
    labelMil="Cases per million / day"
    isLog=False

    make1Dplot(df_swiss_official['new.infections'],"daily_infections",0,len(df_swiss_official.index),labelDay,isLog)
    make1Dplot(df_can['new_cases']  ,"canada_daily_infections",0,len(df_can.index),labelDay,isLog)
    make1Dplot(df_swiss['new_cases'],"swiss_daily_infections",0,len(df_swiss.index),labelDay,isLog)
    make1Dplot(df_bgr['new_cases']  ,"bgr_daily_infections",0,len(df_bgr.index),labelDay,isLog)
    make1Dplot(df_usa['new_cases']  ,"usa_daily_infections",0,len(df_usa.index),labelDay,isLog)

    
    make1Dplot(df_can['new_cases_per_million'],"canada_daily_cases_per_mil",0,len(df_can.index),labelMil,isLog)
    make1Dplot(df_swiss['new_cases_per_million'],"swiss_daily_cases_per_mil",0,len(df_swiss.index),labelMil,isLog)
    make1Dplot(df_bgr['new_cases_per_million'],"bgr_daily_cases_per_mil",0,len(df_bgr.index),labelMil,isLog)
    make1Dplot(df_usa['new_cases_per_million'],"usa_daily_cases_per_mil",0,len(df_usa.index),labelMil,isLog)

    df_can.loc[df_can['new_cases_per_million']<1,'new_cases_per_million']=0.0
    df_usa.loc[df_usa['new_cases_per_million']<1,'new_cases_per_million']=0.0
    df_swiss.loc[df_swiss['new_cases_per_million']<1,'new_cases_per_million']=0.0
    
    make1DplotCompare(df_can['new_cases_per_million'],"Canada",df_usa['new_cases_per_million'],"USA","can_v_usa_per_mil",labelMil,isLog)
    make1DplotCompare(df_swiss['new_cases_per_million'],"Switzerland",df_usa['new_cases_per_million'],"USA","swiss_v_usa_per_mil",labelMil,isLog)
    make1DplotCompare(df_swiss['new_cases_per_million'],"Switzerland",df_can['new_cases_per_million'],"Canada","swiss_v_can_per_mil",labelMil,isLog)

    
if __name__ == '__main__':
    main()
