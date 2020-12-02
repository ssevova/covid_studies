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
#import uproot as up
#import uproot_methods
import pandas as pd
import numpy as np
from scipy.integrate import odeint
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
                        default='/afs/cern.ch/work/s/ssevova/public/covid_studies/owid-covid-data.csv')
    parser.add_argument('-o',
                        '--outdir',
                        dest='outdir',
                        help='Output directory for plots, selection lists, etc',
                        default='outdir')
    return parser
###############################################################################
def getCases100k(name,df):
    if name in ['BGR', 'POL', 'PRT']:
        df['new_cases_smoothed_per_100k'] = (df.new_cases_smoothed_y/df.population_y.iloc[-1])*100000
        print('{country} cases per 100 000: {cases}'.format(country=df.location_y.iloc[-1], cases=df.new_cases_smoothed_per_100k.iloc[-1]))
    else:
        df['new_cases_smoothed_per_100k'] = (df.new_cases_smoothed/df.population.iloc[-1])*100000
        print('{country} cases per 100 000: {cases}'.format(country=df.location.iloc[-1], cases=df.new_cases_smoothed_per_100k.iloc[-1]))


def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

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
    os.chdir(path)

    df_all = pd.read_csv(options.infile)
    print(df_all.columns)

    df_swiss  = df_all[df_all['location']=='Switzerland']
    df_por    = df_all[df_all['location']=='Portugal']
    df_bgr    = df_all[df_all['location']=='Bulgaria']
    df_usa    = df_all[df_all['location']=='United States']
    df_can    = df_all[df_all['location']=='Canada']
    df_pol    = df_all[df_all['location']=='Poland']
    df_france = df_all[df_all['location']=='France']
    df_spain  = df_all[df_all['location']=='Spain']

    df_cen_europe = df_all[(df_all['location']=='Spain') |
                           (df_all['location']=='Italy') |
                           (df_all['location']=='France') |
                           (df_all['location']=='Switzerland') |
                           (df_all['location']=='Germany') |
                           (df_all['location']=='Belgium') |
                           (df_all['location']=='Austria') |
                           (df_all['location']=='United Kingdom')]

    ### Get total pop of central Europe
    totpop=df_cen_europe[df_cen_europe['location']=='Spain'].population.iloc[0]+df_cen_europe[df_cen_europe['location']=='Italy'].population.iloc[0]+df_cen_europe[df_cen_europe['location']=='France'].population.iloc[0]+df_cen_europe[df_cen_europe['location']=='Switzerland'].population.iloc[0]+df_cen_europe[df_cen_europe['location']=='Germany'].population.iloc[0]+df_cen_europe[df_cen_europe['location']=='Belgium'].population.iloc[0]+df_cen_europe[df_cen_europe['location']=='Austria'].population.iloc[0]+df_cen_europe[df_cen_europe['location']=='United Kingdom'].population.iloc[0]
    print('ES+IT+FR+CH+DE+BE+AU+UK population: {}'.format(totpop))
    
    df_eu=pd.DataFrame(df_cen_europe.groupby('date')['new_cases','new_deaths','new_cases_smoothed','new_deaths_smoothed','new_tests_smoothed'].sum())

    df_eu['new_cases_per_million'] = df_eu['new_cases'].truediv((totpop/1000000))
    df_eu['new_cases_smoothed_per_million'] = df_eu['new_cases_smoothed'].dropna().truediv((totpop/1000000))
    df_eu['new_deaths_per_million'] = df_eu['new_deaths'].truediv((totpop/1000000))
    df_eu['new_deaths_smoothed_per_million'] = df_eu['new_deaths_smoothed'].dropna().truediv((totpop/1000000))

    ### Align DFs of countries w/ less stats to Swiss DF
    df_bgr = pd.merge(df_swiss, df_bgr, how='outer', on='date').fillna(0)
    df_pol = pd.merge(df_swiss, df_pol, how='outer', on='date').fillna(0)
    df_por = pd.merge(df_swiss, df_por, how='outer', on='date').fillna(0)
    
    ### Fill NaN with 0
    df_swiss  = df_swiss.fillna(0)
    df_eu     = df_eu.fillna(0)
    df_usa    = df_usa.fillna(0)
    df_can    = df_can.fillna(0)
    df_spain  = df_spain.fillna(0)
    df_france = df_france.fillna(0)
    df_pol    = df_pol.fillna(0)
    df_bgr    = df_bgr.fillna(0)
    df_por    = df_por.fillna(0)

    labelDay="Cases / day"
    labelMil="Cases per million / day"
    label100k = "Cases per 100k / day"
    isLog=False
    
    ### Set min cases to 0 (not negative)
    dfs = [df_swiss, df_usa, df_can, df_bgr, df_pol, df_por, df_spain, df_france, df_eu]
    names = ['CHE', 'USA', 'CAN', 'BGR', 'POL', 'PRT', 'ESP', 'FRA', 'EU']
    for name,df in zip(names,dfs):
        if name in ['POL', 'PRT', 'BGR']:
            df.loc[df['new_cases_smoothed_y']<1, 'new_cases_smoothed_y']=0.0
            df.loc[df['new_cases_smoothed_per_million_y']<1, 'new_cases_smoothed_per_million_y']=0.0
        else:
            df.loc[df['new_cases_smoothed']<1, 'new_cases_smoothed']=0.0
            df.loc[df['new_cases_smoothed_per_million']<1, 'new_cases_smoothed_per_million']=0.0

        if name != 'EU':
            getCases100k(name,df)

    makeOverlayPlot(dfs, names, label100k,isLog)    
    makeOverlayPlot(dfs, names, label100k,True)    
    ######################## Experimental stuff ##########################
    ## SIR fits
    N_US = df_usa.population.iloc[0]
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 1, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0_US = N_US - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    beta, gamma = 0.2, 1./10
    # A grid of time points (in days)
    t_US = np.linspace(0, 225,225)
    # Initial conditions vector
    y0_US = S0_US, I0, R0
    # Integrate the SIR equations over the time grid, t.
    retUS = odeint(deriv, y0_US, t_US, args=(N_US, beta, gamma))
    S_US, I_US, R_US = retUS.T

    N_Swiss = df_swiss.population.iloc[0]
    S0_Swiss = N_Swiss - I0 - R0
    y0_Swiss = S0_Swiss, I0, R0
    t_Swiss = np.linspace(0,225,225)
    retSwiss = odeint(deriv, y0_Swiss, t_Swiss, args=(N_Swiss, beta, gamma))
    S_Swiss, I_Swiss, R_Swiss = retSwiss.T
    #####################################################################

    ### Make cases plots 
    make1Dplot(df_can['new_cases_smoothed']   ,"canada_daily_infections",0,len(df_can.index),labelDay,isLog)
    make1Dplot(df_swiss['new_cases_smoothed'] ,"swiss_daily_infections",0,len(df_swiss.index),labelDay,isLog)
    make1Dplot(df_swiss['new_cases_smoothed'] ,"swiss_daily_infections",0,len(df_swiss.index),labelDay,True)
    #make1DplotSIR(df_swiss['total_cases'],t_Swiss,I_Swiss,"swiss_total_cases_SIR_fit",0,len(df_swiss.index),"Total cases",isLog) 
    make1Dplot(df_bgr['new_cases_smoothed_y'] ,"bgr_daily_infections",0,len(df_bgr.index),labelDay,isLog)
    make1Dplot(df_usa['new_cases_smoothed']   ,"usa_daily_infections",0,len(df_usa.index),labelDay,isLog)
    #make1DplotSIR(df_usa['total_cases'],t_US,I_US,"usa_total_cases_SIR_fit",0,len(df_usa.index),"Total cases",isLog)
    make1Dplot(df_france['new_cases_smoothed'],"france_daily_infections",0,len(df_france.index),labelDay,isLog)
    make1Dplot(df_spain['new_cases_smoothed'] ,"spain_daily_infections",0,len(df_spain.index),labelDay,isLog)
    make1Dplot(df_por['new_cases_smoothed_y'] ,"portugal_daily_infections",0,len(df_por.index),labelDay,isLog)
    make1Dplot(df_pol['new_cases_smoothed_y'] ,"poland_daily_infections",0,len(df_pol.index),labelDay,isLog)
    
    ### Make cases per mill plots
    make1Dplot(df_can['new_cases_smoothed_per_million'],"canada_daily_cases_per_mil",0,len(df_can.index),labelMil,isLog)
    make1Dplot(df_swiss['new_cases_smoothed_per_million'],"swiss_daily_cases_per_mil",0,len(df_swiss.index),labelMil,isLog)
    make1Dplot(df_bgr['new_cases_smoothed_per_million_y'],"bgr_daily_cases_per_mil",0,len(df_bgr.index),labelMil,isLog)
    make1Dplot(df_usa['new_cases_smoothed_per_million'],"usa_daily_cases_per_mil",0,len(df_usa.index),labelMil,isLog)
    make1Dplot(df_france['new_cases_smoothed_per_million'],"france_daily_cases_per_mil",0,len(df_france.index),labelMil,isLog)
    make1Dplot(df_spain['new_cases_smoothed_per_million'],"spain_daily_cases_per_mil",0,len(df_spain.index),labelMil,isLog)
    make1Dplot(df_por['new_cases_smoothed_per_million_y'],"portugal_daily_cases_per_mil",0,len(df_por.index),labelMil,isLog)
    make1Dplot(df_pol['new_cases_smoothed_per_million_y'],"poland_daily_cases_per_mil",0,len(df_pol.index),labelMil,isLog)

    make1DplotCompare(df_can['new_cases_smoothed_per_million'],"Canada",df_usa['new_cases_smoothed_per_million'],"USA","can_v_usa_per_mil",labelMil,isLog)
    make1DplotCompare(df_swiss['new_cases_smoothed_per_million'],"Switzerland",df_usa['new_cases_smoothed_per_million'],"USA","swiss_v_usa_per_mil",labelMil,isLog)
    make1DplotCompare(df_swiss['new_cases_smoothed_per_million'],"Switzerland",df_can['new_cases_smoothed_per_million'],"Canada","swiss_v_can_per_mil",labelMil,isLog)
    make1DplotCompare(df_swiss['new_cases_smoothed_per_million'],"Switzerland",df_bgr['new_cases_smoothed_per_million_y'],"Bulgaria","swiss_v_bgr_per_mill",labelMil,isLog)
    make1DplotCompare(df_swiss['new_cases_smoothed_per_million'],"Switzerland",df_pol['new_cases_smoothed_per_million_y'],"Poland","swiss_v_pol_per_mill",labelMil,isLog)
    make1DplotCompare(df_swiss['new_cases_smoothed_per_million'],"Switzerland",df_por['new_cases_smoothed_per_million_y'],"Portugal","swiss_v_por_per_mill",labelMil,isLog)
    make1DplotCompare(df_can['new_cases_smoothed_per_million'],"Canada",df_bgr['new_cases_smoothed_per_million_y'],"Bulgaria","can_v_bgr_per_mill",labelMil,isLog)
    make1DplotCompare(df_usa['new_cases_smoothed_per_million'],"USA",df_spain['new_cases_smoothed_per_million'],"Spain","usa_v_spain_per_mill",labelMil,isLog)
    make1DplotCompare(df_usa['new_cases_smoothed_per_million'],"USA",df_france['new_cases_smoothed_per_million'],"France","usa_v_france_per_mill",labelMil,isLog)
    make1DplotCompare(df_usa['new_cases_per_million'],"USA (pop=331,002,647)",df_eu['new_cases_per_million'],"ES+IT+FR+CH+DE+BE+AU+UK (pop=353,410,706)","usa_v_eu_per_mill",labelMil,isLog)
    make1DplotCompare(df_usa['new_cases_smoothed_per_million'],"USA (pop=331,002,647)",df_eu['new_cases_smoothed_per_million'],"ES+IT+FR+CH+DE+BE+AU+UK (pop=353,410,706)","usa_v_eu_smoothed_per_mill","Cases per million / day (smoothed)",isLog)

    make1DplotCompare(df_usa['new_deaths_per_million'],"USA deaths",df_eu['new_deaths_per_million'],"ES+IT+FR+CH+DE+BE+AU+UK deaths","usa_v_eu_deaths_per_mill","Deaths per million / day",isLog)
    make1DplotCompare(df_usa['new_deaths_smoothed_per_million'],"USA deaths",df_eu['new_deaths_smoothed_per_million'],"ES+IT+FR+CH+DE+BE+AU+UK deaths","usa_v_eu_deaths_smoothed_per_mill","Deaths per million / day (smoothed)",isLog)
    
    make1DplotCompare(df_usa['new_cases_smoothed_per_million'],"USA cases",df_usa['new_deaths_smoothed_per_million'],"USA deaths","usa_cases_v_deaths_per_mill",labelMil,True)
    make1DplotCompare(df_can['new_cases_smoothed_per_million'],"CAN cases",df_can['new_deaths_smoothed_per_million'],"CAN deaths","can_cases_v_deaths_per_mill",labelMil,True)
    make1DplotCompare(df_swiss['new_cases_smoothed_per_million'],"SWISS cases",df_swiss['new_deaths_smoothed_per_million'],"SWISS deaths","swiss_cases_v_deaths_per_mill",labelMil,True)
    make1DplotCompare(df_france['new_cases_smoothed_per_million'],"FRANCE cases",df_france['new_deaths_smoothed_per_million'],"FRANCE deaths","france_cases_v_deaths_per_mill",labelMil,True)
    make1DplotCompare(df_spain['new_cases_smoothed_per_million'],"SPAIN cases",df_spain['new_deaths_smoothed_per_million'],"SPAIN deaths","spain_cases_v_deaths_per_mill",labelMil,True)    
    make1DplotCompare(df_bgr['new_cases_smoothed_per_million_y'],"BGR cases",df_bgr['new_deaths_smoothed_per_million_y'],"BGR deaths","bgr_cases_v_deaths_per_mill",labelMil,True)    
    make1DplotCompare(df_pol['new_cases_smoothed_per_million_y'],"POL cases",df_bgr['new_deaths_smoothed_per_million_y'],"POL deaths","pol_cases_v_deaths_per_mill",labelMil,True)    
    make1DplotCompare(df_por['new_cases_smoothed_per_million_y'],"POR cases",df_bgr['new_deaths_smoothed_per_million_y'],"POR deaths","por_cases_v_deaths_per_mill",labelMil,True)    

    make1DplotCompare(df_usa['new_tests_smoothed'],"USA tests",df_usa['new_cases_smoothed'],"USA cases","usa_tests_v_cases","Tests / day",isLog)
    make1DplotCompare(df_usa['new_tests_smoothed'],"USA tests",df_eu['new_tests_smoothed'],"ES+IT+FR+CH+DE+BE+AU+UK tests","usa_v_eu_tests","Tests / day",isLog)

    makeHTML("covid19_cases.html","COVID-19 plots")
    
if __name__ == '__main__':
    main()
