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
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import math
import glob
from plotUtils import *
plt.set_cmap("Set3")
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
    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        help='Increase output verbosity',
                        action="store_true")
    return parser
###############################################################################
def getCases100k(name,df):
    df['new_cases_smoothed_per_100k'] = (df.new_cases_smoothed/df.population.iloc[-1])*100000
    df['new_vaccinations_smoothed_per_100k'] = (df.new_vaccinations_smoothed/df.population.iloc[-1])*100000
    
def getTwoWeekTotCases(name,df):
    getCases100k(name,df)
    m_df = df['new_cases_smoothed_per_100k']
    m_df = m_df.reset_index()
    d = {'index':'last', 'new_cases_smoothed_per_100k':'sum'}
    
    res = m_df.groupby(m_df.index // 14).agg(d)
    return res

def getVaxPerPop(name,df):
    df['vax_per_pop'] = df.people_fully_vaccinated/df.population
    df['min_vax_per_pop'] = df.people_vaccinated/df.population
    
def getPercentPositive(df):
    df['positive_rate'] = df.new_cases/df.new_tests
    df['positive_rate'] = df['positive_rate'].rolling(7).mean()
    df['hosp_patients_per_case'] = df.hosp_patients/df.new_cases

def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt
    
def alignDF(inputDF,targetDF):
    inputDF = pd.merge(targetDF, inputDF, how='outer', on='date').fillna(0)

    inputDF = inputDF.rename(columns={"new_cases_smoothed_y": "new_cases_smoothed",
                                      "positive_rate_y":"positive_rate",
                                      "tests_per_case_y":"tests_per_case",
                                      "total_boosters_y":"total_boosters",
                                      "total_boosters_per_hundred_y":"total_boosters_per_hundred",
                                    "new_cases_smoothed_per_million_y": "new_cases_smoothed_per_million",
                                    "total_vaccinations_y":"total_vaccinations",
                                    "new_vaccinations_smoothed_y":"new_vaccinations_smoothed",
                                    "new_deaths_smoothed_y": "new_deaths_smoothed",
                                    "new_deaths_smoothed_per_million_y":"new_deaths_smoothed_per_million",
                                    "population_y":"population",
                                    "location_y":"location",
                                    "people_fully_vaccinated_y":"people_fully_vaccinated",
                                    "people_vaccinated_y":"people_vaccinated"})
    inputDF = inputDF.fillna(0)

    return inputDF
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
    os.chdir(path)

    df_all = pd.read_csv(options.infile)
    print(df_all.columns)
    getPercentPositive(df_all)
    df_all = df_all.fillna(0)
    df_swiss  = df_all[df_all['location']=='Switzerland']

    if options.verbose:
        print('Earliest date from df_swiss: {}'.format(df_swiss['date'].iloc[0]))
        print('Latest date from df_swiss: {}'.format(df_swiss['date'].iloc[-1]))
    df_por    = df_all[df_all['location']=='Portugal']
    df_bgr    = df_all[df_all['location']=='Bulgaria']
    df_usa    = df_all[df_all['location']=='United States']
    if options.verbose:
        print('Earliest date from df_usa: {}'.format(df_usa['date'].iloc[0]))
        print('Latest date from df_usa: {}'.format(df_usa['date'].iloc[-1]))
    df_can    = df_all[df_all['location']=='Canada']
    df_pol    = df_all[df_all['location']=='Poland']
    df_fr = df_all[df_all['location']=='France']
    df_uk     = df_all[df_all['location']=='United Kingdom']
    df_esp  = df_all[df_all['location']=='Spain']
    df_gr     = df_all[df_all['location']=='Greece']
    df_it     = df_all[df_all['location']=='Italy']
    df_ru     = df_all[df_all['location']=='Russia']

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
    
    df_eu=pd.DataFrame(df_cen_europe.groupby('date')['new_cases',
                                                     'new_deaths',
                                                     'new_cases_smoothed',
                                                     'new_deaths_smoothed',
                                                     'new_tests_smoothed',
                                                     'new_vaccinations',
                                                     'new_vaccinations_smoothed'].sum())

    df_eu['new_cases_per_million'] = df_eu['new_cases'].truediv((totpop/1000000))
    df_eu['new_cases_smoothed_per_million'] = df_eu['new_cases_smoothed'].dropna().truediv((totpop/1000000))
    df_eu['new_deaths_per_million'] = df_eu['new_deaths'].truediv((totpop/1000000))
    df_eu['new_deaths_smoothed_per_million'] = df_eu['new_deaths_smoothed'].dropna().truediv((totpop/1000000))
  
    ### Align DFs of countries w/ less stats to USA DF
    df_usa = df_usa.fillna(0)
    df_swiss = alignDF(df_swiss,df_usa)
    df_can = alignDF(df_can,df_usa)
    df_bgr = alignDF(df_bgr,df_usa)
    df_pol = alignDF(df_pol,df_usa)
    df_por = alignDF(df_por,df_usa)
    df_esp = alignDF(df_esp,df_usa)
    df_fr  = alignDF(df_fr,df_usa)
    df_uk  = alignDF(df_uk,df_usa)
    df_gr  = alignDF(df_gr,df_usa)
    df_it  = alignDF(df_it,df_usa)
    df_ru  = alignDF(df_ru,df_usa)

    labelDay = "Cases / day"
    vaxFullDay = "Fraction pop. fully vaccinated / day"
    vaxOneDay = "Fraction pop. w/ min. first vaccine / day"
    labelMil = "Cases per million / day"
    cases100k = "Cases per 100k / day"
    testPos = "Avg Weekly Test Positivity"
    vax100k = "Vaccinations per 100k / day"
    isLog=False
     
    dfs = [df_swiss, df_usa, df_can, df_bgr, df_pol, df_por, df_esp, df_fr, df_uk, df_gr, df_it, df_ru, df_eu]
    names = ['CHE', 'USA', 'CAN', 'BGR', 'POL', 'PRT', 'ESP', 'FRA', 'GBR', 'GRC', 'ITA','RUS', 'EU']
    int_dfs = []
    
    ### Set min cases to 0 (not negative)
    for name,df in zip(names,dfs):
        if name=='EU':
            df.loc[:,'positive_rate']=0.0
            df.loc[:,'total_boosters_per_hundred']=0.0
        if name=='EU':
            df.loc[:,'tests_per_case']=0.0        
        #df.loc[df['positive_rate']<0.0, 'positive_rate']=0.0
        df.loc[df['new_cases_smoothed']<0.0, 'new_cases_smoothed']=0.0
        df.loc[df['new_cases_smoothed_per_million']<0.0, 'new_cases_smoothed_per_million']=0.0
        df.loc[df['new_vaccinations_smoothed']<0.0, 'new_vaccinations_smoothed']=0.0

        if name != 'EU':
            getCases100k(name,df)
            getVaxPerPop(name,df)
            int_dfs.append(getTwoWeekTotCases(name,df))
            
    makeOverlayPlot(dfs, names, 'positive_rate', testPos, 'Days', isLog) 
    makeOverlayPlot(dfs, names, 'total_boosters_per_hundred', 'Booster / 100k', 'Days', isLog)
    makeOverlayPlot(dfs, names, 'tests_per_case', 'Test / Case', 'Days', isLog) 
    makeOverlayPlot(dfs, names, 'cases', cases100k, 'Days', isLog)
    makeOverlayPlot(dfs, names, 'cases', cases100k, 'Days', True)
    makeOverlayPlot(dfs, names, 'vax100k', vax100k, 'Days', isLog)
    makeOverlayPlot(dfs, names, 'vax100k', vax100k, 'Days', True)
    makeOverlayPlot(dfs, names, 'vaxFull', vaxFullDay, 'Days', isLog)
    makeOverlayPlot(dfs, names, 'vaxOne', vaxOneDay, 'Days', isLog)
    #makeOverlayPlot(dfs, names, 'vax', vaxDay, 'Days', True) 
    makeOverlayPlot(int_dfs, names, 'cases', 'Total cases per 100k / 2 weeks', '2 week period', isLog)

    '''
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
    '''
    ### Make cases plots 
    make1Dplot(df_can['new_cases_smoothed']   ,"canada_daily_infections",0,len(df_can.index),labelDay,isLog)
    make1Dplot(df_swiss['new_cases_smoothed'] ,"swiss_daily_infections",0,len(df_swiss.index),labelDay,isLog)
    make1Dplot(df_swiss['new_cases_smoothed'] ,"swiss_daily_infections",0,len(df_swiss.index),labelDay,True)
    #make1DplotSIR(df_swiss['total_cases'],t_Swiss,I_Swiss,"swiss_total_cases_SIR_fit",0,len(df_swiss.index),"Total cases",isLog) 
    make1Dplot(df_bgr['new_cases_smoothed'] ,"bgr_daily_infections",0,len(df_bgr.index),labelDay,isLog)
    make1Dplot(df_usa['new_cases_smoothed']   ,"usa_daily_infections",0,len(df_usa.index),labelDay,isLog)
    #make1DplotSIR(df_usa['total_cases'],t_US,I_US,"usa_total_cases_SIR_fit",0,len(df_usa.index),"Total cases",isLog)
    make1Dplot(df_fr['new_cases_smoothed'],"france_daily_infections",0,len(df_fr.index),labelDay,isLog)
    make1Dplot(df_esp['new_cases_smoothed'] ,"spain_daily_infections",0,len(df_esp.index),labelDay,isLog)
    make1Dplot(df_por['new_cases_smoothed'] ,"portugal_daily_infections",0,len(df_por.index),labelDay,isLog)
    make1Dplot(df_pol['new_cases_smoothed'] ,"poland_daily_infections",0,len(df_pol.index),labelDay,isLog)
    make1Dplot(df_uk['new_cases_smoothed'] ,"uk_daily_infections",0,len(df_uk.index),labelDay,isLog)
    make1Dplot(df_gr['new_cases_smoothed'] ,"gr_daily_infections",0,len(df_gr.index),labelDay,isLog)
    make1Dplot(df_it['new_cases_smoothed'] ,"it_daily_infections",0,len(df_it.index),labelDay,isLog)
    make1Dplot(df_ru['new_cases_smoothed'] ,"ru_daily_infections",0,len(df_ru.index),labelDay,isLog)
    
    ### Make cases per mill plots
    make1Dplot(df_can['new_cases_smoothed_per_million'],"canada_daily_cases_per_mil",0,len(df_can.index),labelMil,isLog)
    make1Dplot(df_swiss['new_cases_smoothed_per_million'],"swiss_daily_cases_per_mil",0,len(df_swiss.index),labelMil,isLog)
    make1Dplot(df_bgr['new_cases_smoothed_per_million'],"bgr_daily_cases_per_mil",0,len(df_bgr.index),labelMil,isLog)
    make1Dplot(df_usa['new_cases_smoothed_per_million'],"usa_daily_cases_per_mil",0,len(df_usa.index),labelMil,isLog)
    make1Dplot(df_fr['new_cases_smoothed_per_million'],"france_daily_cases_per_mil",0,len(df_fr.index),labelMil,isLog)
    make1Dplot(df_esp['new_cases_smoothed_per_million'],"spain_daily_cases_per_mil",0,len(df_esp.index),labelMil,isLog)
    make1Dplot(df_por['new_cases_smoothed_per_million'],"portugal_daily_cases_per_mil",0,len(df_por.index),labelMil,isLog)
    make1Dplot(df_pol['new_cases_smoothed_per_million'],"poland_daily_cases_per_mil",0,len(df_pol.index),labelMil,isLog)
    make1Dplot(df_uk['new_cases_smoothed_per_million'],"uk_daily_cases_per_mil",0,len(df_uk.index),labelMil,isLog)
    make1Dplot(df_gr['new_cases_smoothed_per_million'],"gr_daily_cases_per_mil",0,len(df_gr.index),labelMil,isLog)
    make1Dplot(df_it['new_cases_smoothed_per_million'],"it_daily_cases_per_mil",0,len(df_it.index),labelMil,isLog)
    make1Dplot(df_ru['new_cases_smoothed_per_million'],"ru_daily_cases_per_mil",0,len(df_ru.index),labelMil,isLog)

    make1DplotCompare(df_can['new_cases_smoothed_per_million'],"Canada",df_usa['new_cases_smoothed_per_million'],"USA","can_v_usa_per_mil",labelMil,isLog)
    make1DplotCompare(df_swiss['new_cases_smoothed_per_million'],"Switzerland",df_usa['new_cases_smoothed_per_million'],"USA","swiss_v_usa_per_mil",labelMil,isLog)
    make1DplotCompare(df_swiss['new_cases_smoothed_per_million'],"Switzerland",df_can['new_cases_smoothed_per_million'],"Canada","swiss_v_can_per_mil",labelMil,isLog)
    make1DplotCompare(df_swiss['new_cases_smoothed_per_million'],"Switzerland",df_bgr['new_cases_smoothed_per_million'],"Bulgaria","swiss_v_bgr_per_mill",labelMil,isLog)
    make1DplotCompare(df_swiss['new_cases_smoothed_per_million'],"Switzerland",df_pol['new_cases_smoothed_per_million'],"Poland","swiss_v_pol_per_mill",labelMil,isLog)
    make1DplotCompare(df_swiss['new_cases_smoothed_per_million'],"Switzerland",df_por['new_cases_smoothed_per_million'],"Portugal","swiss_v_por_per_mill",labelMil,isLog)
    make1DplotCompare(df_can['new_cases_smoothed_per_million'],"Canada",df_bgr['new_cases_smoothed_per_million'],"Bulgaria","can_v_bgr_per_mill",labelMil,isLog)
    make1DplotCompare(df_usa['new_cases_smoothed_per_million'],"USA",df_esp['new_cases_smoothed_per_million'],"Spain","usa_v_spain_per_mill",labelMil,isLog)
    make1DplotCompare(df_usa['new_cases_smoothed_per_million'],"USA",df_fr['new_cases_smoothed_per_million'],"France","usa_v_france_per_mill",labelMil,isLog)
    make1DplotCompare(df_usa['new_cases_per_million'],"USA (pop=331,002,647)",df_eu['new_cases_per_million'],"ES+IT+FR+CH+DE+BE+AU+UK (pop=353,410,706)","usa_v_eu_per_mill",labelMil,isLog)
    make1DplotCompare(df_usa['new_cases_smoothed_per_million'],"USA (pop=331,002,647)",df_eu['new_cases_smoothed_per_million'],"ES+IT+FR+CH+DE+BE+AU+UK (pop=353,410,706)","usa_v_eu_smoothed_per_mill","Cases per million / day (smoothed)",isLog)

    make1DplotCompare(df_usa['new_deaths_per_million'],"USA deaths",df_eu['new_deaths_per_million'],"ES+IT+FR+CH+DE+BE+AU+UK deaths","usa_v_eu_deaths_per_mill","Deaths per million / day",isLog)
    make1DplotCompare(df_usa['new_deaths_smoothed_per_million'],"USA deaths",df_eu['new_deaths_smoothed_per_million'],"ES+IT+FR+CH+DE+BE+AU+UK deaths","usa_v_eu_deaths_smoothed_per_mill","Deaths per million / day (smoothed)",isLog)
    
    make1DplotCompare(df_usa['new_cases_smoothed_per_million'],"USA cases",df_usa['new_deaths_smoothed_per_million'],"USA deaths","usa_cases_v_deaths_per_mill",labelMil,True)
    make1DplotCompare(df_can['new_cases_smoothed_per_million'],"CAN cases",df_can['new_deaths_smoothed_per_million'],"CAN deaths","can_cases_v_deaths_per_mill",labelMil,True)
    make1DplotCompare(df_swiss['new_cases_smoothed_per_million'],"SWISS cases",df_swiss['new_deaths_smoothed_per_million'],"SWISS deaths","swiss_cases_v_deaths_per_mill",labelMil,True)
    make1DplotCompare(df_fr['new_cases_smoothed_per_million'],"FRANCE cases",df_fr['new_deaths_smoothed_per_million'],"FRANCE deaths","france_cases_v_deaths_per_mill",labelMil,True)
    make1DplotCompare(df_esp['new_cases_smoothed_per_million'],"SPAIN cases",df_esp['new_deaths_smoothed_per_million'],"SPAIN deaths","spain_cases_v_deaths_per_mill",labelMil,True)
    make1DplotCompare(df_bgr['new_cases_smoothed_per_million'],"BGR cases",df_bgr['new_deaths_smoothed_per_million'],"BGR deaths","bgr_cases_v_deaths_per_mill",labelMil,True)    
    make1DplotCompare(df_pol['new_cases_smoothed_per_million'],"POL cases",df_pol['new_deaths_smoothed_per_million'],"POL deaths","pol_cases_v_deaths_per_mill",labelMil,True)    
    make1DplotCompare(df_por['new_cases_smoothed_per_million'],"POR cases",df_por['new_deaths_smoothed_per_million'],"POR deaths","por_cases_v_deaths_per_mill",labelMil,True)
    make1DplotCompare(df_uk['new_cases_smoothed_per_million'],"UK cases",df_uk['new_deaths_smoothed_per_million'],"UK deaths","uk_cases_v_deaths_per_mill",labelMil,True)
    make1DplotCompare(df_gr['new_cases_smoothed_per_million'],"GR cases",df_gr['new_deaths_smoothed_per_million'],"GR deaths","gr_cases_v_deaths_per_mill",labelMil,True)
    make1DplotCompare(df_it['new_cases_smoothed_per_million'],"IT cases",df_it['new_deaths_smoothed_per_million'],"IT deaths","it_cases_v_deaths_per_mill",labelMil,True)
    make1DplotCompare(df_ru['new_cases_smoothed_per_million'],"RU cases",df_ru['new_deaths_smoothed_per_million'],"RU deaths","ru_cases_v_deaths_per_mill",labelMil,True)

    make1DplotCompare(df_usa['new_tests_smoothed'],"USA tests",df_usa['new_cases_smoothed'],"USA cases","usa_tests_v_cases","Tests / day",isLog)
    make1DplotCompare(df_usa['new_tests_smoothed'],"USA tests",df_eu['new_tests_smoothed'],"ES+IT+FR+CH+DE+BE+AU+UK tests","usa_v_eu_tests","Tests / day",isLog)
    
    makeHTML("covid19_cases.html","COVID-19 plots")
    
if __name__ == '__main__':
    main()
