#!/bin/bash
OUTDIR=covid-cases

# cd to this dir
cd /afs/cern.ch/work/s/ssevova/public/covid_studies/
# Remove the old data file
rm owid-covid-data.csv
# Grab the new data file
wget https://covid.ourworldindata.org/data/owid-covid-data.csv
# Make the plots and html
python plotCovidCases.py -o ${OUTDIR}
# Copy to EOS space 
cp -r ${OUTDIR} /eos/user/s/ssevova/www/

