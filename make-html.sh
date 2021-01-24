#!/bin/bash
OUTDIR=covid-cases

# cd to this dir
cd /Users/ssevova/Documents/covid/
# Remove the old data file
rm owid-covid-data.csv
# Grab the new data file
/usr/local/bin/wget https://covid.ourworldindata.org/data/owid-covid-data.csv
# Make the plots and html
python plotCovidCases.py -i /Users/ssevova/Documents/covid/owid-covid-data.csv -o ${OUTDIR} > /Users/ssevova/Documents/covid_cron.log
# Copy to AWS S3 bucket
/usr/local/bin/aws s3 cp ${OUTDIR} s3://www.covid19studies.website/ --recursive

