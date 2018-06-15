#!/bin/bash
#
# script to run the analysis from start to finish
# currently just downloads data from UCI and does a little formatting
#
# Usage:
#   - bash run.sh

bash src/download_data.sh data/urls.txt data/raw

python src/format_data.py data/raw/housing.data data/clean/ data/housing-names.txt --response=MEDV

python src/format_data.py data/raw/yacht_hydrodynamics.data data/clean/ data/yacht-names.txt --response=resistance

# this does not work - formatting of auto-mpg is super irritating
#python src/format_data.py data/raw/auto-mpg.data data/clean/ '\t' data/auto-names.txt --response=mpg --drop='car-name'



#python src/synthetic-comparison.py --filename=data/simulation_results --num_sim=20 --num_iter=100

#python src/fig-5-generator.py --inputfile=data/simulation_results.pickle --outputfile=aggregate_results90 --target=0.9

#python src/fig-5-generator.py --inputfile=data/simulation_results.pickle --outputfile=aggregate_results95 --target=0.95

#python src/fig-5-generator.py --inputfile=data/simulation_results.pickle --outputfile=aggregate_results99 --target=0.99