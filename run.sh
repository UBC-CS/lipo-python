python src/synthetic-comparison.py --filename=data/simulation_results --num_sim=20 --num_iter=100

python src/fig-5-generator.py --inputfile=data/simulation_results.pickle --outputfile=aggregate_results90 --target=0.9

python src/fig-5-generator.py --inputfile=data/simulation_results.pickle --outputfile=aggregate_results95 --target=0.95

python src/fig-5-generator.py --inputfile=data/simulation_results.pickle --outputfile=aggregate_results99 --target=0.99