This directory provides source codes. 

1. Run experiments with regular neural networks models and traditional machine learning models.

* running experiments with RNN `python rnn_experiment.py --mol_prop MDR1 --split_type random --seed 0`

* running experiments with GCN `python gcn_experiment.py --mol_prop MDR1 --split_type random --seed 0`

* running experiments with GIN `python gin_experiment.py --mol_prop MDR1 --split_type random --seed 0`

* running experiments with traditional machine learning models `python mlmo_experiment.py --model_name RF --mol_rep morganBits --mol_prop MDR1 --split_type random --seed 0`

2. Process raw predictions to get prediction performance results.

* calculating prediction performance based on raw predictions `python analyze_performance.py --folder benchmark --action calc`

* aggregating performance metrics based on individual performance metric `python analyze_performance.py --folder benchmark --action agg`

**Note** <br>
To run the experiments, the exact data splits files are needed, which are availalbe as a zip file (around 767MB) upon request.

To run the data analysis code, the raw prediction files are needed, which are availalbe as a zip file (around 1.2 GB) upon request. 