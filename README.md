# AlgorithmicDrift
1) Configure the run settings in main.py (model, delta, gamma, eta)
2) Run main.py

Configuration details:
1) module:
   1) training, to train models
   2) evaluation, to evaluate pre-trained models
   3) generation, to run the simulation
2) strategy: No_strategy/Organic

Example of proportions: 0.2_0.6_0.2 -> 20% non-radicalized, 60% semi-radicalized, 20% radicalized

Dependencies:
Python 3.9 is recommended.

Please, for a correct execution of this code, install the following packages in this order:
- joblib==1.4.2
- kmeans-pytorch==0.3
- tqdm==4.67.1
- torch==1.11.0
- matplotlib==3.8.3
- scikit-learn==1.6.1
- numpy==1.22.3
- pandas==1.4.2
- scipy==1.8.0
- recbole==1.0.1 --no-dependencies
- colorlog==6.9.0
- tensorboard==2.18.0
- colorama==0.4.6
- pyyaml==6.0.2
- igraph==0.9.11
- pycairo==1.27.0
- networkx==3.4.2
- seaborn==0.13.2

