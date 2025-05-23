📊 Comparison Experiments
Each comparison experiment takes about 30 minutes to run.

fixed Strategy
```shell
# MNIST with fixed strategy
python -m src.experiments.verify_incentives.fixed.verify_incentives-fixed-MNIST --adjustment_literation -1 --parent_path log-verify_incentives
```

PGI-RDFL(Our Method)
```shell
# MNIST with QD-RDFL
python -m src.experiments.verify_incentives.pgi_rdfl.verify_incentives-pgi_rdfl-MNIST --adjustment_literation -1 --parent_path log-verify_incentives
```

random Strategy
```shell
# MNIST with random strategy
python -m src.experiments.verify_incentives.random.verify_incentives-random-MNIST --adjustment_literation -1 --parent_path log-verify_incentives
```