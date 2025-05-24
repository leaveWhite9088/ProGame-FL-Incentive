📊 Comparison Experiments
Each comparison experiment takes about 30 minutes to run.

fixed Strategy

```shell
# MNIST with fixed strategy -utility
python -m src.experiments.verify_incentives.utility.fixed.verify_incentives-fixed-utility-MNIST --adjustment_literation -1 --parent_path log-verify_incentives-utility
# MNIST with fixed strategy -accurancy
python -m src.experiments.verify_incentives.accurancy.fixed.verify_incentives-fixed-accurancy-MNIST --adjustment_literation 99 --parent_path log-verify_incentives-accurancy
```

PGI-RDFL(Our Method)

```shell
# MNIST with PGI-RDFL -utility
python -m src.experiments.verify_incentives.utility.pgi_rdfl.verify_incentives-pgi_rdfl-utility-MNIST --adjustment_literation -1 --parent_path log-verify_incentives-utility
# MNIST with PGI-RDFL -accurancy
python -m src.experiments.verify_incentives.accurancy.pgi_rdfl.verify_incentives-pgi_rdfl-accurancy-MNIST --adjustment_literation 99 --parent_path log-verify_incentives-accurancy
```

random Strategy

```shell
# MNIST with random strategy -utility
python -m src.experiments.verify_incentives.utility.random.verify_incentives-random-utility-MNIST --adjustment_literation -1 --parent_path log-verify_incentives-utility
# MNIST with random strategy -accurancy
python -m src.experiments.verify_incentives.accurancy.random.verify_incentives-random-accurancy-MNIST --adjustment_literation 99 --parent_path log-verify_incentives-accurancy
```