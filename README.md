📊 Verify Incentives Experiments
Each verify incentives experiment takes about 30 minutes to run.

fixed Strategy

```shell
# MNIST with fixed strategy -utility
python -m src.experiments.verify_incentives.fixed.verify_incentives-fixed-utility-MNIST --adjustment_literation -1 --parent_path log-verify_incentives-utility
```

PGI-RDFL(Our Method)

```shell
# MNIST with PGI-RDFL -utility
python -m src.experiments.verify_incentives.pgi_rdfl.verify_incentives-pgi_rdfl-utility-MNIST --adjustment_literation -1 --parent_path log-verify_incentives-utility
# MNIST with PGI-RDFL -accurancy
python -m src.experiments.verify_incentives.pgi_rdfl.verify_incentives-pgi_rdfl-accurancy-MNIST --adjustment_literation 99 --parent_path log-verify_incentives-accurancy
```

random Strategy

```shell
# MNIST with random strategy -utility
python -m src.experiments.verify_incentives.random.verify_incentives-random-utility-MNIST --adjustment_literation -1 --parent_path log-verify_incentives-utility
```