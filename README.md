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
# MNIST with PGI-RDFL -accuracy
python -m src.experiments.verify_incentives.pgi_rdfl.verify_incentives-pgi_rdfl-accuracy-MNIST --adjustment_literation 99 --parent_path log-verify_incentives-accuracy
```

random Strategy

```shell
# MNIST with random strategy -utility
python -m src.experiments.verify_incentives.random.verify_incentives-random-utility-MNIST --adjustment_literation -1 --parent_path log-verify_incentives-utility
```

📊 Verify Selection Experiments
Each verify incentives experiment takes about 30 minutes to run.

non_two_way_selection

```shell
# MNIST with non_two_way_selection -utility
python -m src.experiments.verify_selection.non_two_way_selection.verify_selection-ntws-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility
# MNIST with non_two_way_selection -accuracy
python -m src.experiments.verify_selection.non_two_way_selection.verify_selection-ntws-accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_selection-accuracy
```

random_selection

```shell
# MNIST with random_selection -utility
python -m src.experiments.verify_selection.random_selection.verify_selection-rs-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility
# MNIST with random_selection -accuracy
python -m src.experiments.verify_selection.random_selection.verify_selection-rs-accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_selection-accuracy
```

two_way_selection

```shell
# MNIST with two_way_selection -utility
python -m src.experiments.verify_selection.two_way_selection.verify_selection-tws-utility-MNIST --adjustment_literation -1 --parent_path log-verify_selection-utility
# MNIST with two_way_selection -accuracy
python -m src.experiments.verify_selection.two_way_selection.verify_selection-tws-accuracy-MNIST --adjustment_literation 100 --parent_path log-verify_selection-accuracy
```